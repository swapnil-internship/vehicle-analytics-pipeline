from pathlib import Path
import argparse
import csv
import time
import uuid

import cv2
from ultralytics import YOLO

from database import (
    initialize_database,
    save_vehicle_event,
    shutdown_database,
)
from color_classifier import estimate_vehicle_color


# --------------------------------------------------
# Configuration
# --------------------------------------------------

VIDEO_PATH = "videos/traffic.mp4"
OUTPUT_PATH = "outputs/tracked_vehicles.mp4"
EVENTS_PATH = "data/vehicle_events.csv"

MODEL_NAME = "yolo11n.pt"

# COCO IDs: car, motorcycle, bus, truck
VEHICLE_CLASSES = [2, 3, 5, 7]

LINE_POSITION = 0.76
# A small deadband on both sides of the line reduces jitter-based counts.
# The value is a fraction of frame height (0.015 = 1.5% of height).
LINE_DEADBAND = 0.015
CONFIDENCE_THRESHOLD = 0.35
IMAGE_SIZE = 640

RTSP_RECONNECT_ATTEMPTS = 5
RTSP_RECONNECT_DELAY = 2
RTSP_READ_TIMEOUT_MS = 5000
RTSP_OPEN_TIMEOUT_MS = 5000


def is_rtsp_source(source):
    """Return True when the input is an RTSP URL."""
    return source.lower().startswith(("rtsp://", "rtsps://"))


def open_video_source(source):
    """Open a local video file or an RTSP stream with FFmpeg timeouts."""
    if not is_rtsp_source(source):
        video = Path(source)
        if not video.exists():
            raise FileNotFoundError(f"Video not found: {video.resolve()}")
        return cv2.VideoCapture(str(video))

    # Passing timeout parameters at construction is more reliable than
    # setting them after open() on OpenCV builds that support this API.
    params = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_OPEN_TIMEOUT_MS,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, RTSP_READ_TIMEOUT_MS,
    ]
    try:
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
    except (TypeError, cv2.error):
        # Compatibility fallback for older OpenCV builds.
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_OPEN_TIMEOUT_MS)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, RTSP_READ_TIMEOUT_MS)
        return cap


def track_vehicles(
    source=VIDEO_PATH,
    output_path=OUTPUT_PATH,
    events_path=EVENTS_PATH,
):
    """Detect and track vehicles from MP4 or RTSP input."""
    source_is_rtsp = is_rtsp_source(source)
    run_id = str(uuid.uuid4())

    output = Path(output_path)
    events_file = Path(events_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    events_file.parent.mkdir(parents=True, exist_ok=True)

    cap = None
    writer = None
    database_started = False

    total_crossings = 0
    frame_number = 0
    database_errors = 0
    reconnect_attempts = 0

    # Track the last stable side of the line, rather than comparing two
    # adjacent frame centres. Positions inside the deadband do not change
    # a track's stable side, preventing small box jitter from counting.
    stable_side = {}
    counted_ids = set()
    vehicle_colors = {}
    vehicle_types = {}
    direction_counts = {"UP": 0, "DOWN": 0}

    start_time = time.monotonic()
    last_fps_time = start_time
    frames_since_fps_check = 0
    processing_fps = 0.0

    csv_fields = [
        "run_id",
        "track_id",
        "vehicle_type",
        "vehicle_color",
        "timestamp_seconds",
        "direction",
    ]

    try:
        print("Connecting to MySQL database...")
        initialize_database()
        database_started = True

        print(f"Run ID: {run_id}")
        print("Loading YOLO model...")
        model = YOLO(MODEL_NAME)

        print(f"Opening input source: {source}")
        cap = open_video_source(source)
        if not cap.isOpened():
            cap.release()
            cap = None
            raise RuntimeError(f"Could not open input source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            success, first_frame = cap.read()
            if not success:
                raise RuntimeError(
                    "Input opened, but no valid video frame was received."
                )
            height, width = first_frame.shape[:2]
            pending_frame = first_frame
        else:
            pending_frame = None

        counting_line_y = int(height * LINE_POSITION)
        deadband_px = max(2, int(height * LINE_DEADBAND))
        upper_threshold = counting_line_y - deadband_px
        lower_threshold = counting_line_y + deadband_px

        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(
                f"Could not create output video: {output.resolve()}"
            )

        print(f"Input resolution: {width}x{height}")
        print(f"Nominal FPS: {fps:.2f}")
        print(
            f"Counting line: y={counting_line_y}, "
            f"deadband={deadband_px}px"
        )
        print(f"Output video: {output.resolve()}")
        print("Press Ctrl+C to stop the detector.")

        with events_file.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
            csv_writer.writeheader()

            while True:
                if pending_frame is not None:
                    frame = pending_frame
                    pending_frame = None
                    success = True
                else:
                    success, frame = cap.read()

                if not success:
                    if not source_is_rtsp:
                        print("End of input video.")
                        break

                    cap.release()
                    cap = None

                    if reconnect_attempts >= RTSP_RECONNECT_ATTEMPTS:
                        raise RuntimeError(
                            "RTSP reconnection limit reached. "
                            "Check MediaMTX, the publisher, and network."
                        )

                    reconnect_attempts += 1
                    print(
                        f"RTSP read failed. Reconnecting "
                        f"({reconnect_attempts}/"
                        f"{RTSP_RECONNECT_ATTEMPTS})..."
                    )
                    time.sleep(RTSP_RECONNECT_DELAY)

                    cap = open_video_source(source)
                    if not cap.isOpened():
                        cap.release()
                        cap = None
                        continue

                    success, frame = cap.read()
                    if not success:
                        cap.release()
                        cap = None
                        continue

                    # Do not infer a crossing across a stream interruption.
                    # A track must establish a stable side again after recovery.
                    stable_side.clear()
                    reconnect_attempts = 0

                else:
                    reconnect_attempts = 0

                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))

                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=VEHICLE_CLASSES,
                    conf=CONFIDENCE_THRESHOLD,
                    imgsz=IMAGE_SIZE,
                    device="cpu",
                    verbose=False,
                )

                annotated = frame.copy()
                cv2.line(
                    annotated,
                    (0, counting_line_y),
                    (width, counting_line_y),
                    (0, 255, 255),
                    2,
                )
                # Visualize the deadband to make the counting zone clear.
                cv2.line(
                    annotated,
                    (0, upper_threshold),
                    (width, upper_threshold),
                    (180, 180, 0),
                    1,
                )
                cv2.line(
                    annotated,
                    (0, lower_threshold),
                    (width, lower_threshold),
                    (180, 180, 0),
                    1,
                )

                boxes = results[0].boxes
                if boxes is not None and boxes.id is not None:
                    coordinates = boxes.xyxy.cpu().numpy().astype(int)
                    track_ids = boxes.id.cpu().numpy().astype(int)
                    class_ids = boxes.cls.cpu().numpy().astype(int)

                    for box, track_id, class_id in zip(
                        coordinates, track_ids, class_ids
                    ):
                        x1, y1, x2, y2 = box
                        center_y = (y1 + y2) // 2
                        track_id = int(track_id)
                        vehicle_type = model.names[int(class_id)]

                        if track_id not in vehicle_colors:
                            vehicle_colors[track_id] = estimate_vehicle_color(
                                frame, (x1, y1, x2, y2)
                            )
                            vehicle_types[track_id] = vehicle_type

                        vehicle_color = vehicle_colors[track_id]
                        vehicle_type = vehicle_types[track_id]

                        cv2.rectangle(
                            annotated, (x1, y1), (x2, y2), (0, 200, 0), 2
                        )
                        label = (
                            f"{vehicle_type} ID:{track_id} {vehicle_color}"
                        )
                        cv2.putText(
                            annotated,
                            label,
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 200, 0),
                            2,
                        )

                        # Establish/change side only after the vehicle centre
                        # clears the deadband. Crossing direction is determined
                        # by the transition between stable sides.
                        current_stable_side = None
                        if center_y < upper_threshold:
                            current_stable_side = "ABOVE"
                        elif center_y > lower_threshold:
                            current_stable_side = "BELOW"

                        previous_side = stable_side.get(track_id)
                        if current_stable_side is not None:
                            if previous_side is None:
                                stable_side[track_id] = current_stable_side
                            elif current_stable_side != previous_side:
                                direction = (
                                    "DOWN"
                                    if previous_side == "ABOVE"
                                    and current_stable_side == "BELOW"
                                    else "UP"
                                )
                                stable_side[track_id] = current_stable_side

                                if track_id not in counted_ids:
                                    counted_ids.add(track_id)
                                    total_crossings += 1
                                    direction_counts[direction] += 1

                                    timestamp = (
                                        time.monotonic() - start_time
                                        if source_is_rtsp
                                        else frame_number / fps
                                    )
                                    timestamp = round(timestamp, 2)

                                    event = {
                                        "run_id": run_id,
                                        "track_id": track_id,
                                        "vehicle_type": vehicle_type,
                                        "vehicle_color": vehicle_color,
                                        "timestamp_seconds": timestamp,
                                        "direction": direction,
                                    }
                                    csv_writer.writerow(event)
                                    csv_file.flush()

                                    save_vehicle_event(
                                        track_id=track_id,
                                        vehicle_type=vehicle_type,
                                        vehicle_color=vehicle_color,
                                        timestamp_seconds=timestamp,
                                        direction=direction,
                                        run_id=run_id,
                                    )
                                    print(
                                        f"Queued event: run={run_id[:8]}, "
                                        f"ID={track_id}, type={vehicle_type}, "
                                        f"color={vehicle_color}, "
                                        f"direction={direction}, "
                                        f"time={timestamp:.2f}s"
                                    )

                frames_since_fps_check += 1
                now = time.monotonic()
                elapsed = now - last_fps_time
                if elapsed >= 1.0:
                    processing_fps = frames_since_fps_check / elapsed
                    frames_since_fps_check = 0
                    last_fps_time = now

                cv2.rectangle(
                    annotated, (10, 10), (410, 128), (25, 25, 25), -1
                )
                cv2.putText(
                    annotated,
                    f"Run: {run_id[:8]}",
                    (22, 33),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (200, 210, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    f"Crossings: {total_crossings}",
                    (22, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (80, 230, 120),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    f"UP: {direction_counts['UP']}  DOWN: {direction_counts['DOWN']}",
                    (22, 87),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (230, 230, 230),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    f"Processing: {processing_fps:.1f} FPS",
                    (22, 113),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (255, 210, 90),
                    2,
                    cv2.LINE_AA,
                )

                writer.write(annotated)
                frame_number += 1

                if frame_number % 100 == 0:
                    print(
                        f"Processed {frame_number} frames | "
                        f"Crossings: {total_crossings} "
                        f"(UP: {direction_counts['UP']}, "
                        f"DOWN: {direction_counts['DOWN']}) | "
                        f"Processing FPS: {processing_fps:.1f}"
                    )

    except KeyboardInterrupt:
        print("\nStop requested by user.")

    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if database_started:
            database_errors = shutdown_database()

    print("\n" + "=" * 50)
    print("VEHICLE ANALYTICS STOPPED")
    print("=" * 50)
    print(f"Run ID: {run_id}")
    print(f"Frames processed: {frame_number}")
    print(f"Crossing events this run: {total_crossings}")
    print(
        f"Direction totals: UP={direction_counts['UP']}, "
        f"DOWN={direction_counts['DOWN']}"
    )
    print(f"Output video: {output.resolve()}")
    print(f"CSV event log: {events_file.resolve()}")
    print(f"MySQL write failures: {database_errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vehicle analytics for MP4 and RTSP sources"
    )
    parser.add_argument(
        "--source", default=VIDEO_PATH, help="Input MP4 path or RTSP URL"
    )
    parser.add_argument(
        "--output", default=OUTPUT_PATH, help="Annotated output video path"
    )
    parser.add_argument(
        "--events", default=EVENTS_PATH, help="CSV event log path"
    )
    args = parser.parse_args()

    started = time.time()
    try:
        track_vehicles(
            source=args.source,
            output_path=args.output,
            events_path=args.events,
        )
    except Exception as error:
        print(f"\nPipeline failed: {error}")
        raise
    finally:
        elapsed = time.time() - started
        print(f"Total execution time: {elapsed / 60:.2f} minutes")
