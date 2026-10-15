
from pathlib import Path
import subprocess
import imageio_ffmpeg

video = Path("videos/traffic.mp4")

if not video.exists():
    raise FileNotFoundError(
        f"Traffic video not found: {video.resolve()}"
    )

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

command = [
    ffmpeg,
    "-hide_banner",
    "-re",
    "-stream_loop", "-1",
    "-i", str(video.resolve()),
    "-an",
    "-vf", "fps=25",
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-pix_fmt", "yuv420p",
    "-r", "25",
    "-g", "25",
    "-keyint_min", "25",
    "-sc_threshold", "0",
    "-bf", "0",
    "-f", "rtsp",
    "-rtsp_transport", "tcp",
    "rtsp://127.0.0.1:8554/traffic",
]

print("Publishing video to MediaMTX over RTSP/TCP...")
print("URL: rtsp://127.0.0.1:8554/traffic")
print("Press Ctrl+C to stop.")

subprocess.run(command, check=True)
