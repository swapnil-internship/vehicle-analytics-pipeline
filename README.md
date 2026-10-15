# RoadSight Analytics — Vehicle Detection, Tracking & Traffic Analytics

A computer vision project that detects and tracks vehicles, records line-crossing events, estimates vehicle colours, stores events in MySQL, and presents traffic analytics through a Streamlit dashboard.

## 1. Project Overview

RoadSight Analytics demonstrates a video-based traffic monitoring workflow using YOLO11n, ByteTrack, OpenCV, MySQL, and Streamlit.

The system processes a traffic video, identifies supported vehicle classes, tracks detected vehicles across frames, and records crossing events when tracked vehicles move across a configured virtual line. The results are saved to a CSV file and MySQL database, then displayed through an interactive dashboard.

The project was developed for a CPU-based Windows environment without a dedicated NVIDIA GPU.

### Key Features

- Vehicle detection using YOLO11n.
- Multi-object tracking using ByteTrack.
- Virtual-line crossing detection with direction classification.
- Approximate vehicle-colour estimation using OpenCV and HSV values.
- CSV event logging and MySQL persistence.
- Asynchronous database writes through a background worker.
- Streamlit dashboard with traffic metrics, charts, searchable records, and CSV export.
- H.264 video conversion for browser-compatible playback.
- RTSP video publishing through FFmpeg and MediaMTX for prototype testing.

## 2. Technology Stack

| Component                          | Technology                    |
| ---------------------------------- | ----------------------------- |
| Programming language               | Python                        |
| Object detection                   | Ultralytics YOLO11n           |
| Object tracking                    | ByteTrack                     |
| Video processing                   | OpenCV                        |
| Colour estimation                  | OpenCV and NumPy              |
| Database                           | MySQL                         |
| Dashboard                          | Streamlit                     |
| Charts and analytics               | Plotly and Pandas             |
| Video encoding and RTSP publishing | FFmpeg through imageio-ffmpeg |
| RTSP server                        | MediaMTX                      |

### Implementation Note

The original assignment references NVIDIA DeepStream, PostgreSQL, and React. The current implementation uses YOLO11n with ByteTrack, MySQL, and Streamlit instead. It is **not a DeepStream, PostgreSQL, or React implementation**.

The RTSP workflow has been tested as a prototype, but intermittent decoding errors remain. These technology differences and limitations are documented to accurately represent the submitted project.

## 3. Project Structure

The repository is expected to follow this structure:

```text
vehicle-analytics-pipeline/
├── app.py
├── convert_video.py
├── publish_video.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── vehicle_detector.py
│   ├── database.py
│   └── color_classifier.py
├── videos/
│   └── traffic.mp4                 # Sample input; may be supplied separately
├── outputs/
│   ├── tracked_vehicles.mp4        # Generated
│   └── tracked_vehicles_web.mp4    # Generated
└── data/
    └── vehicle_events.csv          # Generated
```

Generated videos, local datasets, model weights, and other large files may be excluded from Git. Create the `videos/` directory and provide a suitable input video before running the detector.

## 4. Prerequisites

- Windows 10 or Windows 11.
- Python 3.10.
- MySQL Server.
- Git, if cloning the repository.
- A compatible traffic video saved as `videos/traffic.mp4`.
- MediaMTX for RTSP testing.

A dedicated NVIDIA GPU is not required for the current CPU-based implementation. Processing speed depends on the computer, input resolution, and video length.

## 5. Installation

Open PowerShell in the project root.

### Step 1: Create and activate a virtual environment

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks virtual-environment activation, use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

### Step 2: Install dependencies

```powershell
pip install -r requirements.txt
```

If the dependency file is unavailable, install the required packages manually:

```powershell
pip install ultralytics opencv-python numpy streamlit mysql-connector-python pandas plotly imageio-ffmpeg
```

The first execution of YOLO11n may require downloading the model weights. Ensure they are available locally or that the environment has internet access for the initial download.

## 6. Database Configuration

The project uses MySQL to store vehicle-crossing events.

### Step 1: Create the database and event table

Open MySQL Workbench or a MySQL client and execute:

```sql
CREATE DATABASE IF NOT EXISTS vehicle_analytics;

USE vehicle_analytics;

CREATE TABLE IF NOT EXISTS vehicle_events (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    track_id INT NOT NULL,
    vehicle_type VARCHAR(50),
    vehicle_color VARCHAR(50),
    event_timestamp DOUBLE,
    direction VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

The database helper in `src/database.py` manages the `run_id` column used to associate events with detector runs.

If you already have an event table, inspect its schema before making changes. Do not delete existing records simply to set up the project.

### Step 2: Configure database credentials

The application reads these environment variables:

| Variable      | Default             |
| ------------- | ------------------- |
| `DB_HOST`     | `localhost`         |
| `DB_PORT`     | `3306`              |
| `DB_USER`     | `root`              |
| `DB_PASSWORD` | Empty               |
| `DB_NAME`     | `vehicle_analytics` |

For the current PowerShell session, set your credentials as follows:

```powershell
$env:DB_HOST = "localhost"
$env:DB_PORT = "3306"
$env:DB_USER = "root"
$env:DB_PASSWORD = "YOUR_MYSQL_PASSWORD"
$env:DB_NAME = "vehicle_analytics"
```

Replace the example password with your local MySQL password.

The detector and dashboard use the configured credentials to access the database. The dashboard also allows a password to be entered through its sidebar when `DB_PASSWORD` is not set.

**Security:** Never commit real passwords, private credentials, or `.env` files to GitHub.

## 7. Run the Vehicle Detector

Ensure the MySQL service is running, the virtual environment is active, and `videos/traffic.mp4` exists.

Run:

```powershell
python src/vehicle_detector.py
```

The default source is the sample video configured in the detector.

To specify a different input:

```powershell
python src/vehicle_detector.py --source "videos/traffic.mp4"
```

The detector also supports output-path arguments:

```powershell
python src/vehicle_detector.py `
  --source "videos/traffic.mp4" `
  --output "outputs/tracked_vehicles.mp4" `
  --events "data/vehicle_events.csv"
```

### Expected outputs

- `outputs/tracked_vehicles.mp4` — annotated video containing detections and tracking information.
- `data/vehicle_events.csv` — crossing-event records for the current run.
- MySQL `vehicle_events` table — stored crossing events.

The CSV file is opened in write mode, so a new detector run replaces the previous CSV file. MySQL records are stored separately and may remain available across runs.

Vehicle IDs belong to tracker runs and should not be interpreted as permanent identities for physical vehicles.

## 8. Convert the Processed Video

After the detector finishes, convert its output into a browser-compatible video:

```powershell
python convert_video.py
```

The conversion script reads:

```text
outputs/tracked_vehicles.mp4
```

It generates:

```text
outputs/tracked_vehicles_web.mp4
```

The output uses H.264 encoding and settings intended to improve compatibility with browser video players.

Run this conversion again after generating a new annotated video so the dashboard does not display an older preview.

## 9. Launch the Streamlit Dashboard

Run:

```powershell
streamlit run app.py
```

Open the local URL displayed in the terminal.

### Dashboard functionality

- Displays the annotated video when the converted file is available.
- Shows total crossing events and upward/downward counts.
- Summarizes vehicle categories and estimated colours.
- Displays charts for direction, vehicle type, colour, and event timestamps.
- Provides searchable event records.
- Allows event records to be exported as a CSV file.
- Refreshes database-backed monitoring data periodically.

The dashboard reads stored database events. It does not start or restart the detector. Run the detector separately to generate new events.

## 10. RTSP Testing with MediaMTX

The RTSP workflow publishes a video to MediaMTX and allows the detector to attempt processing the stream.

### Terminal 1: Start MediaMTX

Start the MediaMTX executable from its installation directory. For example:

```powershell
.\mediamtx.exe
```

Keep the server running. Its configuration must permit RTSP publishing on port `8554`.

### Terminal 2: Publish the sample video

From the project root, with the virtual environment activated:

```powershell
python publish_video.py
```

The script publishes the video to:

```text
rtsp://127.0.0.1:8554/traffic
```

It loops the source video continuously. Press `Ctrl+C` to stop publishing.

### Terminal 3: Run the detector on the RTSP stream

```powershell
python src/vehicle_detector.py --source "rtsp://127.0.0.1:8554/traffic"
```

Monitor the terminal for frame reception, decoding, and detector messages.

**Current status:** RTSP frame reception and crossing-event detection have been observed during testing. Intermittent FFmpeg/RTP decoding errors remain, so this workflow should be treated as a prototype rather than a fully stabilized real-time pipeline.

## 11. How the Pipeline Works

```text
Traffic Video
     |
     v
YOLO11n Vehicle Detection
     |
     v
ByteTrack Object Tracking
     |
     v
Virtual-Line Crossing Detection
     |
     +------> Approximate Colour Estimation
     |
     v
CSV Event Logging + MySQL Persistence
     |
     v
Streamlit Dashboard and Analytics
```

For RTSP testing, the input can instead be published through FFmpeg to MediaMTX before being consumed by the detector.

## 12. Known Limitations

- NVIDIA DeepStream is not implemented.
- PostgreSQL is replaced by MySQL.
- The dashboard is built with Streamlit rather than React.
- CPU-based inference has limited throughput compared with suitable GPU inference.
- Colour labels are heuristic estimates, not outputs from a trained colour-classification model.
- Shadows, lighting, vehicle appearance, and background pixels may affect colour estimates.
- Missed detections, occlusion, and tracker ID changes may affect crossing-event counts.
- The RTSP workflow can encounter intermittent decoding errors.
- The event count represents recorded crossing events, not necessarily the number of unique physical vehicles.
- Results depend on the input video, camera angle, line position, and detection settings.

## 13. Troubleshooting

**MySQL connection failed**

- Confirm the MySQL service is running.
- Verify the host, port, username, password, and database name.
- Check that the `vehicle_events` table exists and has the expected columns.

**Input video not found**

- Confirm the file exists at `videos/traffic.mp4`.
- Run commands from the project root, or supply an appropriate source path.

**Dashboard video is missing**

- Run the detector first.
- Run `python convert_video.py`.
- Confirm that `outputs/tracked_vehicles_web.mp4` exists.

**RTSP connection or decoding errors**

- Confirm MediaMTX is running.
- Confirm the publisher and detector use the same RTSP path.
- Check that port `8554` is available and the server permits publishing.
- If errors persist, use the recorded-video workflow to demonstrate the main pipeline.

**Missing Python module**

- Activate the correct virtual environment.
- Install dependencies using `pip install -r requirements.txt`.
- Check that the required source files are present in the repository.

## 14. Pre-Submission Checklist

- [x] Source code and required helper modules are included.
- [x] `requirements.txt` contains the dependencies used by the application.
- [x] Database setup and configuration instructions are documented.
- [x] The detector runs on the sample video.
- [x] The annotated video and CSV outputs are generated.
- [x] Crossing events are saved to MySQL.
- [x] The converted video plays in the dashboard.
- [x] The dashboard displays current database records.
- [x] RTSP testing instructions and limitations are documented.
- [x] No passwords, private credentials, or unnecessary large files are committed.
- [x] The repository accurately describes the technologies actually implemented.

## 15. Conclusion

RoadSight Analytics is a resource-conscious vehicle detection and traffic analytics prototype. It combines object detection, multi-object tracking, crossing-event logging, relational database storage, video conversion, and dashboard visualization in a single workflow.

The project demonstrates an end-to-end computer vision application while transparently documenting the differences from the original requested stack and the areas requiring further development.
