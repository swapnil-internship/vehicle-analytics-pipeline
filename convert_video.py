
from pathlib import Path
import subprocess

import imageio_ffmpeg


input_video = Path("outputs/tracked_vehicles.mp4")
output_video = Path("outputs/tracked_vehicles_web.mp4")

if not input_video.exists():
    raise FileNotFoundError(
        f"Input video not found: {input_video.resolve()}"
    )

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

command = [
    ffmpeg,
    "-y",
    "-i", str(input_video),
    "-c:v", "libx264",
    "-preset", "fast",
    "-crf", "23",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    "-an",
    str(output_video),
]

print("Converting video to browser-compatible H.264...")
subprocess.run(command, check=True)

print("Conversion completed!")
print(f"Output: {output_video.resolve()}")
print(f"Size: {output_video.stat().st_size / (1024 * 1024):.1f} MB")
