import sys
from cx_Freeze import setup, Executable
import os

# Define paths for CUDA and FFmpeg
cuda_path = "YOUR_CUDA_PATH/bin"
cuda_libs = [
    "cudart64_12.dll",
    "cublas64_12.dll",
    "cufft64_11.dll",
    "curand64_10.dll",
    "cusolver64_11.dll",
]

# Verify CUDA files exist
for lib in cuda_libs:
    if not os.path.exists(os.path.join(cuda_path, lib)):
        raise FileNotFoundError(f"Cannot find file: {os.path.join(cuda_path, lib)}")

ffmpeg_path = "YOUR_FFMPEG_PATH/bin"
ffmpeg_executable = "ffmpeg.exe"

# Verify FFmpeg file exists
if not os.path.exists(os.path.join(ffmpeg_path, ffmpeg_executable)):
    raise FileNotFoundError(f"Cannot find file: {os.path.join(ffmpeg_path, ffmpeg_executable)}")

# Build options for cx_Freeze
build_exe_options = {
    "packages": ["os", "torch", "transformers", "streamlink", "whisper", "langdetect"],
    "include_files": [(os.path.join(cuda_path, lib), lib) for lib in cuda_libs] + [(os.path.join(ffmpeg_path, ffmpeg_executable), ffmpeg_executable)],
    "excludes": ["tkinter"],
    "include_msvcr": True,
}

# Base setup
base = None
if sys.platform == "win32":
    base = "Console"

# Executable configuration
executables = [
    Executable("transcribe_translate.py", base=base, target_name="transcribe_translate.exe")
]

# Setup configuration
setup(
    name="transcribe_translate",
    version="0.1",
    description="Twitch Stream Transcription and Translation Script",
    options={"build_exe": build_exe_options},
    executables=executables
)
