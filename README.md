# Twitch Stream Transcription and Translation Script

This repository contains a Python script that transcribes and translates live audio from a Twitch stream. The script uses OpenAI's Whisper model for transcription and Hugging Face's MarianMT model for translation. It supports dynamic language translation for the top 20 languages.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Setting Up a Virtual Environment](#setting-up-a-virtual-environment)
- [Configuration](#configuration)
- [Usage](#usage)
- [Adding a Language](#adding-a-language)
- [How It Works](#how-it-works)
- [CUDA vs CPU](#cuda-vs-cpu)
- [License](#license)

## Features
- Transcribes live audio from a specified Twitch channel.
- Translates transcribed text from a source language to a target language.
- Supports dynamic language selection for both source and target languages.
- Uses Whisper model for transcription and MarianMT model for translation.
- Logs system messages and errors for easy debugging.

## Requirements
- Python 3.7 or higher
- ffmpeg
- pip

## Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/gorgarp/TwitchTranslate.git
    cd TwitchTranslate
    ```

2. **Install FFmpeg**
    - **Windows**: Download and install from [FFmpeg official website](https://ffmpeg.org/download.html).
    - **macOS**: Use Homebrew
      ```bash
      brew install ffmpeg
      ```
    - **Linux**: Use your package manager
      ```bash
      sudo apt-get install ffmpeg
      ```

## Setting Up a Virtual Environment
1. **Create a Virtual Environment**
    ```bash
    python -m venv myenv
    ```

2. **Activate the Virtual Environment**
    - **Windows**
      ```bash
      myenv\Scripts\activate
      ```
    - **macOS/Linux**
      ```bash
      source myenv/bin/activate
      ```

3. **Install the Required Python Packages**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration
1. **Set Up Twitch API Token**
    - Go to the [Twitch Developer Portal](https://dev.twitch.tv/console/apps).
    - Register your application to get the `CLIENT_ID` and `CLIENT_SECRET`.
    - Replace `YOUR_TWITCH_CLIENT_ID` and `YOUR_TWITCH_CLIENT_SECRET` in the script with your actual Twitch API credentials.

2. **Configure the Twitch Channel**
    - Replace `YOUR_TWITCH_CHANNEL_NAME` with the name of the Twitch channel you want to transcribe and translate.

## Usage
1. **Run the Script**
    ```bash
    python transcribe_translate.py <source_lang> <target_lang>
    ```
    - Example:
      ```bash
      python transcribe_translate.py es en  # Translates Spanish to English
      python transcribe_translate.py pl en  # Translates Polish to English
      ```

## Adding a Language
1. **Check Supported Languages**
    - The script currently supports the following languages:
      ```
      "en", "fr", "de", "es", "it", "nl", "sv", "pl", "pt", "ru", "zh", "ja", "ko", 
      "ar", "tr", "da", "fi", "no", "cs", "el"
      ```

2. **Add Language to Supported List**
    - To add a new language, ensure it is supported by MarianMT and Whisper models.
    - Update the `supported_languages` list in the script with the new language code.

## How It Works
1. **Transcription**
    - The script captures live audio from a specified Twitch channel using FFmpeg.
    - It uses the Whisper model to transcribe the audio into text.

2. **Translation**
    - The detected language of the transcribed text is checked against the specified source language.
    - If it matches, the text is translated into the target language using MarianMT.
    - The translated text is printed to the console.

3. **System Messages and Error Handling**
    - The script logs system messages such as model loading and errors for easy debugging and monitoring.

## CUDA vs CPU
The script can run on either CUDA (GPU) or CPU. Using CUDA significantly improves the performance and speed of both transcription and translation.

1. **Checking CUDA Availability**
    - The script automatically checks if CUDA is available and uses it if possible:
      ```python
      device = "cuda" if torch.cuda.is_available() else "cpu"
      logging.info(f"Using device: {device}")
      ```

2. **Installing CUDA (if needed)**
    - **Windows**:
      1. Download and install the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).
      2. Add CUDA to your PATH:
         ```powershell
         [Environment]::SetEnvironmentVariable("CUDA_PATH", "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1", "User")
         $env:Path += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
         ```
      3. Reboot your system to ensure the changes take effect.
      4. Verify the installation by running:
         ```bash
         nvcc --version
         ```
      > **Note:** The above commands reference CUDA version 12.1. If you install a different version, adjust the paths accordingly.

    - **macOS**: CUDA is not supported on macOS.
    
    - **Linux**:
      1. Download and install the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).
      2. Add CUDA to your PATH:
         ```bash
         export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
         export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64\
                              ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
         ```
      3. Verify the installation by running:
         ```bash
         nvcc --version
         ```
      > **Note:** The above commands reference CUDA version 12.1. If you install a different version, adjust the paths accordingly.

3. **Installing PyTorch with CUDA Support**
    - Install PyTorch with CUDA support:
      ```bash
      pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
      ```
    > **Note:** The above command installs PyTorch with CUDA 11.7 support. Ensure the versions are compatible with your CUDA installation.

## Detailed Script Breakdown
1. **Authentication**: The script authenticates with the Twitch API using the provided client ID and secret. It obtains an access token required for making API requests.
2. **Fetching Stream Metadata**: The script fetches metadata for the specified Twitch channel to check if the channel is live.
3. **Getting Stream URL**: The script uses Streamlink to get the best quality stream URL from the Twitch channel.
4. **Capturing Audio**: The script uses FFmpeg to capture audio from the Twitch stream.
5. **Transcribing Audio**: The Whisper model is used to transcribe the captured audio into text.
6. **Translating Text**: The script detects the language of the transcribed text and translates it into the target language using MarianMT if the language matches the specified source language.
7. **Output**: The translated text is printed to the console.

## Troubleshooting
- **Common Issues**:
  - Ensure FFmpeg is installed and added to your system's PATH.
  - Ensure you have the correct client ID, client secret, and Twitch channel name in the script.
  - Verify CUDA installation if using GPU for better performance.
- **Logs and Debugging**:
  - Check the logs for any error messages or system messages to identify issues.
  - The script logs system messages and errors for easy debugging and monitoring.

## Contributing
1. **Fork the Repository**
2. **Create a Feature Branch**
    ```bash
    git checkout -b feature-branch
    ```
3. **Commit Your Changes**
    ```bash
    git commit -m "Add some feature"
    ```
4. **Push to the Branch**
    ```bash
    git push origin feature-branch
    ```
5. **Open a Pull Request**

## Contact
For any questions or issues, please open an issue in the [GitHub repository](https://github.com/gorgarp/TwitchTranslate).

## License
This project is licensed under the MIT License.
