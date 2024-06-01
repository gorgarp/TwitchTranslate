import os
import logging
import subprocess
import requests
import time
import streamlink
import numpy as np
import torch
import whisper
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)

# Twitch client credentials (use placeholders for security)
CLIENT_ID = "YOUR_TWITCH_CLIENT_ID"
CLIENT_SECRET = "YOUR_TWITCH_CLIENT_SECRET"

# Twitch channel name (use a placeholder)
CHANNEL_NAME = "YOUR_TWITCH_CHANNEL_NAME"

# Check if CUDA is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Load the Whisper model for transcription
whisper_model = whisper.load_model("large").to(device)

# List of supported languages for translation
supported_languages = [
    "en", "fr", "de", "es", "it", "nl", "sv", "pl", "pt", "ru", "zh", "ja", "ko", 
    "ar", "tr", "da", "fi", "no", "cs", "el"
]

# Dictionaries to store loaded tokenizers and models
tokenizers = {}
models = {}

def load_model(source_lang, target_lang):
    """
    Load the translation model and tokenizer for the specified language pair.

    Args:
        source_lang (str): Source language code.
        target_lang (str): Target language code.

    Returns:
        tuple: Tokenizer and model for the specified language pair.
    """
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    if (source_lang, target_lang) not in tokenizers:
        logging.info(f"Loading model and tokenizer for {source_lang} to {target_lang}")
        tokenizers[(source_lang, target_lang)] = MarianTokenizer.from_pretrained(model_name)
        models[(source_lang, target_lang)] = MarianMTModel.from_pretrained(model_name).to(device)
    return tokenizers[(source_lang, target_lang)], models[(source_lang, target_lang)]

def get_access_token(client_id, client_secret):
    """
    Authenticate with the Twitch API to obtain an access token.

    Args:
        client_id (str): Twitch client ID.
        client_secret (str): Twitch client secret.

    Returns:
        str: Access token for Twitch API.
    """
    auth_url = "https://id.twitch.tv/oauth2/token"
    params = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials"
    }
    response = requests.post(auth_url, params=params)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception("Failed to authenticate with Twitch API")

def fetch_stream_metadata(access_token, channel_name):
    """
    Fetch stream metadata for the specified Twitch channel.

    Args:
        access_token (str): Access token for Twitch API.
        channel_name (str): Twitch channel name.

    Returns:
        dict: Metadata for the live stream if found, otherwise None.
    """
    stream_url = f"https://api.twitch.tv/helix/streams?user_login={channel_name}"
    headers = {
        "Client-ID": CLIENT_ID,
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(stream_url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data["data"]:
            return data["data"][0]  # Return metadata for the first live stream found
        else:
            return None
    else:
        raise Exception("Failed to fetch stream metadata")

def get_stream_url(channel_name):
    """
    Get the stream URL using Streamlink.

    Args:
        channel_name (str): Twitch channel name.

    Returns:
        str: URL of the best quality stream.
    """
    streams = streamlink.streams(f"https://www.twitch.tv/{channel_name}")
    if "best" in streams:
        return streams["best"].url
    else:
        raise Exception("No 'best' quality stream found.")

def transcribe_audio_stream(audio_stream):
    """
    Transcribe streaming audio using the Whisper model.

    Args:
        audio_stream (file-like object): Audio stream from FFmpeg.

    Yields:
        str: Transcribed text.
    """
    buffer = bytearray()
    CHUNK_SIZE = 16000 * 10  # 10 seconds of audio
    while True:
        data = audio_stream.read(4000)
        if len(data) == 0:
            break
        buffer.extend(data)
        if len(buffer) >= CHUNK_SIZE * 2:
            audio_data = np.frombuffer(buffer[:CHUNK_SIZE*2], np.int16).astype(np.float32) / 32768.0  # Convert to float32
            buffer = buffer[CHUNK_SIZE*2:]  # Remove processed data from buffer
            audio_tensor = torch.tensor(audio_data, device=device)
            result = whisper_model.transcribe(audio_tensor)
            text = result.get('text', '')
            if text:
                yield text

def translate_text(text, source_lang, target_lang):
    """
    Translate text using Hugging Face Transformers.

    Args:
        text (str): Text to be translated.
        source_lang (str): Source language code.
        target_lang (str): Target language code.

    Returns:
        str: Translated text.
    """
    if source_lang not in supported_languages or target_lang not in supported_languages:
        return None  # Unsupported language pair
    
    tokenizer, model = load_model(source_lang, target_lang)
    
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True).to(device))
    translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return translated_text[0]

def capture_audio_from_stream(stream_url):
    """
    Capture audio from Twitch stream using FFmpeg.

    Args:
        stream_url (str): URL of the Twitch stream.

    Returns:
        file-like object: FFmpeg process output stream.
    """
    command = [
        "ffmpeg",
        "-i", stream_url,
        "-f", "s16le",
        "-ac", "1",
        "-ar", "16000",
        "-loglevel", "quiet",
        "-"
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process.stdout

def main(source_lang, target_lang):
    """
    Main function to continuously fetch stream metadata, transcribe audio, and translate text.

    Args:
        source_lang (str): Source language code.
        target_lang (str): Target language code.
    """
    while True:
        try:
            access_token = get_access_token(CLIENT_ID, CLIENT_SECRET)
            stream_metadata = fetch_stream_metadata(access_token, CHANNEL_NAME)
            if stream_metadata:
                logging.info("Stream is live. Starting transcription...")
                audio_url = get_stream_url(CHANNEL_NAME)
                audio_stream = capture_audio_from_stream(audio_url)

                for transcribed_text in transcribe_audio_stream(audio_stream):
                    if transcribed_text:
                        detected_lang = detect(transcribed_text)
                        if detected_lang == source_lang:
                            translated_text = translate_text(transcribed_text, source_lang, target_lang)
                            if translated_text:
                                print(f"Translated Text: {translated_text}")
            else:
                time.sleep(10)  # Sleep before retrying if the stream is not live

            time.sleep(10)  # Sleep before fetching stream metadata again
        except Exception as e:
            logging.error(f"Error: {e}")
            time.sleep(10)  # Sleep before retrying in case of an error

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        source_lang = sys.argv[1]
        target_lang = sys.argv[2]
        if source_lang in supported_languages and target_lang in supported_languages:
            main(source_lang, target_lang)
        else:
            logging.error("Unsupported language. Please use one of the supported languages: en, fr, de, es, it, nl, sv, pl, pt, ru, zh, ja, ko, ar, tr, da, fi, no, cs, el")
    else:
        logging.error("Usage: python transcribe_translate.py <source_lang> <target_lang>")
