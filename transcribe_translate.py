import os
import logging
import subprocess
import requests
import time
import streamlink
import html
import numpy as np
import whisper
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect

# Configure logging
logging.basicConfig(level=logging.INFO)

# Twitch client credentials
CLIENT_ID = "YOUR_TWITCH_CLIENT_ID"
CLIENT_SECRET = "YOUR_TWITCH_CLIENT_SECRET"

# Twitch channel name
CHANNEL_NAME = "YOUR_TWITCH_CHANNEL_NAME"

# Load Whisper model for transcription
whisper_model = whisper.load_model("large-v2")

# Define exceptions for specific language pairs
exceptions = {
    "en-ja": "Helsinki-NLP/opus-tatoeba-en-ja",
    "pt-en": "Helsinki-NLP/opus-mt-mul-en",
    "zh-en": "Helsinki-NLP/opus-mt-zh-en",
    "en-zh": "Helsinki-NLP/opus-mt-en-zh",
    "ja-en": "Helsinki-NLP/opus-mt-ja-en",
    "ko-en": "Helsinki-NLP/opus-mt-ko-en",
    "en-ko": "Helsinki-NLP/opus-mt-en-ko",
    "ar-en": "Helsinki-NLP/opus-mt-ar-en",
    "en-ar": "Helsinki-NLP/opus-mt-en-ar",
    "ru-en": "Helsinki-NLP/opus-mt-ru-en",
    "en-ru": "Helsinki-NLP/opus-mt-en-ru",
    "fr-de": "Helsinki-NLP/opus-mt-fr-de",
    "de-fr": "Helsinki-NLP/opus-mt-de-fr",
    "es-it": "Helsinki-NLP/opus-mt-es-it",
    "it-es": "Helsinki-NLP/opus-mt-it-es",
    "nl-sv": "Helsinki-NLP/opus-mt-nl-sv",
    "sv-nl": "Helsinki-NLP/opus-mt-sv-nl",
    "pl-ru": "Helsinki-NLP/opus-mt-pl-ru",
    "ru-pl": "Helsinki-NLP/opus-mt-ru-pl",
    "zh-ja": "Helsinki-NLP/opus-mt-zh-ja",
    "ja-zh": "Helsinki-NLP/opus-mt-ja-zh",
    "ko-ar": "Helsinki-NLP/opus-mt-ko-ar",
    "ar-ko": "Helsinki-NLP/opus-mt-ar-ko",
    "tr-da": "Helsinki-NLP/opus-mt-tr-da",
    "da-tr": "Helsinki-NLP/opus-mt-da-tr",
    "fi-no": "Helsinki-NLP/opus-mt-fi-no",
    "no-fi": "Helsinki-NLP/opus-mt-no-fi",
    "cs-el": "Helsinki-NLP/opus-mt-cs-el",
    "el-cs": "Helsinki-NLP/opus-mt-el-cs"
}

# Load translation models dynamically based on the source and target languages
def load_translation_model(source_lang, target_lang):
    model_name = exceptions.get(f"{source_lang}-{target_lang}", f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}")
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        logging.info(f"Loaded model and tokenizer for {source_lang} to {target_lang}")
        return tokenizer, model
    except Exception as e:
        logging.error(f"Error loading model {model_name}: {e}")
        raise

# Authenticate with Twitch API to obtain access token
def get_access_token(client_id, client_secret):
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
        logging.error("Failed to authenticate with Twitch API")
        raise Exception("Failed to authenticate with Twitch API")

# Fetch stream metadata for the specified channel
def fetch_stream_metadata(access_token, channel_name):
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
            logging.info("Channel is not currently live")
            return None
    else:
        logging.error("Failed to fetch stream metadata")
        raise Exception("Failed to fetch stream metadata")

# Get the stream URL using streamlink
def get_stream_url(channel_name):
    streams = streamlink.streams(f"https://www.twitch.tv/{channel_name}")
    if "audio_only" in streams:
        return streams["audio_only"].url
    elif "best" in streams:
        return streams["best"].url
    else:
        raise Exception("No suitable stream found.")

# Transcribe streaming audio using Whisper
def transcribe_audio_stream(audio_stream):
    buffer = bytearray()
    CHUNK_SIZE = 16000 * 10  # 10 seconds of audio
    while True:
        data = audio_stream.read(4000)
        if len(data) == 0:
            logging.info("End of audio stream")
            break
        buffer.extend(data)
        if len(buffer) >= CHUNK_SIZE * 2:
            audio_data = np.frombuffer(buffer[:CHUNK_SIZE*2], np.int16).astype(np.float32) / 32768.0  # Convert to float32
            buffer = buffer[CHUNK_SIZE*2:]  # Remove processed data from buffer
            result = whisper_model.transcribe(audio_data)
            text = result.get('text', '')
            if text:
                yield text

# Translate text using Hugging Face Transformers
def translate_text(text, tokenizer, model):
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return translated_text[0]

# Capture audio from Twitch stream using FFmpeg
def capture_audio_from_stream(stream_url):
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

# Main function to continuously fetch stream metadata, transcribe audio, and translate text
def main():
    import sys
    if len(sys.argv) != 3:
        print("Usage: python transcribe_translate.py <source_lang> <target_lang>")
        return

    source_lang = sys.argv[1]
    target_lang = sys.argv[2]

    # Load translation model
    tokenizer, model = load_translation_model(source_lang, target_lang)

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
                        try:
                            detected_lang = detect(transcribed_text)
                            if detected_lang == source_lang:
                                translated_text = translate_text(transcribed_text, tokenizer, model)
                                print(f"Translated Text: {translated_text}")
                            else:
                                logging.info(f"Ignoring text in unsupported language: {transcribed_text}")
                        except Exception as e:
                            logging.error(f"Language detection error: {e}")
            else:
                logging.info("Stream is not live. Retrying in 10 seconds...")

            # Sleep before fetching stream metadata again
            time.sleep(10)
        except Exception as e:
            logging.error(f"Error: {e}")
            time.sleep(10)  # Sleep before retrying in case of an error

if __name__ == "__main__":
    main()
