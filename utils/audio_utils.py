import os
import tempfile
import subprocess
import librosa
import numpy as np
import torch
import streamlit as st
from moviepy import VideoFileClip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Audio extraction
def safe_extract_audio(video_path):
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_tmp:
            audio_path = audio_tmp.name
        
        try:
            with VideoFileClip(video_path) as video_clip:
                video_clip.audio.write_audiofile(audio_path, logger=None)
            return audio_path
        except:
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn', '-ac', '1', '-ar', '16000', '-y', audio_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return audio_path
    except Exception as e:
        st.error(f"Audio extraction failed: {str(e)}")
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.unlink(audio_path)
        return None

# MFCC feature extraction
def safe_extract_mfcc(audio_path):
    try:
        audio, _ = librosa.load(audio_path, sr=16000)
        mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        st.error(f"MFCC extraction failed: {str(e)}")
        return None
