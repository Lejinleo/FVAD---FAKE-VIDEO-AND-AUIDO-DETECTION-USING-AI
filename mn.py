import os
import tempfile
import numpy as np
import librosa
import torch
import torch.nn as nn
import cv2
from mtcnn import MTCNN
from torchvision import models, transforms
from PIL import Image
import torch_geometric.nn as pyg_nn
import streamlit as st
import joblib
from moviepy import VideoFileClip
import warnings
import subprocess  # Added for ffmpeg alternative

# Suppress warnings
warnings.filterwarnings('ignore')

# GPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- MODEL DEFINITIONS ----------------------

class AudioClassifier(nn.Module):
    def __init__(self, input_size=13, hidden_size=128, num_classes=2):
        super(AudioClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MiniGNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = pyg_nn.GCNConv(input_dim, 64)
        self.conv2 = pyg_nn.GCNConv(64, output_dim)

    def forward(self, x, edge_index):
        if edge_index.numel() == 0:
            return torch.nn.functional.linear(
                x, 
                torch.nn.Parameter(torch.randn(x.size(1), 64)).to(x.device)
            )
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(512, 128)
        self.gnn = MiniGNN(input_dim=128, output_dim=64)
        self.fc = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x_img):
        x_img = x_img.to(device)
        cnn_features = self.cnn(x_img)
        batch_size = cnn_features.shape[0]
        
        if batch_size > 1:
            edge_index = torch.tensor([[i, j] for i in range(batch_size) for j in range(batch_size) if i != j], 
                                    dtype=torch.long).T.to(device)
        else:
            edge_index = torch.tensor([[0, 0]], dtype=torch.long).T.to(device)
            
        gnn_features = self.gnn(cnn_features, edge_index)
        combined_features = torch.cat((cnn_features, gnn_features), dim=1)
        return self.fc(combined_features)

# ---------------------- UTILITY FUNCTIONS ----------------------

def safe_extract_faces(video_path, output_dir, label, max_frames=40):
    """Robust face extraction with proper resource cleanup"""
    os.makedirs(output_dir, exist_ok=True)
    detector = MTCNN()
    cap = cv2.VideoCapture(video_path)
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sampling_rate = max(1, total_frames // max_frames)
        saved_count = 0
        
        for frame_idx in range(0, total_frames, sampling_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(frame_rgb)
            
            if faces:
                x, y, w, h = faces[0]['box']
                face = frame_rgb[y:y+h, x:x+w]
                face = cv2.resize(face, (128, 128))
                face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                output_path = os.path.join(output_dir, f"{label}_{frame_idx}.jpg")
                cv2.imwrite(output_path, face)
                saved_count += 1
                
                if saved_count >= max_frames:
                    break
                    
        return saved_count
        
    except Exception as e:
        st.error(f"Face extraction error: {str(e)}")
        return 0
    finally:
        cap.release()

def safe_extract_audio(video_path):
    """Robust audio extraction with proper resource cleanup"""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_tmp:
            audio_path = audio_tmp.name
        
        # Option 1: Using MoviePy (simplified version)
        try:
            with VideoFileClip(video_path) as video_clip:
                video_clip.audio.write_audiofile(audio_path, logger=None)
            return audio_path
        except:
            # Fallback to Option 2: Using ffmpeg directly if MoviePy fails
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',              # no video
                '-ac', '1',        # mono audio
                '-ar', '16000',    # sample rate
                '-y',              # overwrite output
                audio_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return audio_path
            
    except Exception as e:
        st.error(f"Audio extraction failed: {str(e)}")
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.unlink(audio_path)
        return None

def safe_extract_mfcc(audio_path):
    """Robust MFCC feature extraction"""
    try:
        audio, _ = librosa.load(audio_path, sr=16000)
        mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        st.error(f"MFCC extraction failed: {str(e)}")
        return None

def predict(image_path, model):
    """Predict whether an image is real or fake"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        batch = torch.cat([image_tensor, image_tensor], dim=0).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(batch)
            prob = torch.softmax(output, dim=1)[0]
            return "Fake" if prob[1] > 0.5 else "Real", prob[1].item()
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return "Error", 0.0

def compare_results(audio_path, svm_model, nn_model, threshold=0.7):
    """Compare results from SVM and NN models"""
    try:
        features = safe_extract_mfcc(audio_path)
        if features is None:
            return "MFCC extraction failed"
            
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        
        # SVM Prediction

        
        svm_pred = svm_model.predict([features])[0]
        svm_conf = max(svm_model.predict_proba([features])[0])
        
        # Neural Network Prediction
        nn_output = nn_model(features_tensor)
        nn_pred = torch.argmax(nn_output).item()
        nn_conf = torch.softmax(nn_output, dim=0)[nn_pred].item()

        st.markdown(f"🎙️ SVM Prediction: {'Fake' if svm_pred == 1 else 'Real'} (Confidence: {svm_conf:.2f})")
        st.markdown(f"🎙️ NN Prediction: {'Fake' if nn_pred == 1 else 'Real'} (Confidence: {nn_conf:.2f})")

        # Improved decision logic
        if svm_pred == nn_pred:
            verdict = "Fake" if nn_pred == 1 else "Real"
            return f"Final Prediction: {verdict} Audio (Models agree)"
        else:
            # When models disagree, trust the more confident one
            if nn_conf >= svm_conf:
                verdict = "Fake" if nn_pred == 1 else "Real"
                return f"Final Prediction: {verdict} Audio (NN model more confident)"
            else:
                verdict = "Fake" if svm_pred == 1 else "Real"
                return f"Final Prediction: {verdict} Audio (SVM model more confident)"
    except Exception as e:
        st.error(f"Comparison error: {str(e)}")
        return "Audio analysis failed"
def combine_results(video_result, audio_result):
    """Combine video and audio analysis results with correct percentage display"""
    # Handle error cases first
    if "error" in video_result or "error" in audio_result:
        errors = []
        if "error" in video_result:
            errors.append(f"Video: {video_result['error']}")
        if "error" in audio_result:
            errors.append(f"Audio: {audio_result['error']}")
        
        available = []
        if "verdict" in video_result:
            available.append(f"Video: {video_result['verdict']} ({100-video_result['fake_percentage']:.1f}%)")
        if "verdict" in audio_result:
            available.append(f"Audio: {audio_result['verdict']} ({audio_result['confidence']:.0f}%)")
        
        return ("Partial results:\n" + "\n".join(available)) if available else "Analysis failed:\n" + "\n".join(errors)
    
    # Calculate real percentages correctly
    video_real_pct = 100 - video_result['fake_percentage']
    audio_real_pct = 100 - audio_result['confidence'] if audio_result['verdict'] == "Real" else audio_result['confidence']
    
    # STRICT LOGIC: Either component fake = overall fake
    if video_result['verdict'] == "Fake" or audio_result['verdict'] == "Fake":
        verdict = "HIGH CONFIDENCE: DEEPFAKE DETECTED"
        icon = "⚠️"
        color = "red"
        
        # Calculate combined confidence (weighted average favoring video)
        video_score = video_result['fake_percentage'] if video_result['verdict'] == "Fake" else video_real_pct
        audio_score = audio_result['confidence'] if audio_result['verdict'] == "Fake" else audio_real_pct
        combined = (0.6 * video_score) + (0.4 * audio_score)
    else:
        verdict = "Likely AUTHENTIC"
        icon = "✅"
        color = "green"
        combined = (video_real_pct * 0.6) + (audio_real_pct * 0.4)
    
    # Build detailed result message with CORRECT percentages
    details = (
        f"{icon} {verdict}\n"
        f"- Video: {video_result['verdict']} ({video_real_pct:.1f}%)\n"
        f"- Audio: {audio_result['verdict']} ({audio_result['confidence']:.0f}%)\n"
        f"- Combined confidence: {combined:.1f}%"
    )
    
    # Add specific warnings if only one modality is fake
    if video_result['verdict'] == "Fake" and audio_result['verdict'] != "Fake":
        details += "\n⚠️ Warning: Video shows signs of manipulation while audio appears authentic"
    elif audio_result['verdict'] == "Fake" and video_result['verdict'] != "Fake":
        details += "\n⚠️ Warning: Audio shows signs of manipulation while video appears authentic"
    
    return details, color
# ---------------------- MODEL LOADING ----------------------

def load_visual_model():
    """Load the visual detection model"""
    try:
        model = FusionNet().to(device)
        state_dict = torch.load('deepfake_model.pth', map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Visual model error: {str(e)}")
        return None

def load_audio_models():
    """Load both audio detection models"""
    try:
        svm_model = joblib.load("svm_audio_moodel.pkl")
    except Exception as e:
        st.error(f"SVM audio model error: {str(e)}")
        svm_model = None
    
    try:
        nn_model = AudioClassifier().to(device)
        state_dict = torch.load("deepfake_nn.pth", map_location=device)
        nn_model.load_state_dict(state_dict, strict=False)
        nn_model.eval()
        return svm_model, nn_model
    except Exception as e:
        st.error(f"NN audio model error: {str(e)}")
        return None, None

# ---------------------- MAIN PROCESSING ----------------------

def analyze_video(video_path):
    """Analyze video frames for deepfake detection"""
    faces_dir = tempfile.mkdtemp(prefix="faces_")
    try:
        faces_count = safe_extract_faces(video_path, faces_dir, "test", 40)
        if faces_count == 0:
            return {"error": "No faces detected in video"}
        
        model = load_visual_model()
        if not model:
            return {"error": "Failed to load visual model"}
        
        predictions = []
        for img in os.listdir(faces_dir):
            try:
                result, confidence = predict(os.path.join(faces_dir, img), model)
                predictions.append((result, confidence))
            except Exception as e:
                st.warning(f"Skipping corrupted face image: {str(e)}")
            finally:
                os.remove(os.path.join(faces_dir, img))
        
        if not predictions:
            return {"error": "No valid faces processed"}
            
        fake_count = sum(1 for p in predictions if p[0] == "Fake")
        fake_percentage = fake_count / len(predictions) * 100
        
        return {
            "total_frames": len(predictions),
            "fake_percentage": fake_percentage,
            "verdict": "Fake" if fake_percentage > 50 else "Real",
            "confidence": max(fake_percentage, 100 - fake_percentage)
        }
    except Exception as e:
        return {"error": f"Video analysis failed: {str(e)}"}
    finally:
        # Clean up faces directory
        if os.path.exists(faces_dir):
            for f in os.listdir(faces_dir):
                os.remove(os.path.join(faces_dir, f))
            os.rmdir(faces_dir)

def analyze_audio(audio_input):
    """Analyze audio for deepfake detection"""
    audio_path = None
    try:
        if isinstance(audio_input, str):
            audio_path = audio_input
            should_delete = False
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_input.getbuffer())
                audio_path = f.name
                should_delete = True
        
        if not os.path.exists(audio_path):
            return {"error": "Audio file not found"}
        
        svm_model, nn_model = load_audio_models()
        if not svm_model or not nn_model:
            return {"error": "Failed to load audio models"}
        
        result = compare_results(audio_path, svm_model, nn_model)
        confidence = 80 if "high" in result.lower() else 50 if "need" in result.lower() else 70
        
        return {
            "verdict": "Fake" if "fake" in result.lower() else "Real",
            "confidence": confidence,
            "details": result
        }
    except Exception as e:
        return {"error": f"Audio analysis failed: {str(e)}"}
    finally:
        if audio_path and 'should_delete' in locals() and should_delete and os.path.exists(audio_path):
            os.unlink(audio_path)

# ---------------------- STREAMLIT UI ----------------------

def main():
    st.title("Deepfake Detection System")
    st.write("Choose analysis type and upload media file")
    
    # Analysis type selection
    analysis_type = st.radio(
        "Select analysis type:",
        ("Video Only", "Audio Only", "Combined Video+Audio"),
        horizontal=True
    )

    uploaded_file = st.file_uploader("Choose file", type=["mp4", "avi", "mov", "mp3", "wav","opus"])
    
    if uploaded_file:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        temp_path = None
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_path = tmp.name
            
            # VIDEO ONLY ANALYSIS
            if analysis_type == "Video Only":
                if file_ext not in ['mp4', 'avi', 'mov']:
                    st.error("Please upload a video file for video analysis")
                    return
                
                st.subheader("Video Analysis")
                with st.spinner("Analyzing video frames..."):
                    video_result = analyze_video(temp_path)
                
                if "error" in video_result:
                    st.error(video_result["error"])
                else:
                    st.write(f"Analyzed {video_result['total_frames']} frames")
                    st.write(f"Verdict: {video_result['verdict']} ({video_result['confidence']:.1f}%)")
                    if video_result['verdict'] == "Fake":
                        st.warning(f"⚠️ Detected as fake with {video_result['fake_percentage']:.1f}% confidence")
                    else:
                        st.success(f"✅ Likely authentic ({100-video_result['fake_percentage']:.1f}% confidence)")

            # AUDIO ONLY ANALYSIS
            elif analysis_type == "Audio Only":
                if file_ext not in ['mp3', 'wav','opus']:
                    st.error("Please upload an audio file for audio analysis")
                    return
                
                st.subheader("Audio Analysis")
                with st.spinner("Analyzing audio features..."):
                    audio_result = analyze_audio(uploaded_file)
                
                if "error" in audio_result:
                    st.error(audio_result["error"])
                else:
                    st.write(audio_result["details"])
                    if audio_result['verdict'] == "Fake":
                        st.warning(f"⚠️ Detected as fake with {audio_result['confidence']:.0f}% confidence")
                    else:
                        st.success(f"✅ Likely authentic ({100-audio_result['confidence']:.0f}% confidence)")

            # COMBINED ANALYSIS
            elif analysis_type == "Combined Video+Audio":
                if file_ext not in ['mp4', 'avi', 'mov']:
                    st.error("Please upload a video file for combined analysis")
                    return
                
                cols = st.columns(2)
                
                with cols[0]:
                    st.subheader("Video Analysis")
                    with st.spinner("Extracting and analyzing video frames..."):
                        video_result = analyze_video(temp_path)
                    
                    if "error" in video_result:
                        st.error(video_result["error"])
                    else:
                        st.write(f"Analyzed {video_result['total_frames']} frames")
                        st.write(f"Verdict: {video_result['verdict']} ({video_result['confidence']:.1f}%)")
                
                with cols[1]:
                    st.subheader("Audio Analysis")
                    with st.spinner("Extracting audio..."):
                        audio_path = safe_extract_audio(temp_path)
                    
                    if audio_path:
                        # Check if audio was actually extracted (file size > 1KB)
                        if os.path.getsize(audio_path) > 1024:
                            with st.spinner("Analyzing audio features..."):
                                audio_result = analyze_audio(audio_path)
                            
                            if "error" in audio_result:
                                st.error(audio_result["error"])
                            else:
                                st.write(audio_result["details"])
                        else:
                            audio_result = {"error": "No audio track found in video"}
                            st.warning("No audio track detected in video")
                    else:
                        audio_result = {"error": "Audio extraction failed"}
                        st.error("Audio extraction failed")
                
                st.subheader("Final Analysis")
                if "error" not in video_result:
                    if "error" not in audio_result and os.path.getsize(audio_path) > 1024:
                        combined = combine_results(video_result, audio_result)
                    else:
                        combined = f"⚠️ Video Only Analysis\n{video_result['verdict']} ({video_result['confidence']:.1f}%)\nNo audio available for analysis"
                    
                    result_text, color = combined  # Unpack the tuple returned by combine_results()
                    st.write(result_text)
                    
                    if color == "red":
                       st.error("STRONG DEEPFAKE INDICATORS DETECTED!")
                    elif color == "orange":
                      st.warning("Suspicious content - potential manipulation")
                    else:
                        st.success("Content appears authentic")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

if __name__ == "__main__":
    main()