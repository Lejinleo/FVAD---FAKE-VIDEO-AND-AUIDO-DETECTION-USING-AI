import os
import cv2
import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN as FaceNetMTCNN
import streamlit as st
import tempfile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Face extraction
def safe_extract_faces(video_path, output_dir, label, max_frames=40):
    os.makedirs(output_dir, exist_ok=True)
    detector = FaceNetMTCNN(keep_all=False, device=device)
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
            boxes, _ = detector.detect(frame_rgb)
            
            if boxes is not None and len(boxes) > 0:
                x1, y1, x2, y2 = map(int, boxes[0])
                face = frame_rgb[y1:y2, x1:x2]
                face = cv2.resize(face, (128, 128))
                face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                output_path = os.path.join(output_dir, f"{label}_{frame_idx}.jpg")
                cv2.imwrite(output_path, face_bgr)
                saved_count += 1
                
                if saved_count >= max_frames:
                    break
        return saved_count
    except Exception as e:
        st.error(f"Face extraction error: {str(e)}")
        return 0
    finally:
        cap.release()

# Image prediction
def predict(image_path, model):
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
