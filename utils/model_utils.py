import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import joblib
import streamlit as st

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Audio Classifier
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

# GNN model
class MiniGNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = pyg_nn.GCNConv(input_dim, 64)
        self.conv2 = pyg_nn.GCNConv(64, output_dim)

    def forward(self, x, edge_index):
        if edge_index.numel() == 0:
            return torch.nn.functional.linear(
                x, torch.nn.Parameter(torch.randn(x.size(1), 64)).to(x.device)
            )
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# Fusion Model
from torchvision import models
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
            edge_index = torch.tensor(
                [[i, j] for i in range(batch_size) for j in range(batch_size) if i != j],
                dtype=torch.long
            ).T.to(device)
        else:
            edge_index = torch.tensor([[0, 0]], dtype=torch.long).T.to(device)
            
        gnn_features = self.gnn(cnn_features, edge_index)
        combined_features = torch.cat((cnn_features, gnn_features), dim=1)
        return self.fc(combined_features)

# Load models
def load_visual_model():
    try:
        model = FusionNet().to(device)
        state_dict = torch.load('models/deepfake_model.pth', map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Visual model error: {str(e)}")
        return None

def load_audio_models():
    try:
        svm_model = joblib.load("models/svm_audio_model.pkl")
    except Exception as e:
        st.error(f"SVM audio model error: {str(e)}")
        svm_model = None
    
    try:
        nn_model = AudioClassifier().to(device)
        state_dict = torch.load("models/deepfake_nn.pth", map_location=device)
        nn_model.load_state_dict(state_dict, strict=False)
        nn_model.eval()
        return svm_model, nn_model
    except Exception as e:
        st.error(f"NN audio model error: {str(e)}")
        return svm_model, None
