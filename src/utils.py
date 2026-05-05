import torch
import torch.nn as nn


# =========================
# 🔹 Build PyTorch Model
# =========================
def build_pytorch_model(input_dim=3000):
    """
    Returns a PyTorch neural network model for spam classification.
    
    Args:
        input_dim (int): Number of input features (TF-IDF size)

    Returns:
        torch.nn.Module
    """
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )
    return model


# =========================
# 🔹 Predict with PyTorch Model
# =========================
def predict_pytorch(model, vectorized_input):
    """
    Runs inference using a trained PyTorch model.

    Args:
        model: Trained PyTorch model
        vectorized_input: TF-IDF vector (sparse matrix)

    Returns:
        (prediction, probability)
    """
    model.eval()

    tensor_input = torch.tensor(vectorized_input.toarray(), dtype=torch.float32)

    with torch.no_grad():
        output = model(tensor_input).item()

    prediction = 1 if output >= 0.5 else 0

    return prediction, output