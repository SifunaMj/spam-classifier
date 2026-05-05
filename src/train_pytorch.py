import torch
import torch.nn as nn
import time

from train_sklearn import load_data, apply_tfidf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utils import build_pytorch_model  # ✅ NEW


if __name__ == "__main__":
    print("\n🚀 Starting PyTorch Training Pipeline")

    # =========================
    # 🔹 Load + Prepare Data
    # =========================
    df = load_data("data/processed/cleaned.csv")
    X, y, _ = apply_tfidf(df)

    input_dim = X.shape[1]  # ✅ dynamic (safer than hardcoding 3000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # =========================
    # 🔹 Convert to tensors
    # =========================
    X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
    X_test_tensor  = torch.tensor(X_test.toarray(), dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_tensor  = torch.tensor(y_test.values, dtype=torch.float32)

    print("✅ Data converted to tensors")

    # =========================
    # 🔹 Model
    # =========================
    model = build_pytorch_model(input_dim)  # ✅ USE UTILS

    # =========================
    # 🔹 Loss + Optimizer
    # =========================
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # =========================
    # 🔹 Training
    # =========================
    epochs = 30
    batch_size = 64

    print("\n🧠 Training Neural Network...")

    start = time.time()

    for epoch in range(epochs):
        model.train()

        permutation = torch.randperm(X_train_tensor.size(0))
        epoch_loss = 0

        for i in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[i:i+batch_size]

            batch_X = X_train_tensor[indices]
            batch_y = y_train_tensor[indices]

            outputs = model(batch_X).squeeze()
            loss = loss_fn(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    training_time = time.time() - start

    # =========================
    # 🔹 Evaluation
    # =========================
    model.eval()

    with torch.no_grad():
        preds = model(X_test_tensor).squeeze()
        preds = (preds >= 0.5).float()

        accuracy = (preds == y_test_tensor).float().mean()

    print(f"\n📊 PyTorch Model Accuracy: {accuracy:.4f}")

    print("\n📊 Classification Report:")
    print(classification_report(y_test_tensor.numpy(), preds.numpy()))

    # =========================
    # 🔹 Save model
    # =========================
    torch.save(model.state_dict(), "models/pytorch_model.pt")
    print("\n✅ PyTorch model saved")

    # =========================
    # 🔹 Training Time
    # =========================
    print(f"\n⏱️ Training Time: {training_time:.4f} seconds")

    print("\n✅ PyTorch pipeline complete")