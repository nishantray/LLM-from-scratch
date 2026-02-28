# ============================================================
# classifier.py
# Utilities for Spam Classification using Fine-Tuned GPT
# ============================================================

import torch
import torch.nn.functional
import matplotlib.pyplot as plt
import pandas as pd
import os
import zipfile
import requests
from pathlib import Path


# ============================================================
# Accuracy Calculation
# ============================================================

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """
    Computes classification accuracy over a data loader.

    Only the final token's logits are used for prediction,
    since the classifier head outputs 2 classes per sequence.

    Args:
        data_loader: PyTorch DataLoader
        model: Trained classification model
        device: CPU/GPU device
        num_batches: Optional limit for evaluation

    Returns:
        Accuracy (float)
    """
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                # Use last token output for classification
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break

    return correct_predictions / num_examples


# ============================================================
# Loss Functions
# ============================================================

def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Computes cross-entropy loss for a single batch.

    Only the final token logits are used for classification.
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)

    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Computes average loss over multiple batches.
    """
    total_loss = 0.

    if len(data_loader) == 0:
        return float("nan")

    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluates training and validation loss without updating gradients.
    """
    model.eval()

    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )

    model.train()
    return train_loss, val_loss


# ============================================================
# Training Loop
# ============================================================

def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    """
    Fine-tunes GPT for binary spam classification.

    Tracks:
        - Training loss
        - Validation loss
        - Training accuracy
        - Validation accuracy
        - Number of examples seen
    """
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            examples_seen += input_batch.shape[0]
            global_step += 1

            # Periodic evaluation
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)

        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")

        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


# ============================================================
# Inference Function
# ============================================================

def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    """
    Classifies a single text as 'spam' or 'not spam'.

    - Tokenizes input
    - Pads to max_length
    - Uses final token logits for prediction
    """
    model.eval()

    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    input_ids = input_ids[:min(max_length, supported_context_length)]

    assert max_length is not None, (
        "max_length must be specified. If you want to use the full model context, "
        "pass max_length=model.pos_emb.weight.shape[0]."
    )

    assert max_length <= supported_context_length, (
        f"max_length ({max_length}) exceeds model's supported context length ({supported_context_length})."
    )

    # Pad sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))

    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]

    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"


# ============================================================
# Plotting Utilities
# ============================================================

def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    """
    Plots training vs validation curves (loss or accuracy).
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Secondary axis: examples seen
    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()


# ============================================================
# Dataset Utilities
# ============================================================

def create_balanced_dataset(df):
    """
    Balances dataset by downsampling 'ham' class
    to match number of 'spam' samples.
    """
    num_spam = df[df["Label"] == "spam"].shape[0]

    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df


# ============================================================
# Dataset Download Utility
# ============================================================

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    """
    Downloads and extracts SMS Spam Collection dataset.

    Skips download if file already exists.
    """
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download.")
        return

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with open(zip_path, "wb") as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                out_file.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)

    print(f"File downloaded and saved as {data_file_path}")