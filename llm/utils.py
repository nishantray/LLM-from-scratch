# ============================================================
# utils.py
# General Utilities for Training, Evaluation, and Generation
# ============================================================

import torch
import torch.nn.functional
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ============================================================
# Tokenization Utilities
# ============================================================

def text_to_token_ids(text, tokenizer):
    """
    Converts text into token ID tensor of shape (1, sequence_length).

    Allows special token <|endoftext|>.
    """
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """
    Converts token ID tensor back into decoded text.
    """
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


# ============================================================
# Loss Functions (Language Modeling)
# ============================================================

def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Computes next-token prediction loss for a single batch.

    Flattens batch and sequence dimensions for cross-entropy.
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)

    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )

    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Computes average loss across multiple batches.
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
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


# ============================================================
# Evaluation Helper
# ============================================================

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluates training and validation loss without gradient updates.
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
# Simple Greedy Text Generation
# ============================================================

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Greedy decoding (argmax at each step).
    """

    for _ in range(max_new_tokens):

        # Crop context window
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)

        idx_next = torch.argmax(probs, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


# ============================================================
# Sample Generation Helper (Prints Text)
# ============================================================

def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    Generates sample text and prints it.
    """
    model.eval()

    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size
        )

    decoded_text = token_ids_to_text(token_ids, tokenizer)

    print(decoded_text.replace("\n", " "))

    model.train()


# ============================================================
# Advanced Generation (Temperature + Top-k)
# ============================================================

def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    """
    Advanced text generation with:

        - Temperature scaling
        - Top-k sampling
        - Optional EOS stopping
        - Greedy decoding if temperature=0
    """

    for _ in range(max_new_tokens):

        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        # ---------------- Top-k Filtering ----------------
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]

            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits
            )

        # ---------------- Temperature Sampling ----------------
        if temperature > 0.0:
            logits = logits / temperature

            # Numerical stability
            logits = logits - logits.max(dim=-1, keepdim=True).values

            probs = torch.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

        else:
            # Greedy decoding
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Optional EOS stopping
        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


# ============================================================
# Training Loop (Language Modeling)
# ============================================================

def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter,
                       start_context, tokenizer):
    """
    Simple GPT training loop for next-token prediction.
    """

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):

        model.train()

        for input_batch, target_batch in train_loader:

            optimizer.zero_grad()

            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )

            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            # Periodic evaluation
            if global_step % eval_freq == 0:

                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")

        # Generate sample after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


# ============================================================
# Plot Training Loss
# ============================================================

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """
    Plots training vs validation loss across epochs.
    """

    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
        epochs_seen,
        val_losses,
        linestyle="-.",
        label="Validation loss"
    )

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Secondary axis for tokens seen
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.show()