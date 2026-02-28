# ============================================================
# gpt2llm_functions.py
# Utility functions for instruction fine-tuning and data prep
# ============================================================

import json
import os
import requests
import torch


# ============================================================
# Download & Load JSON File
# ============================================================

def download_and_load_file(file_path, url):
    """
    Downloads a JSON file from a URL (if not already present),
    then loads and returns its contents.

    Args:
        file_path (str): Local path to save/load file
        url (str): Remote URL to download from

    Returns:
        data (dict or list): Parsed JSON content
    """

    # Download only if file does not already exist
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text

        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    # Load JSON file
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


# ============================================================
# Instruction Formatting
# ============================================================

def format_input(entry):
    """
    Formats instruction-style training examples.

    Expected entry format:
        {
            "instruction": "...",
            "input": "...",
            "output": "..."
        }

    Produces structured prompt:

        Below is an instruction...
        ### Instruction:
        ...
        ### Input:
        ...
    """

    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    # Include input section only if it exists
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


# ============================================================
# Custom Collate Function (Instruction Fine-Tuning)
# ============================================================

def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """
    Custom collate function for instruction fine-tuning.

    Converts a batch of token ID sequences into:
        inputs  (shifted right)
        targets (shifted left)

    Key Features:
        - Pads sequences to batch max length
        - Appends EOS token
        - Masks padding tokens with ignore_index
        - Optionally truncates to allowed_max_length
        - Moves tensors to specified device
    """

    # Determine longest sequence in batch (+1 for EOS)
    batch_max_length = max(len(item) + 1 for item in batch)

    inputs_lst, targets_lst = [], []

    for item in batch:
        # Copy original token sequence
        new_item = item.copy()

        # Append EOS/pad token
        new_item += [pad_token_id]

        # Pad to batch maximum length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )

        # Create shifted inputs and targets
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # Mask padding tokens in targets
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()

        # Keep first pad token as valid, mask remaining
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # Optional truncation
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Stack batch tensors
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor