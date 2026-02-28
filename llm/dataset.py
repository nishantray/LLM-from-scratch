# ============================================================
# dataset.py
# Dataset classes and data-loading utilities
# ============================================================

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import tiktoken
from .gpt2llm_functions import *


# ============================================================
# GPT Language Modeling Dataset
# ============================================================

class GPTDatasetV1(Dataset):
    """
    Dataset for next-token prediction training.

    Splits a large text corpus into overlapping chunks.
    Each input sequence predicts the next token sequence.
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Encode entire text corpus into token IDs
        token_ids = tokenizer.encode(txt)

        # Create sliding window chunks
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# ============================================================
# Spam Classification Dataset
# ============================================================

class SpamDataset(Dataset):
    """
    Dataset for binary spam classification.

    - Reads CSV file containing:
        Label | Text
    - Tokenizes text
    - Pads sequences to uniform length
    - Returns (input_ids, label)
    """

    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        # Load CSV file
        self.data = pd.read_csv(csv_file)

        # Tokenize all text samples
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        # Determine maximum sequence length
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

            # Truncate sequences if max_length provided
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Pad all sequences to max_length
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]

        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        """
        Computes maximum token length across dataset.
        """
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


# ============================================================
# GPT Dataloader Helper
# ============================================================

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128,
                         shuffle=True, drop_last=True, num_workers=0):
    """
    Creates DataLoader for GPT language modeling dataset.
    """
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


# ============================================================
# Dataset Split Utility
# ============================================================

def random_split(df, train_frac, validation_frac):
    """
    Randomly splits dataframe into:
        - Train
        - Validation
        - Test

    Fractions are relative to full dataset.
    """
    # Shuffle dataset
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


# ============================================================
# Instruction Fine-Tuning Dataset
# ============================================================

class InstructionDataset(Dataset):
    """
    Dataset for instruction-based fine-tuning.

    Each entry:
        {
            "instruction": ...,
            "input": ...,
            "output": ...
        }

    Combines instruction + input + formatted response.
    """

    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []

        for entry in data:
            # Format instruction + input
            instruction_plus_input = format_input(entry)

            # Append expected response section
            response_text = f"\n\n### Response:\n{entry['output']}"

            # Full training example
            full_text = instruction_plus_input + response_text

            # Tokenize full sequence
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)