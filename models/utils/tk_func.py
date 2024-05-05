import torch 
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from datasets import load_dataset
import torch.nn.functional as F


def tokenizer_fn(examples, tokenizer, max_length):
    inputs = tokenizer(examples["completion"], max_length=max_length, padding="max_length", truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["prompt"], max_length=max_length, padding="max_length", truncation=True)
    inputs["labels"] = labels["input_ids"]
    return inputs

# 将数据转换为 PyTorch Tensor
def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])
    return {"input_ids": input_ids, "labels": labels}