import torch
from transformers import AutoTokenizer


class CONFIG:
    model_name = "roberta-base"
    num_classes = 5
    max_len = 200
    model_path = "checkpoints/model.bin"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, torchscript=True)
