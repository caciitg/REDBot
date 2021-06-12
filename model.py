import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from config import CONFIG


class EmotionRoBERTa(nn.Module):
    def __init__(self):
        super(EmotionRoBERTa, self).__init__()
        config = AutoConfig.from_pretrained(CONFIG.model_name, torchscript=True)
        self.roberta = AutoModel.from_config(config)
        self.fc = nn.Linear(768, CONFIG.num_classes)

    def forward(self, ids, mask, token_type_ids):
        _, features = self.roberta(ids, attention_mask=mask,
                                   token_type_ids=token_type_ids)
        output: torch.Tensor = self.fc(features)
        return output
