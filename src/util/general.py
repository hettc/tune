from random import randint

import torch
import torch.nn as nn
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast

from .model import LoRA

class TrainingTokenizer(LlamaTokenizerFast):
    """ chooses random subsequence of tokens of length <= max_seq_length during tokenization"""
    
    max_seq_length: int = 1024
    def __call__(self, *args, **kwargs):
        tokenized = super().__call__(*args, **kwargs)
        ntokens = len(tokenized["input_ids"])
        if ntokens <= self.max_seq_length:
            return tokenized # full sequence
        
        # random subsequence
        s_idx = randint(0, 10)
        e_idx = s_idx + self.max_seq_length
        return {
            "input_ids": tokenized["input_ids"][s_idx:e_idx],
            "attention_mask": tokenized["attention_mask"][s_idx:e_idx]
        }

def save_model(model, save_path):
    model_data = { 'model_state_dict': model.state_dict() }
    torch.save(model_data, save_path)
def save_adapters(model, save_path):
    raise NotImplementedError("save_adapters not implemented")


def add_adapter(model: nn.Module, r: int, alpha: float = 16, dropout: float = 0):
    """ wrap all nn.Linear layers in a model with LoRA Adapters """

    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRA(module))
        else:
            add_adapter(module, r, alpha, dropout)
    return model


def generate_response(prompt: str, model: nn.Module, tokenizer: nn.Module):
    """ generates a response to a prompt using the model """

    # encode & generate
    encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
    model_inputs = encoded_input.to('cuda')
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # decode & return
    decoded_output = tokenizer.batch_decode(generated_ids)
    return decoded_output[0].replace(prompt, "")


def print_trainable_parameters(model):
    """ prints the number of trainable parameters in the model """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )