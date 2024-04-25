import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

class LoRA(nn.Module):
    """ wraps a nn.Linear layer with LoRA Adapters """

    def __init__(self, layer: nn.Linear, r: int, alpha: float = 16, dropout: float = 0):
        super().__init__()
        assert isinstance(layer, nn.Linear), "LoRA adapters only wrap nn.Linear layers"
        self.base_layer = layer
        self.alpha = alpha
        self.r = r

        layers = []
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))
        layers.append(nn.Linear(layer.in_features, r, device=layer.weight.device))
        layers.append(nn.Linear(r, layer.out_features, device=layer.weight.device))
        self.lora_net = nn.Sequential(*layers)

    def forward(self, x):
        base_out = self.base_layer(x)

        lora_in = x.to(
            device=self.lora_net[-1].weight.device,
            dtype=self.lora_net[-1].weight.dtype
        )
        lora_out = self.lora_net(lora_in).to(base_out.dtype)
        return (lora_out * (self.alpha / self.r)) + self.base_layer(x)


def build_tuneable_hf_model(untuned_model_id: str, quantized: bool, device_map: str = 'auto'):
    # load raw model
    if quantized:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    model = AutoModelForCausalLM.from_pretrained(
        untuned_model_id,
        device_map=device_map,
        use_cache=False,
        torch_dtype=torch.bfloat16,
        quantization_config=nf4_config if quantized else None,
        trust_remote_code=True
    )

    # add adapters
    for param in model.parameters():
        param.requires_grad = False
    def get_adapter(layer):
        return LoRA(layer, r=32, alpha=16, dropout=0.1)
    for layer in model.model.layers:
        layer.attn.q_proj = get_adapter(layer.attn.q_proj)
        layer.attn.k_proj = get_adapter(layer.attn.k_proj)
        layer.attn.v_proj = get_adapter(layer.attn.v_proj)
        layer.attn.o_proj = get_adapter(layer.attn.o_proj)
        layer.moe_block.gate = get_adapter(layer.moe_block.gate)
        for i in [0, 1]:
            layer.moe_block.experts[i].linear_v = get_adapter(layer.moe_block.experts[i].linear_v)
            layer.moe_block.experts[i].linear_1 = get_adapter(layer.moe_block.experts[i].linear_1)
            layer.moe_block.experts[i].linear = get_adapter(layer.moe_block.experts[i].linear)
    return model


def build_tuned_grok(model_pth: str, quantized: bool, device_map: str = 'auto'):
    model = build_tuneable_hf_model(quantized, device_map)
    loaded_data = torch.load(model_pth)
    model.load_state_dict(loaded_data['model_state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    return model.eval()
