# model_base.py
import os
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Dict, List, Callable

class AdaptiveTransformerBase:
    def __init__(self, model_name: str = "gpt2", local_dir: str = "./local_gpt2"):
        """
        Lädt das Modell lokal herunter, falls nicht vorhanden, und verwaltet die Hooks.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Prüfen, ob das Modell bereits dauerhaft lokal auf der Festplatte existiert
        if os.path.exists(local_dir) and os.path.isdir(local_dir):
            print(f"[System] Liede Modell lokal von der Festplatte: {local_dir}")
            self.tokenizer = GPT2Tokenizer.from_pretrained(local_dir)
            self.model = GPT2LMHeadModel.from_pretrained(local_dir).to(self.device)
        else:
            print(f"[System] Lade Modell erstmals aus dem Web und speichere dauerhaft in {local_dir}...")
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
            # Dauerhaft lokal sichern
            os.makedirs(local_dir, exist_ok=True)
            self.tokenizer.save_pretrained(local_dir)
            self.model.save_pretrained(local_dir)

        self.model.eval()
        self.flush_storage = {"residual_stream": {}, "mlp_activations": {}}
        self.hook_handles = []
        self._register_activation_hooks()

    def _get_activation_hook(self, layer_idx: int, storage_key: str) -> Callable:
        def hook_fn(module, input_tensor, output_tensor):
            self.flush_storage[storage_key][layer_idx] = output_tensor.detach().cpu()
        return hook_fn

    def _register_activation_hooks(self):
        for idx, layer in enumerate(self.model.transformer.h):
            self.hook_handles.append(layer.register_forward_hook(self._get_activation_hook(idx, "residual_stream")))
            self.hook_handles.append(layer.mlp.act.register_forward_hook(self._get_activation_hook(idx, "mlp_activations")))

    def run_forward_pass(self, text: str) -> torch.Tensor:
        self.clear_storage()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits

    def get_captured_data(self) -> Dict[str, Dict[int, torch.Tensor]]:
        return self.flush_storage

    def clear_storage(self):
        self.flush_storage["residual_stream"].clear()
        self.flush_storage["mlp_activations"].clear()

    def remove_all_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
