# model_base_llama.py
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Callable

class AdaptiveLlamaBase:
    def __init__(self, model_name: str = "unsloth/Llama-3.2-1B", local_dir: str = "./local_llama_free"):
        """
        Lädt ein echtes Llama-3-Modell lokal und injiziert die Hooks 
        in die Llama-Architektur.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(local_dir) and os.path.exists("./local_llama_free"):
            local_dir = "./local_llama_free"
            
        if os.path.exists(local_dir) and os.path.isdir(local_dir):
            print(f"[Llama-System] Lade Llama-Modell lokal von Festplatte: {local_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
            self.model = AutoModelForCausalLM.from_pretrained(local_dir).to(self.device)
        else:
            print(f"[Llama-System] Lade {model_name} erstmals herunter (kann kurz dauern)...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            
            os.makedirs(local_dir, exist_ok=True)
            self.tokenizer.save_pretrained(local_dir)
            self.model.save_pretrained(local_dir)

        self.model.eval()
        self.flush_storage = {"residual_stream": {}, "mlp_activations": {}}
        self.hook_handles = []
        self._register_activation_hooks()
        print(f"[Llama-System] Llama-Hooks erfolgreich aktiv auf: {self.device}")

    def _get_activation_hook(self, layer_idx: int, storage_key: str) -> Callable:
        def hook_fn(module, input_tensor, output_tensor):
            # Llama-Ausgaben sind oft komplexe Tupel, wir isolieren den reinen Tensor
            if isinstance(output_tensor, tuple):
                tensor_data = output_tensor[0].detach().cpu()
            else:
                tensor_data = output_tensor.detach().cpu()
            self.flush_storage[storage_key][layer_idx] = tensor_data
        return hook_fn

    def _register_activation_hooks(self):
        """
        Passt die Hooks an die offizielle Meta-Llama-Architektur an.
        """
        # Llama nutzt 'model.layers' anstelle von GPT-2s 'transformer.h'
        for idx, layer in enumerate(self.model.model.layers):
            
            # 1. Vertikaler Residual Stream Hook
            res_hook = layer.register_forward_hook(self._get_activation_hook(idx, "residual_stream"))
            self.hook_handles.append(res_hook)
            
            # 2. MLP-Aktivierungs-Hook (Llama nutzt standardmäßig die 'down_proj' Schicht im MLP)
            mlp_hook = layer.mlp.down_proj.register_forward_hook(self._get_activation_hook(idx, "mlp_activations"))
            self.hook_handles.append(mlp_hook)
            
        print(f"-> {len(self.hook_handles)} Llama-Schichten-Knoten erfolgreich verknüpft.")

    def run_forward_pass(self, text: str) -> torch.Tensor:
        self.clear_storage()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits

    def get_captured_data(self): return self.flush_storage
    def clear_storage(self):
        self.flush_storage["residual_stream"].clear()
        self.flush_storage["mlp_activations"].clear()

    def remove_all_hooks(self):
        for handle in self.hook_handles: handle.remove()
        self.hook_handles.clear()

