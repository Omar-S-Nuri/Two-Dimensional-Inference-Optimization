# analyzer.py
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

class LogitLensFlowAnalyzer:
    def __init__(self, base_framework):
        """
        Initialisiert den Analysator und verknüpft ihn mit dem aktiven Basis-Modell.
        """
        self.base = base_framework
        self.global_mlp_activation_counters = {}
        self.total_processed_tokens = 0

    def compute_layer_logits(self, layer_idx: int, res_tensor: torch.Tensor) -> torch.Tensor:
        """
        Der universelle Logit-Lens-Mechanismus für GPT-2 und Llama 3.2.
        Nimmt den Zustand einer Zwischenschicht und projiziert ihn vorzeitig durch den finalen Ausgabekopf.
        """
        res_tensor = res_tensor.to(self.base.device)
        
        # Falls es sich um die LlamaBridge handelt, müssen wir auf das echte Modell zugreifen
        actual_model = self.base.model
        
        with torch.no_grad():
            # UNIVERSAL-WEICHENSTELLUNG FÜR DIE SCHICHT-NORMALISIERUNG (LayerNorm / RMSNorm)
            if hasattr(actual_model, "transformer"):
                # Pfad für GPT-2
                ln_f = actual_model.transformer.ln_f
                normalized_state = ln_f(res_tensor)
            elif hasattr(actual_model, "model") and hasattr(actual_model.model, "norm"):
                # Pfad für Llama 3 / 3.2 (Meta nutzt 'model.norm' statt 'transformer.ln_f')
                ln_f = actual_model.model.norm
                normalized_state = ln_f(res_tensor)
            else:
                # Fallback, falls keine Normalisierung am Ende benötigt wird
                normalized_state = res_tensor
                
            # Projizieren in das Vokabular über den Unembedding-Kopf (lm_head)
            lm_head = actual_model.lm_head
            logits = lm_head(normalized_state)
            
        return logits

    def get_top_k_words_for_layer(self, layer_idx: int, res_tensor: torch.Tensor, top_k: int = 100) -> List[List[Tuple[str, float]]]:
        """
        Extrahiert die Top-K Wörter (Tokens) und deren Wahrscheinlichkeiten pro Schicht.
        Gibt eine Liste für jeden Token im Eingabesatz zurück.
        """
        logits = self.compute_layer_logits(layer_idx, res_tensor)
        probabilities = F.softmax(logits, dim=-1)
        
        batch_probs = probabilities 
        num_tokens = batch_probs.size(0)
        
        layer_results = []
        for t_idx in range(num_tokens):
            token_probs = batch_probs[t_idx]
            top_probs, top_ids = torch.topk(token_probs, top_k)
            
            token_predictions = []
            for p, idx in zip(top_probs.tolist(), top_ids.tolist()):
                word = self.base.tokenizer.decode([idx])
                token_predictions.append((word, p))
                
            layer_results.append(token_predictions)
            
        return layer_results

    def update_flow_statistics(self, captured_data: Dict[str, Dict[int, torch.Tensor]]):
        """
        Misst und aggregiert die Aktivierungsenergie der MLP-Neuronen über die Epoche.
        """
        mlp_data = captured_data["mlp_activations"]
        
        for layer_idx, activation_tensor in mlp_data.items():
            # Wir reduzieren über Batch- und Token-Dimensionen, um die Gesamtenergie pro Neuron zu messen
            if len(activation_tensor.shape) == 3:
                activations_per_neuron = activation_tensor.sum(dim=0).sum(dim=0)
            else:
                activations_per_neuron = activation_tensor.sum(dim=0)
            
            if layer_idx not in self.global_mlp_activation_counters:
                self.global_mlp_activation_counters[layer_idx] = torch.zeros_like(activations_per_neuron)
                
            self.global_mlp_activation_counters[layer_idx] += torch.abs(activations_per_neuron)
            
        self.total_processed_tokens += 1

    def get_top_winner_nodes(self, layer_idx: int, top_x: int = 10) -> List[Tuple[int, float]]:
        """
        Gibt die Top-X Gewinner-Knoten einer Schicht zurück.
        """
        if layer_idx not in self.global_mlp_activation_counters or self.total_processed_tokens == 0:
            return []
            
        avg_activations = self.global_mlp_activation_counters[layer_idx] / self.total_processed_tokens
        top_values, top_indices = torch.topk(avg_activations, top_x)
        
        return list(zip(top_indices.tolist(), top_values.tolist()))
