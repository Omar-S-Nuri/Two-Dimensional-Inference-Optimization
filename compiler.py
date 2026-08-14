# compiler.py
import torch
import torch.nn as nn
from typing import Dict, Tuple, List

class OperatorFusionCompiler:
    def __init__(self, base_framework, shortcut_engine):
        """
        Initialisiert den Operator-Fusion-Compiler für GPT-2 (v2.0).
        Fuziert zersplitterte lineare Layer-Komplexe zu kompakten Super-Matrizen.
        """
        self.base = base_framework
        self.engine = shortcut_engine
        self.fused_shortcuts: Dict[Tuple[int, int], nn.Linear] = {}

    def get_fused_shortcuts(self) -> Dict[Tuple[int, int], nn.Linear]:
        return self.fused_shortcuts

    def find_continuous_chains(self) -> List[Tuple[int, int]]:
        """
        Analysiert die shortcut_map und findet kontinuierliche Brückenketten.
        CRITICAL REPAIR: Filtert Strings und andere ungültige Typen rigoros heraus.
        """
        shortcut_map = self.engine.get_shortcut_map()
        chains = []
        
        # Sicherstellen, dass wir ein gültiges Dictionary vorliegen haben
        if isinstance(shortcut_map, dict) and "vertical" in shortcut_map:
            target_map = shortcut_map["vertical"]
        else:
            target_map = shortcut_map

        # Filtert alle Schlüssel heraus, die keine reinen Layer-Ganzzahlen (int) sind
        valid_keys = [k for k in target_map.keys() if isinstance(k, int)]
        
        for start in sorted(valid_keys):
            route_info = target_map[start]
            
            # Überprüfen, ob route_info ein gültiges Tupel/Liste aus Version 1.0 ist
            if isinstance(route_info, (tuple, list)):
                end = int(route_info[0])
                status = str(route_info[1])
                
                # Wir kompilieren nur Routen, die den Validierungs-Prozess überstanden haben
                if status == "VALIDATED":
                    chains.append((start, end))
                    
        return chains

    def compile_all_eligible_chains(self):
        """
        Durchläuft alle validierten Ketten und fuziert deren Gewichtsmatrizen.
        CRITICAL REPAIR: Verhindert unhashable type: 'dict' Konflikte durch 2D-Mischmasch.
        """
        # Falls self.fused_shortcuts fälschlicherweise verschachtelt oder kein Dict ist, bereinigen
        if not hasattr(self, "fused_shortcuts") or not isinstance(self.fused_shortcuts, dict):
            self.fused_shortcuts = {}
        elif "vertical" in self.fused_shortcuts or "horizontal" in self.fused_shortcuts:
            self.fused_shortcuts = {}

        chains = self.find_continuous_chains()
        if not chains:
            return

        actual_model = getattr(self.base, "model", self.base)
        device = actual_model.device
        
        # Typengesicherter Loop durch alle berechneten Brücken
        for start, end in chains:
            if isinstance(self.fused_shortcuts, dict) and (start, end) not in self.fused_shortcuts:
                print(f"[Compiler] 🛠️ Fuziere Matrix-Komplex für Route: L{start} -> L{end}...")
                
                try:
                    # Mathematische Extraktion der beteiligten Gewichtsmatrizen (GPT-2 Architektur)
                    layers = actual_model.transformer.h
                    
                    # Start-Gewichte als Fundament (Identitäts-Transformation simulieren)
                    hidden_dim = actual_model.config.n_embd
                    fused_weight = torch.eye(hidden_dim, device=device)
                    fused_bias = torch.zeros(hidden_dim, device=device)
                    
                    # Iterative Multiplikation aller Gewichtsmatrizen der übersprungenen Schichten
                    for layer_idx in range(start, end):
                        current_layer = layers[layer_idx]
                        
                        # Extraktion von c_attn (Attention) und c_fc/c_proj (MLP)
                        w_mlp1 = current_layer.mlp.c_fc.weight     # (hidden_dim, 4 * hidden_dim)
                        b_mlp1 = current_layer.mlp.c_fc.bias
                        w_mlp2 = current_layer.mlp.c_proj.weight   # (4 * hidden_dim, hidden_dim)
                        b_mlp2 = current_layer.mlp.c_proj.bias
                        
                        # Lineare Fusions-Approximation (Post-Training Operator Fusion)
                        fused_weight = torch.matmul(fused_weight, w_mlp1)
                        fused_weight = torch.matmul(fused_weight, w_mlp2)
                        
                        fused_bias = torch.matmul(fused_bias, w_mlp1) + b_mlp1
                        fused_bias = torch.matmul(fused_bias, w_mlp2) + b_mlp2
                    
                    # Einbetten der fusionierten Super-Matrix in ein natives PyTorch-Modul
                    linear_bridge = nn.Linear(hidden_dim, hidden_dim, bias=True)
                    with torch.no_grad():
                        linear_bridge.weight.copy_(fused_weight.t())
                        linear_bridge.bias.copy_(fused_bias)
                        
                    linear_bridge.to(device)
                    self.fused_shortcuts[(start, end)] = linear_bridge
                    print(f"[Compiler] 🟢 Fused Super-Matrix erfolgreich registriert: L{start} -> L{end} ({hidden_dim}x{hidden_dim})")
                    
                except Exception as e:
                    print(f"[Compiler] ⚠️ Überspringe Fusion für L{start}->L{end} aufgrund eines Matrix-Fehlers: {e}")
