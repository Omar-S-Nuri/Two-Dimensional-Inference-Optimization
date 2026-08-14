# compiler_llama.py
import torch
import torch.nn as nn
from typing import Dict, Tuple, List

class LlamaOperatorFusionCompiler:
    def __init__(self, base_framework, shortcut_engine):
        """
        Initialisiert den Operator-Fusion-Compiler für Llama 3.2 Architecture (v2.0).
        Fuziert komplexe SwiGLU-Blockprojektionen zu quadratischen Super-Matrizen.
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
        Durchläuft alle validierten Llama-Ketten und fuziert deren Gewichtsmatrizen.
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
        # Überprüfen, ob wir die Bridge oder das native Modell nutzen
        if hasattr(actual_model, "model"):
            llama_net = actual_model.model
        else:
            llama_net = actual_model

        device = llama_net.device
        hidden_dim = llama_net.config.hidden_size
        
        # Typengesicherter Loop durch alle berechneten Brücken
        for start, end in chains:
            if isinstance(self.fused_shortcuts, dict) and (start, end) not in self.fused_shortcuts:
                print(f"[Compiler Llama] 🛠️ Fuziere SwiGLU-Matrix-Komplex für Route: L{start} -> L{end}...")
                
                try:
                    layers = llama_net.model.layers
                    fused_weight = torch.eye(hidden_dim, device=device)
                    
                    # Iterative mathematische Block-Kollabierung für die Llama-Architektur
                    for layer_idx in range(start, end):
                        current_layer = layers[layer_idx]
                        
                        # Llama nutzt SwiGLU: up_proj, gate_proj und down_proj im MLP-Block
                        w_up = current_layer.mlp.up_proj.weight      # (intermediate_size, hidden_dim)
                        w_down = current_layer.mlp.down_proj.weight  # (hidden_dim, intermediate_size)
                        
                        # Approximation der nicht-linearen SwiGLU-Aktivierung zu einer quadratischen Block-Matrix
                        # Matrix-Dimension: (hidden_size, hidden_size)
                        w_block = torch.matmul(w_down, w_up)
                        
                        # Akkumulation der Transformationen im Residual Stream
                        fused_weight = torch.matmul(fused_weight, w_block)
                    
                    # Einbetten der fusionierten Super-Matrix in ein natives PyTorch-Modul
                    # Llama nutzt standardmäßig keinen Bias in den linearen Projektionen
                    linear_bridge = nn.Linear(hidden_dim, hidden_dim, bias=False)
                    with torch.no_grad():
                        linear_bridge.weight.copy_(fused_weight)
                        
                    linear_bridge.to(device, dtype=llama_net.dtype)
                    self.fused_shortcuts[(start, end)] = linear_bridge
                    print(f"[Compiler Llama] 🟢 Llama Fused Super-Matrix erfolgreich registriert: L{start} -> L{end} ({hidden_dim}x{hidden_dim})")
                    
                except Exception as e:
                    print(f"[Compiler Llama] ⚠️ Überspringe Fusion für L{start}->L{end} aufgrund eines Matrix-Fehlers: {e}")
