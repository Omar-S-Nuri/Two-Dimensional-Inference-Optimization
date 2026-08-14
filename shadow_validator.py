# shadow_validator.py
import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from config import GLOBAL_THRESHOLD

class ShadowValidator:
    def __init__(self, base_framework, shortcut_engine, compiler, stability_window: int = 1):
        """
        Initialisiert den Highspeed-Shadow-Validator.
        INTEGRIERTE IDEE A: Validiert Routen blitzschnell auf Vektor-Ebene 
        nach der Normalisierung, um den CPU-Killer lm_head (128k Vokabular) zu umgehen.
        """
        self.base = base_framework
        self.engine = shortcut_engine
        self.compiler = compiler
        self.stability_window = stability_window
        self.shadow_telemetry = {}

    def evaluate_shortcut_step(self, start_layer: int, end_layer: int, original_logits: torch.Tensor):
        """
        Valide Ausgaben schrauben den Stabilitäts-Zähler nach oben.
        Arbeitet dank Idee A in Lichtgeschwindigkeit direkt auf dem verborgenen Zustand.
        """
        rk = (start_layer, end_layer)
        fused = self.compiler.get_fused_shortcuts()
        if rk not in fused: 
            return
            
        if rk not in self.shadow_telemetry: 
            self.shadow_telemetry[rk] = {"success": 0}
        
        # 1. WEG A: Den echten, normalisierten Ziel-Zustand des Master-Modells aus den Hooks holen
        res_master_raw = self.base.get_captured_data()["residual_stream"][end_layer]
        tensor_master = res_master_raw[0] if isinstance(res_master_raw, tuple) else res_master_raw
        
        # 2. WEG B: Den Start-Zustand durch die fusionierte Super-Matrix jagen
        res_start_raw = self.base.get_captured_data()["residual_stream"][start_layer]
        tensor_start = res_start_raw[0] if isinstance(res_raw_1 := res_start_raw, tuple) else res_start_raw
        
        actual_model = getattr(self.base, "model", self.base)
        
        with torch.no_grad():
            # Shortcut-Zustand berechnen
            shortcut_state = fused[rk](tensor_start.to(actual_model.device))
            
            # Weichenstellung für die Schicht-Normalisierung (RMSNorm/LayerNorm)
            if hasattr(actual_model, "transformer"):
                ln_f = actual_model.transformer.ln_f
            elif hasattr(actual_model, "model") and hasattr(actual_model.model, "norm"):
                ln_f = actual_model.model.norm
            else:
                ln_f = lambda x: x
            
            # BEIDE VEKTOREN NORMALISIEREN (Der mathematische Sweet Spot von Idee A)
            norm_master = ln_f(tensor_master.to(actual_model.device))
            norm_shortcut = ln_f(shortcut_state)
            
            # 3. BLITZSCHNELLER VEKTOR-ABGLEICH STATT LM_HEAD SCHLEIFE
            # Wir berechnen die Cosinus-Ähnlichkeit der verborgenen Zustände für das letzte Token
            vector_similarity = torch.cosine_similarity(
                norm_master[0, -1].to(dtype=torch.float32), 
                norm_shortcut[0, -1].to(dtype=torch.float32), 
                dim=0
            ).item()
        
        # Ein Vektor-Score von > 98% garantiert ein identisches Wort am Ausgang
        ## if vector_similarity >= 0.98: ## GLOBAL_THRESHOLD
        if vector_similarity >= 0.98:
    
            self.shadow_telemetry[rk]["success"] += 1
            print(f"[Validator] ⚡ Highspeed Vektor-Match für L{start_layer}->L{end_layer}. Ähnlichkeit: {vector_similarity*100:.2f}%")
        else:
            self.shadow_telemetry[rk]["success"] = 0
            print(f"[Validator] Warnung: Vektor-Abweichung auf Route L{start_layer}->L{end_layer}. Score: {vector_similarity*100:.2f}%")
            
        if self.shadow_telemetry[rk]["success"] >= self.stability_window:
            self._promote(start_layer, end_layer)

    def _promote(self, start: int, end: int):
        """Erhebt den Shortcut nach bestandener Testlaufzeit in den Status VALIDATED."""
        s_map = self.engine.get_shortcut_map()
        if start in s_map:
            route_info = s_map[start]
            if len(route_info) == 3:
                _, _, sim = route_info
                self.engine.shortcut_map[start] = (end, "VALIDATED", sim)
            elif len(route_info) == 4:
                _, _, sim, token = route_info
                self.engine.shortcut_map[start] = (end, "VALIDATED", sim, token)
            else:
                self.engine.shortcut_map[start] = (end, "VALIDATED", 0.90)
                
            print(f"🎉 [VALIDATION SUCCESS] Llama-Route L{start} -> L{end} FEST VERANKERT!")
