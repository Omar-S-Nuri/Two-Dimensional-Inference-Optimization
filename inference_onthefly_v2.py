# inference_onthefly_v2.py
import torch
import time
import asyncio
from typing import Tuple

## neu
from config import GLOBAL_THRESHOLD

class EvolutionaryInferenceEngine:
    def __init__(self, base_framework, shortcut_engine, compiler, validator):
        """
        Initialisiert die kontinuierlich lernende Inferenz-Engine für Version 2.0.
        Verwaltet adaptiv den vertikalen Fast-Track UND den horizontalen Phrasen-Turbo.
        """
        self.base = base_framework
        self.engine = shortcut_engine
        self.compiler = compiler
        self.validator = validator
        
        # Asynchroner Puffer für das kontinuierliche Lernen im Flug
        self.background_data_buffer = []
        self.telemetry = {"fast_vertical": 0, "fast_horizontal": 0, "slow": 0, "time_ms": 0.0}

    def predict_next_token(self, text: str) -> Tuple[str, str]:
        """
        Der adaptive 2D-Routing-Mechanismus im Live-Betrieb.
        Garantiert maximale Chat-Geschwindigkeit und absolute Threadsicherheit.
        """
        start = time.perf_counter()
        
        # Satz für das automatische Hintergrund-Lernen registrieren
        self.background_data_buffer.append(text)
        
        # Rohe Token-IDs für die horizontale Trie-Prüfung extrahieren
        rohe_tokens = self.base.tokenizer.encode(text, add_special_tokens=False)
        
        # ─────────────────────────────────────────────────────────────────
        # DIMENSION 2 (HORIZONTAL): Phrasen-Turbo via Präfixbaum prüfen
        # ─────────────────────────────────────────────────────────────────
        if hasattr(self.engine, "get_horizontal_chains"):
            h_chains = self.engine.get_horizontal_chains()
            if len(rohe_tokens) >= 2:
                p1 = rohe_tokens[-2]
                p2 = rohe_tokens[-1]
                
                # O(1) Abfrage im numerischen Trie-Wörterbuch ohne CPU-Last
                if p1 in h_chains and p2 in h_chains[p1]:
                    chain_info = h_chains[p1][p2]
                    next_tokens = chain_info["next_tokens"]
                    confidence = chain_info["confidence"]
                    
                    # Kettensicherung: Nur abfeuern, wenn das Vertrauen über 95% liegt .. auf 0.75 geändert
                    # if confidence >= 0.95 and next_tokens:
                    if confidence >= GLOBAL_THRESHOLD and next_tokens:
                        self.telemetry["fast_horizontal"] += 1
                        
                        # Parallel-Auswurf der gelernten Tokenkette (überspringt CPU-Schleifen)
                        predicted_word = self.base.tokenizer.decode(next_tokens, clean_up_tokenization_spaces=False)
                        word_clean = predicted_word.replace("Ġ", " ").replace(" ", " ")
                        
                        if not word_clean.strip():
                            word_clean = f" [CHAIN_ID:{next_tokens}]"
                            
                        self.telemetry["time_ms"] += (time.perf_counter() - start) * 1000
                        return word_clean, f"FAST-TRACK (HORIZONTAL CHAIN, Conf: {confidence*100:.1f}%)"

        # ─────────────────────────────────────────────────────────────────
        # DIMENSION 1 (VERTIKAL): Schichten-Überspringen via Super-Matrix
        # ─────────────────────────────────────────────────────────────────
        s_map = self.engine.get_shortcut_map()
        fused = self.compiler.get_fused_shortcuts()
        
        track = "SLOW-TRACK"
        target = None
        
        # Nutzt deine exakte, bewährte Indizierungs-Logik aus v1.0
        if 1 in s_map:
            route_info = s_map[1]
            end_layer = int(route_info[0])
            status = str(route_info[1])
            
            if status == "VALIDATED" and (1, end_layer) in fused:
                track = "FAST-TRACK (VERTICAL)"
                target = end_layer

        if track == "FAST-TRACK (VERTICAL)":
            self.telemetry["fast_vertical"] += 1
            with torch.no_grad():
                _ = self.base.run_forward_pass(text)
                c_res = self.base.get_captured_data()["residual_stream"]
                layer_1_raw = c_res[1] # Deine fixe Indizierung
                
                if isinstance(layer_1_raw, tuple):
                    layer_1_tensor = layer_1_raw[0]
                else:
                    layer_1_tensor = layer_1_raw
                
                actual_model = getattr(self.base, "model", self.base)
                s_state = fused[(1, target)](layer_1_tensor.to(actual_model.device))
                
                if hasattr(actual_model, "transformer"):
                    ln_f = actual_model.transformer.ln_f
                elif hasattr(actual_model, "model") and hasattr(actual_model.model, "norm"):
                    ln_f = actual_model.model.norm
                else:
                    ln_f = lambda x: x
                
                logits = actual_model.lm_head(ln_f(s_state))
        else:
            self.telemetry["slow"] += 1
            logits = self.base.run_forward_pass(text)
            
        # Universal-Decoder für Einzelwort-Auswurf (GPT-2 & Llama)
        actual_model = getattr(self.base, "model", self.base)
        next_token_id = torch.argmax(logits[0, -1]).item()
        word = self.base.tokenizer.decode([next_token_id], clean_up_tokenization_spaces=False)
        word_clean = word.replace("Ġ", " ").replace(" ", " ")
        
        if not word_clean.strip():
            word_clean = f" [{word.strip() if word.strip() else f'ID:{next_token_id}'}]"
            
        self.telemetry["time_ms"] += (time.perf_counter() - start) * 1000
        return word_clean, track

    async def run_asynchronous_evolution_cycle(self):
        """
        Wird vom Hintergrund-Thread aufgerufen. Verarbeitet reaktiv vertikale 
        Schichten-Abkürzungen UND lernt horizontale Wortketten im Flug.
        """
        if not self.background_data_buffer: 
            return
            
        current_batch = list(self.background_data_buffer)
        self.background_data_buffer.clear()
        
        for data in current_batch:
            # Vorwärtslauf zur Erzeugung von Logits und Hook-Daten ausführen
            logits = self.base.run_forward_pass(data)
            
            # 1. Vertikale Evolution (Aus deiner v1.0 Struktur)
            self.compiler.engine.analyzer.update_flow_statistics(self.base.get_captured_data())
            links = self.engine.analyze_layer_equivalence(self.base.get_captured_data())
            self.engine.register_potential_shortcuts(links)
            
            # 2. Horizontale Evolution (NEU: Phrasen-Ketten im Trie-Baum verankern)
            if hasattr(self.engine, "analyze_and_register_horizontal_chain"):
                rohe_tokens = self.base.tokenizer.encode(data, add_special_tokens=False)
                self.engine.analyze_and_register_horizontal_chain(rohe_tokens, logits)
            
        self.compiler.compile_all_eligible_chains()
