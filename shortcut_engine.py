# shortcut_engine.py
import torch
import torch.nn.functional as F
import os
from config import GLOBAL_THRESHOLD, EARLY_ABORT_THRESHOLD

class ShortcutSynthesizer:
    """
    Der Architekt deines kognitiven Abkürzungsnetzwerks.
    Verwaltet das zweidimensionale Inferenz-Gedächtnis im RAM und auf der Festplatte.
    """
    def __init__(self, base_model, analyzer, convergence_threshold=None, storage_file="shortcuts.pt"):
        self.model = base_model
        self.analyzer = analyzer
        self.convergence_threshold = convergence_threshold if convergence_threshold is not None else GLOBAL_THRESHOLD
        self.storage_file = storage_file
        
        # Das zweidimensionale kognitive Inferenz-Gedächtnis im RAM
        self.shortcut_map = {}       # Dimension 1: Vertikale Schichten-Shortcuts {start_layer: (end_layer, status, confidence)}
        self.horizontal_chains = {}  # Dimension 2: Horizontaler Phrasen-Trie {token_id: {next_token_id: {...}}}
        
        # Vorhandenes Wissen inkrementell beim Start laden
        self.load_shortcuts()

    def get_shortcut_map(self):
        """Liefert das vertikale Brückennetzwerk für den Compiler."""
        return self.shortcut_map

    def get_horizontal_chains(self):
        """Liefert den horizontalen Trie-Cache für die InferenceEngine."""
        return self.horizontal_chains

    def _get_layer_tensor(self, raw):
        """Hilfsmethode: Gibt den Tensor aus einem rohen Hook-Output zurück."""
        return raw[0] if isinstance(raw, tuple) else raw

    def analyze_layer_equivalence(self, captured_data):
        """
        [Dimension 1 - Vertikal]: Durchsucht den Residual Stream nach Schichten-Äquivalenzen.
        Bewährte Methode: Vergleicht projizierte Logit-Verteilungen (wie in V1).
        """
        res_stream = captured_data.get("residual_stream", captured_data)
        if not res_stream or len(res_stream) < 3:
            return []

        layer_keys = sorted([k for k in res_stream.keys() if isinstance(k, int)])
        num_layers = len(layer_keys)

        # 1. EARLY-ABORT: Baseline-Ähnlichkeit zwischen Schicht 0 und mittlerer Schicht
        try:
            first_layer = layer_keys[0]
            mid_layer   = layer_keys[num_layers // 2]

            t_first = self._get_layer_tensor(res_stream[first_layer])
            t_mid   = self._get_layer_tensor(res_stream[mid_layer])

            log_first = self.analyzer.compute_layer_logits(first_layer, t_first)[0, -1]
            log_mid   = self.analyzer.compute_layer_logits(mid_layer,   t_mid  )[0, -1]

            baseline_sim = torch.cosine_similarity(
                F.softmax(log_first, dim=-1),
                F.softmax(log_mid,   dim=-1),
                dim=0
            ).item()

            if baseline_sim < EARLY_ABORT_THRESHOLD:
                return []   # Semantisches Chaos -> kein vertikaler Scan
        except Exception:
            pass  # Sicherheits-Fallback

        # 2. Systematischer Tiefen-Scan über alle Schichtenpaare
        discovered_links = []
        for i in range(num_layers):
            start = layer_keys[i]
            try:
                t_start  = self._get_layer_tensor(res_stream[start])
                log_start = self.analyzer.compute_layer_logits(start, t_start)[0, -1]
                prob_start = F.softmax(log_start, dim=-1)
            except Exception:
                continue

            for j in range(i + 1, num_layers):
                end = layer_keys[j]
                try:
                    t_end  = self._get_layer_tensor(res_stream[end])
                    log_end = self.analyzer.compute_layer_logits(end, t_end)[0, -1]
                    prob_end = F.softmax(log_end, dim=-1)

                    similarity = torch.cosine_similarity(prob_start, prob_end, dim=0).item()

                    if similarity >= self.convergence_threshold:
                        discovered_links.append((start, end, similarity))
                except Exception:
                    continue

        return discovered_links

    def register_potential_shortcuts(self, links):
        """Registriert gefundene vertikale Brücken als PENDING im RAM, sofern nicht bereits VALIDATED."""
        for start, end, sim in links:
            # Bestehende validierte Brücken werden niemals durch PENDING-Zustände überschrieben
            if start in self.shortcut_map:
                current_status = self.shortcut_map[start]
                if isinstance(current_status, tuple) and len(current_status) >= 2:
                    if current_status[1] == "VALIDATED":
                        continue
            
            # Neue Brücke im Status PENDING (wartend) anlegen
            self.shortcut_map[start] = (end, "PENDING", float(sim))


    def analyze_and_register_horizontal_chain(self, raw_tokens, logits):
        """
        [Dimension 2 - Horizontal]: Scannt die gesamte Sequenz Token für Token.
        Schmiedet bei hoher Vorhersagesicherheit feste Phrasen-Ketten im Trie-Baum.
        """
        # Squeeze, falls Batch-Dimension vorhanden ist (macht aus [1, N, Vocab] -> [N, Vocab])
        if len(logits.shape) == 3:
            logits = logits.squeeze(0)

        seq_len = len(raw_tokens)
        if seq_len < 3 or logits.shape[0] < 2:
            return

        # Wir loopen durch den Satz (ab dem 2. Token, um ein 2-Token-Präfix zu haben)
        # logits[i] sagt das Token an Position raw_tokens[i+1] voraus!
        for i in range(1, seq_len - 1):
            # Wahrscheinlichkeiten für die Vorhersage an Position i berechnen
            probabilities = torch.softmax(logits[i], dim=-1)
            top_probs, top_ids = torch.topk(probabilities, k=1)
            
            top_confidence = float(top_probs[0].item())
            predicted_token_id = int(top_ids[0].item())
            actual_next_token_id = int(raw_tokens[i + 1])

            # OPTIONALER SAFETY-CHECK: Nur lernen, wenn das Modell das REALE nächste Token
            # mit hoher Sicherheit vorausgesagt hat (verhindert das Lernen von falschem Halluzinations-Wissen)
            if top_confidence >= self.convergence_threshold and predicted_token_id == actual_next_token_id:
                
                # Wir verwenden die aktuellen zwei Token als eindeutiges Kontext-Präfix
                prefix_1 = int(raw_tokens[i - 1])
                prefix_2 = int(raw_tokens[i])

                # Inkrementelles Einfügen in die verschachtelte Trie-Baum-Struktur
                if prefix_1 not in self.horizontal_chains:
                    self.horizontal_chains[prefix_1] = {}
                if prefix_2 not in self.horizontal_chains[prefix_1]:
                    self.horizontal_chains[prefix_1][prefix_2] = {
                        "next_tokens": [],
                        "confidence": top_confidence
                    }

                # Kette verlängern, falls das Token noch nicht registriert ist
                chain_node = self.horizontal_chains[prefix_1][prefix_2]
                if predicted_token_id not in chain_node["next_tokens"]:
                    chain_node["next_tokens"].append(predicted_token_id)
                    
                # Aktualisiere das Vertrauen mit dem gleitenden Höchstwert
                chain_node["confidence"] = max(chain_node["confidence"], top_confidence)


    def save_shortcuts(self):
        """
        💾 DIE VERBESSERTE SPEICHER-ENGINE:
        Sichert den aktuellen 2D-Wissensstand absolut atomar und verunreinigungsfrei 
        im binären PyTorch-Format auf der Festplatte.
        """
        try:
            # Daten-Kapselung vorbereiten
            payload = {
                "vertical": self.shortcut_map,
                "horizontal": self.horizontal_chains
            }
            
            # Atomares Schreiben über eine temporäre Datei (Schutz vor Schreibabbrüchen)
            tmp_file = self.storage_file + ".tmp"
            torch.save(payload, tmp_file)
            
            if os.path.exists(tmp_file):
                if os.path.exists(self.storage_file):
                    os.remove(self.storage_file)
                os.rename(tmp_file, self.storage_file)
                
        except Exception as e:
            print(f"[Engine] ❌ Kritischer Fehler beim Auto-Checkpointing: {e}")

    def load_shortcuts(self):
        """Lädt die 5-KB-Wissensbasis beim Systemstart inkrementell in den RAM."""
        if os.path.exists(self.storage_file):
            try:
                # Gewichte und Strukturen unblockierbar im CPU-RAM laden
                payload = torch.load(self.storage_file, map_location="cpu", weights_only=False)
                
                if isinstance(payload, dict):
                    # Altes 1D-Format abfangen und in die neue 2D-Struktur überführen
                    if "vertical" in payload and "horizontal" in payload:
                        self.shortcut_map = payload["vertical"]
                        self.horizontal_chains = payload["horizontal"]
                    else:
                        # Rückfallebene für v1.0 Altlasten
                        self.shortcut_map = payload
                        self.horizontal_chains = {}
                        
                v_count = len(self.shortcut_map)
                h_count = len(self.horizontal_chains)
                print(f"[Engine] Zweidimensionale Wissensbasis geladen ({v_count} vertikal, {h_count} horizontal).")
            except Exception as e:
                print(f"[Engine] ⚠️ Warnung beim Laden des Speichers (Datei beschädigt?): {e}")
                self.shortcut_map = {}
                self.horizontal_chains = {}
        else:
            # Jungfräulicher Start bei Erstausführung
            self.shortcut_map = {}
            self.horizontal_chains = {}
