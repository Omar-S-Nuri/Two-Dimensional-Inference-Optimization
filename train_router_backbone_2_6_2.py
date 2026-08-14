# train_router_backbone_2_6.py (Version 2.6.2 - Modell-differenzierter History-Tracker)
import os
import sys
import time
import torch
import asyncio
import re

from model_base_llama import AdaptiveLlamaBase
from model_base import AdaptiveTransformerBase
from analyzer import LogitLensFlowAnalyzer
from shortcut_engine import ShortcutSynthesizer
from compiler_llama import LlamaOperatorFusionCompiler
from compiler import OperatorFusionCompiler
from shadow_validator import ShadowValidator
from inference_onthefly_v2 import EvolutionaryInferenceEngine

# Intervall für die Zwischenspeicherung (Alle X Sätze auf Festplatte brennen)
SAVE_EVERY_X_STEPS = 50

class LlamaRuntimeBridge:
    def __init__(self, base_model):
        self.model = base_model
        self.tokenizer = base_model.tokenizer
        self.device = base_model.device
    def run_forward_pass(self, text): return self.model.run_forward_pass(text)
    def get_captured_data(self): return self.model.get_captured_data()
    def remove_all_hooks(self): self.model.remove_all_hooks()

def load_new_sentences_from_corpus(model_id, corpus_dir="training_corpus", max_sentences=5000):
    """
    Liest NUR Textdateien ein, die noch nicht in der modellspezifischen Chronik stehen.
    CRITICAL REPAIR: Differenziert den Lernstand strikt nach Modell-ID.
    """
    # Dynamischer Dateiname basierend auf der Modell-ID
    history_file = f"processed_files_{model_id}.txt"
    
    processed_files = set()
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as hf:
            processed_files = set([line.strip() for line in hf if line.strip()])

    sentences = []
    newly_processed = []

    if not os.path.exists(corpus_dir):
        print(f"❌ Ordner '{corpus_dir}' nicht gefunden! Bitte zuerst 'generate_local_training_data.py' ausführen.")
        return [], [], history_file
        
    all_files = sorted(os.listdir(corpus_dir))
    skipped_count = 0

    for filename in all_files:
        if filename.endswith(".txt"):
            # JUMP-FILTER: Vergleicht nur mit der Chronik des AKTUELLEN Modells
            if filename in processed_files:
                skipped_count += 1
                continue
                
            filepath = os.path.join(corpus_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
                raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 15]
                
                for s in raw_sentences:
                    if len(s) > 800:
                        short_s = s[:800]
                    else:
                        short_s = s
                    sentences.append(short_s)
            
            newly_processed.append(filename)
            
            if len(sentences) >= max_sentences:
                break

    if skipped_count > 0:
        print(f"⏩ [History] {skipped_count} Textdateien für {model_id.upper()} übersprungen (bereits gelernt).")
        
    return sentences[:max_sentences], newly_processed, history_file

def run_backbone_training(mode="both", model_id="llama3", max_sentences=3000):
    print(f"\n==================================================================")
    print(f"🔥 STARTING AIR HISTORY-TRACKING ENGINE (Version 2.6.2)")
    print(f"🤖 Modell: {model_id.upper()} | 🎯 Modus: {mode.upper()} | 📈 Sätze: {max_sentences}")
    print(f"==================================================================")
    
    from config import GLOBAL_THRESHOLD, EARLY_ABORT_THRESHOLD, STORAGE_PATH_GPT2, STORAGE_PATH_LLAMA
    
    if model_id == "llama3":
        abs_storage_path = STORAGE_PATH_LLAMA
        print(f"[Pipeline] Fahre Llama-Infrastruktur hoch...")
        base_model = AdaptiveLlamaBase(model_name="unsloth/Llama-3.2-1B", local_dir="./local_llama_free")
        analyzer = LogitLensFlowAnalyzer(base_model)
        engine = ShortcutSynthesizer(base_model, analyzer, convergence_threshold=GLOBAL_THRESHOLD, storage_file=abs_storage_path)
        compiler = LlamaOperatorFusionCompiler(base_model, engine)
        validator = ShadowValidator(base_model, engine, compiler, stability_window=1)
        bridge = LlamaRuntimeBridge(base_model)
        runtime = EvolutionaryInferenceEngine(bridge, engine, compiler, validator)
    else: # gpt2
        abs_storage_path = STORAGE_PATH_GPT2
        print(f"[Pipeline] Fahre GPT-2-Infrastruktur hoch...")
        base_model = AdaptiveTransformerBase("gpt2")
        analyzer = LogitLensFlowAnalyzer(base_model)
        engine = ShortcutSynthesizer(base_model, analyzer, convergence_threshold=GLOBAL_THRESHOLD, storage_file=abs_storage_path)
        compiler = OperatorFusionCompiler(base_model, engine)
        validator = ShadowValidator(base_model, engine, compiler, stability_window=1)
        runtime = EvolutionaryInferenceEngine(base_model, engine, compiler, validator)

    initial_vertical = len(engine.shortcut_map)
    initial_horizontal = len(engine.horizontal_chains)

    # Übergabe der model_id für die strikte Daten-Darm-Trennung
    sentences, newly_processed_files, active_history_file = load_new_sentences_from_corpus(
        model_id=model_id, max_sentences=max_sentences
    )
    
    if not sentences:
        print(f"🎉 [INFO] Keine neuen Dateien für {model_id.upper()} gefunden. Alles auf dem neuesten Stand!")
        return
    print(f"[Pipeline] {len(sentences)} neue Sätze erfolgreich geladen.")

    print("\n[Training] Starte optimierten In-Flight-Lernzyklus...")
    start_time = time.time()
    loop = asyncio.new_event_loop()
    
    for i, sentence in enumerate(sentences):
        runtime.background_data_buffer.append(sentence)
        
        if mode == "v1":
            current_batch = list(runtime.background_data_buffer)
            runtime.background_data_buffer.clear()
            for data in current_batch:
                runtime.base.run_forward_pass(data)
                runtime.compiler.engine.analyzer.update_flow_statistics(runtime.base.get_captured_data())
                links = runtime.engine.analyze_layer_equivalence(runtime.base.get_captured_data())
                runtime.engine.register_potential_shortcuts(links)
            runtime.compiler.compile_all_eligible_chains()
        elif mode == "v2":
            current_batch = list(runtime.background_data_buffer)
            runtime.background_data_buffer.clear()
            for data in current_batch:
                logits = runtime.base.run_forward_pass(data)
                rohe_tokens = runtime.base.tokenizer.encode(data, add_special_tokens=False)
                runtime.engine.analyze_and_register_horizontal_chain(rohe_tokens, logits)
        else: # both
            loop.run_until_complete(runtime.run_asynchronous_evolution_cycle())
            # Horizontale Ketten-Analyse: war im both-Modus komplett fehlend
            logits = runtime.base.run_forward_pass(sentence)
            rohe_tokens = runtime.base.tokenizer.encode(sentence, add_special_tokens=False)
            runtime.engine.analyze_and_register_horizontal_chain(rohe_tokens, logits)

        # Shadow-Validierung (Alle 5 Sätze)
        if mode in ["v1", "both"] and i % 5 == 0:
            sample_prompt = "The capital of Germany is" if model_id == "llama3" else "The documentation for this library is"
            logits = runtime.base.run_forward_pass(sample_prompt)
            
            target_map = engine.shortcut_map
            if isinstance(target_map, dict) and "vertical" in target_map:
                target_map = target_map["vertical"]
                
            valid_layers = [k for k in target_map.keys() if isinstance(k, int)]
            for start_layer in list(valid_layers):
                route_info = target_map[start_layer]
                
                if isinstance(route_info, (tuple, list)):
                    end_layer = int(route_info[0])
                else:
                    end_layer = int(route_info)
                    
                validator.evaluate_shortcut_step(start_layer, end_layer, logits)

        # Checkpointing während des Laufs
        if (i + 1) % SAVE_EVERY_X_STEPS == 0:
            engine.save_shortcuts()
            elapsed = time.time() - start_time
            print(f"💾 [Checkpoint] Fortschritt gesichert bei Satz {i+1}/{len(sentences)}.")

        elif (i + 1) % 50 == 0 or (i + 1) == len(sentences):
            elapsed = time.time() - start_time
            print(f" -> Progress: [{i+1}/{len(sentences)}] Sätze verarbeitet...")

    # Finaler Speicherstand für Inferenz-Wissen
    engine.save_shortcuts()
    
    # 💾 CHRONIK AKTUALISIEREN: Schreibt nur in die modellspezifische Log-Datei
    with open(active_history_file, "a", encoding="utf-8") as hf:
        for f_name in newly_processed_files:
            hf.write(f"{f_name}\n")
            
    total_duration = time.time() - start_time
    final_vertical = len(engine.shortcut_map)
    final_horizontal = len(engine.horizontal_chains)
    
    print(f"\n==================================================================")
    print(f"🎉 AIR BACKBONE TRAINING INKREMENTELL BEENDET!")
    print(f"⏱️ Gesamtdauer: {total_duration:.1f} Sekunden")
    print(f"📂 {len(newly_processed_files)} Textdateien in '{active_history_file}' als gelernt markiert.")
    print(f"📈 Netto-Zuwachs in diesem Lauf:")
    print(f"    ➕ Vertikale Brücken:   +{final_vertical - initial_vertical} neu (Gesamt: {final_vertical})")
    print(f"    ➕ Horizontale Ketten:  +{final_horizontal - initial_horizontal} neu (Gesamt: {final_horizontal})")
    print(f"==================================================================")

if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "both"
    model = sys.argv[2] if len(sys.argv) > 2 else "llama3"
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    
    run_backbone_training(mode=m, model_id=model, max_sentences=count)
