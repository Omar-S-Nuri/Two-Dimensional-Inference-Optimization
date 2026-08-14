# run_accuracy_check.py (Version 2.7 - Reiner 2D-Wissens-Prüfstand)
import os
import torch
from transformers import AutoTokenizer

from model_base import AdaptiveTransformerBase
from model_base_llama import AdaptiveLlamaBase
from analyzer import LogitLensFlowAnalyzer
from shortcut_engine import ShortcutSynthesizer
from compiler import OperatorFusionCompiler
from compiler_llama import LlamaOperatorFusionCompiler
from inference_onthefly_v2 import EvolutionaryInferenceEngine

# Globale Pfade aus deiner zentralen config.py laden
from config import GLOBAL_THRESHOLD, STORAGE_PATH_GPT2, STORAGE_PATH_LLAMA

class UniversalRuntimeBridge:
    def __init__(self, base_model):
        self.model = base_model
        self.tokenizer = base_model.tokenizer
        self.device = base_model.device
    def run_forward_pass(self, text): return self.model.run_forward_pass(text)
    def get_captured_data(self): return self.model.get_captured_data()
    def remove_all_hooks(self): self.model.remove_all_hooks()

def run_pure_knowledge_check(model_id="gpt2"):
    print(f"\n==================================================================")
    print(f"🎯 AIR PURE 2D-KNOWLEDGE PRECISION CHECK (v2.7)")
    print(f"🤖 Modell: {model_id.upper()} | 📂 Filter: NUR ECHTE FAST-TRACKS")
    print(f"==================================================================\n")

    if model_id == "llama3":
        filepath = STORAGE_PATH_LLAMA
        if not os.path.exists(filepath):
            print("❌ Keine Llama-Speicherdatei gefunden.")
            return
        base_net = AdaptiveLlamaBase(model_name="unsloth/Llama-3.2-1B", local_dir="./local_llama_free")
    else:
        filepath = STORAGE_PATH_GPT2
        if not os.path.exists(filepath):
            print("❌ Keine GPT-2-Speicherdatei gefunden.")
            return
        base_net = AdaptiveTransformerBase("gpt2")

    analyzer = LogitLensFlowAnalyzer(base_net)
    engine = ShortcutSynthesizer(base_net, analyzer, convergence_threshold=GLOBAL_THRESHOLD, storage_file=filepath)
    
    if model_id == "llama3":
        compiler = LlamaOperatorFusionCompiler(base_net, engine)
    else:
        compiler = OperatorFusionCompiler(base_net, engine)
    compiler.compile_all_eligible_chains()
    
    bridge = UniversalRuntimeBridge(base_net)
    runtime = EvolutionaryInferenceEngine(bridge, engine, compiler, None)

    # Die 50 standardisierten Benchmark-Sätze
    test_corpus = [
        "The capital of Germany is", "The capital of France is", "The capital of Italy is",
        "The capital of Spain is", "The capital of Japan is", "The capital of China is",
        "The capital of Canada is", "The capital of Brazil is", "The capital of Russia is",
        "The capital of India is", "The documentation for this library is", "The source code for this project is",
        "The main feature of this framework is", "The license file for this repository is", "The default configuration for this node is",
        "The installation guide for this package is", "The performance benchmark for this model is", "The execution latency for this process is",
        "The activation stream for this layer is", "The residual stream for this transformer is", "Once upon a time there was a",
        "In a faraway galaxy there was a", "Long ago in an ancient kingdom there was a", "Deep in the dark forest there was a",
        "High up in the frozen mountains there was a", "Inside the hidden laboratory there was a", "Under the deep blue ocean there was a",
        "Beneath the surface of the planet there was a", "At the very edge of the universe there was a", "Far beyond the solar system there was a",
        "The matrix multiplication operator is", "The cosine similarity coefficient is", "The SwiGLU activation block is",
        "The multi-head attention mechanism is", "The root mean square normalization is", "The dynamic adaptive layer routing is",
        "The post-training operator fusion is", "The latent alignment verification is", "The entropy based abort filter is",
        "The asynchronous continuous learning loop is", "Artificial intelligence and neural networks are", "Large language models and transformers are",
        "Deep learning and backpropagation are", "Gradient descent and optimization parameters are", "Supervised learning and training datasets are",
        "Reinforcement learning and reward functions are", "Unsupervised pretraining and tokens are", "Linear projections and bias vectors are",
        "Floating point precision and tensors are", "Hardware latency and memory bandwidth are"
    ]

    # Caches sichern
    saved_map = engine.shortcut_map
    saved_chains = engine.horizontal_chains

    evaluated_prompts = 0
    matched_tokens = 0

    print(f"[Check] Scanne 50 Phrasen nach aktiven Treffern in deiner Wissensbasis...\n")

    for sentence in test_corpus:
        # Schritt A: Im Fast-Track prüfen, welcher Pfad gewählt wird
        engine.shortcut_map = saved_map
        engine.horizontal_chains = saved_chains
        fast_word, track = runtime.predict_next_token(sentence)
        
        # 🛡️ DER FILTRATION-SHIELD: Wenn es ein Fallback-Slow-Track war, ignorieren wir den Satz!
        if "SLOW-TRACK" in track:
            continue
            
        # Schritt B: Wenn es ein ECHTER Fast-Track war, messen wir die Slow-Referenz dagegen
        evaluated_prompts += 1
        engine.shortcut_map = {}
        engine.horizontal_chains = {}
        slow_word, _ = runtime.predict_next_token(sentence)
        
        # Vergleich ziehen
        is_match = slow_word.strip().lower() == fast_word.strip().lower()
        if is_match:
            matched_tokens += 1
            
        status_symbol = "🟢 [MATCH]" if is_match else "❌ [MISMATCH]"
        print(f" 🔥 TREFFER ({evaluated_prompts:02d}) ──► Route: [{track}]")
        print(f"    {status_symbol} Slow-Modell sagt: '{slow_word.strip()}' | Fast-Autobahn sagt: '{fast_word.strip()}'\n")

    # Endergebnis auswerfen
    print("==================================================================")
    print("📊 AIR REINER ACCURACY-REPORT FÜR DEINE WISSENSBASIS")
    print("==================================================================")
    print(f"📦 Geladener Wissensstand:        {len(saved_map)} vertikal | {len(saved_chains)} horizontal")
    print(f"⚡ Aktivierte Fast-Track-Demos:   {evaluated_prompts} von 50 Sätzen haben gezündet")
    
    if evaluated_prompts > 0:
        fidelity = (matched_tokens / evaluated_prompts) * 100
        print(f"🟢 Davon identische Wort-Treffer:  {matched_tokens} von {evaluated_prompts}")
        print(f"📈 NET KOGNITIVE PRÄZISION:       {fidelity:.2f}% echte Token-Fidelity")
    else:
        print("      (Deine .pt-Dateien enthalten noch keine Muster für diese 50 Sätze.)")
        print("      💡 Tipp: Starte 'python train_router_backbone.py both llama3 50'!")
    print("==================================================================\n")

if __name__ == "__main__":
    # Kann frei zwischen 'gpt2' und 'llama3' geswitcht werden
    run_pure_knowledge_check("gpt2")
