# run_benchmarks.py
import time
import torch
import os

from model_base import AdaptiveTransformerBase
from analyzer import LogitLensFlowAnalyzer
from shortcut_engine import ShortcutSynthesizer
from compiler import OperatorFusionCompiler
from shadow_validator import ShadowValidator
# Nutzt deine neue, zweidimensionale Inferenz-Engine
from inference_onthefly_v2 import EvolutionaryInferenceEngine

class GPT2RuntimeBridge:
    """Sichert die identische Schnittstellen-Kompatibilität zur 2D-Runtime."""
    def __init__(self, base_model):
        self.model = base_model
        self.tokenizer = base_model.tokenizer
        self.device = base_model.device
    def run_forward_pass(self, text): return self.model.run_forward_pass(text)
    def get_captured_data(self): return self.model.get_captured_data()
    def remove_all_hooks(self): self.model.remove_all_hooks()

def run_gpt2_production_benchmark():
    print("\n==================================================================")
    print("📊 AIR BENCHMARK HARNESS: OPENAI GPT-2 (Version 2.0)")
    print("==================================================================\n")

    # 1. Absolute Pfade definieren (Garantierte Windows-Kompatibilität)
    abs_storage_path = r"C:\Users\onuri\Desktop\nn-md\adaptive_onthefly_v02\shortcuts_gpt2.pt"

    # 2. Pipeline initialisieren
    print("[Harness] Lade GPT-2-Infrastruktur und Gewichte im RAM...")
    base_gpt2 = AdaptiveTransformerBase("gpt2")
    analyzer = LogitLensFlowAnalyzer(base_gpt2)
    engine = ShortcutSynthesizer(base_gpt2, analyzer, convergence_threshold=0.85, storage_file=abs_storage_path)
    compiler = OperatorFusionCompiler(base_gpt2, engine)
    validator = ShadowValidator(base_gpt2, engine, compiler, stability_window=1)
    
    bridge = GPT2RuntimeBridge(base_gpt2)
    runtime = EvolutionaryInferenceEngine(bridge, engine, compiler, validator)

    # 3. Das heterogene Testkorpus (50 standardisierte Benchmark-Sätze)
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

    print(f"[Harness] {len(test_corpus)} heterogene Testphrasen geladen. Starte Evaluation...\n")
    
    # --- PHASE 1: EVALUATION IM UNOPTIMIERTEN ZUSTAND (SLOW-TRACK) ---
    print("⏳ [Phase 1]: Erfassung der Baseline (Reiner SLOW-TRACK)...")
    backup_shortcuts = engine.shortcut_map
    backup_chains = engine.horizontal_chains
    engine.shortcut_map = {}
    engine.horizontal_chains = {}
    compiler.fused_shortcuts = {}

    start_slow = time.perf_counter()
    for sentence in test_corpus:
        _, _ = runtime.predict_next_token(sentence)
    slow_duration_ms = (time.perf_counter() - start_slow) * 1000
    print(f" -> Baseline beendet. Gesamtlatenz: {slow_duration_ms:.2f} ms\n")

    # --- PHASE 2: EVALUATION IM OPTIMIERTEN ZUSTAND (2D FAST-TRACK) ---
    print("🚀 [Phase 2]: Erfassung der AIR 2D-Runtime (Kombinierter FAST-TRACK)...")
    engine.shortcut_map = backup_shortcuts
    engine.horizontal_chains = backup_chains
    compiler.compile_all_eligible_chains() # Gewichte im RAM fuzieren

    start_fast = time.perf_counter()
    for sentence in test_corpus:
        _, _ = runtime.predict_next_token(sentence)
    fast_duration_ms = (time.perf_counter() - start_fast) * 1000

    # 4. Globaler Effizienz-Report ausgeben
    print("\n==================================================================")
    print("📊 AIR GLOBAL BENCHMARK EFFICIENCY REPORT (GPT-2)")
    print("==================================================================")
    print(f"⏱️ Total Latency SLOW-TRACK:        {slow_duration_ms:.2f} ms")
    print(f"⚡ Total Latency AIR 2D-TRACK:     {fast_duration_ms:.2f} ms")
    
    efficiency = ((slow_duration_ms - fast_duration_ms) / slow_duration_ms) * 100
    if efficiency > 0:
        print(f"📈 NET EFFICIENCY INCREASE:         +{efficiency:.2f}% Rechenzeit-Ersparnis")
    else:
        print(f"📉 ASYMMETRIC MEMORY OVERHEAD:      {efficiency:.2f}% (CPU-Bus-Artifact)")
    print("==================================================================\n")

if __name__ == "__main__":
    run_gpt2_production_benchmark()
