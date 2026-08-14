# run_accuracy_benchmark.py
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

# Globale Pfade und Parameter aus deiner zentralen config.py laden
from config import GLOBAL_THRESHOLD, STORAGE_PATH_GPT2, STORAGE_PATH_LLAMA

class UniversalRuntimeBridge:
    def __init__(self, base_model):
        self.model = base_model
        self.tokenizer = base_model.tokenizer
        self.device = base_model.device
    def run_forward_pass(self, text): return self.model.run_forward_pass(text)
    def get_captured_data(self): return self.model.get_captured_data()
    def remove_all_hooks(self): self.model.remove_all_hooks()

def run_precision_evaluation(model_id="gpt2"):
    print(f"\n==================================================================")
    print(f"🎯 AIR ACCURACY & TOKEN IDENTITY BENCHMARK (v2.0)")
    print(f"🤖 Modell: {model_id.upper()} | ⚙️ Target Threshold: {GLOBAL_THRESHOLD*100:.1f}%")
    print(f"==================================================================\n")

    # 1. Infrastruktur basierend auf dem Modell laden
    if model_id == "llama3":
        filepath = STORAGE_PATH_LLAMA
        if not os.path.exists(filepath):
            print("❌ Keine Llama-Speicherdatei gefunden. Bitte zuerst trainieren!")
            return
        base_net = AdaptiveLlamaBase(model_name="unsloth/Llama-3.2-1B", local_dir="./local_llama_free")
    else:
        filepath = STORAGE_PATH_GPT2
        if not os.path.exists(filepath):
            print("❌ Keine GPT-2-Speicherdatei gefunden. Bitte zuerst trainieren!")
            return
        base_net = AdaptiveTransformerBase("gpt2")

    # SAUBERE REIHENFOLGE: 1. Analyzer -> 2. Engine hochfahren
    analyzer = LogitLensFlowAnalyzer(base_net)
    engine = ShortcutSynthesizer(base_net, analyzer, convergence_threshold=GLOBAL_THRESHOLD, storage_file=filepath)
    
    # 3. Compiler erst JETZT mit der fertigen Engine füttern
    if model_id == "llama3":
        compiler = LlamaOperatorFusionCompiler(base_net, engine)
    else:
        compiler = OperatorFusionCompiler(base_net, engine)
    
    # Jetzt kann der Compiler fehlerfrei die Ketten im RAM fuzieren
    compiler.compile_all_eligible_chains()
    
    bridge = UniversalRuntimeBridge(base_net)
    runtime = EvolutionaryInferenceEngine(bridge, engine, compiler, None)

    # Das standardisierte 50-Sätze-Testkorpus
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

    total_prompts = len(test_corpus)
    matched_tokens = 0
    activated_shortcuts = 0

    print(f"[Harness] Vergleiche Token-Identität für {total_prompts} Testphrasen...\n")

    for idx, sentence in enumerate(test_corpus):
        # 1. SLOW-TRACK: Referenz-Wort ermitteln (Wir leeren die Shortcuts im RAM)
        backup_map = engine.shortcut_map
        backup_chains = engine.horizontal_chains
        engine.shortcut_map = {}
        engine.horizontal_chains = {}
        
        slow_word, _ = runtime.predict_next_token(sentence)
        
        # 2. FAST-TRACK: Optimiertes Wort ermitteln (Shortcuts wieder aktivieren)
        engine.shortcut_map = backup_map
        engine.horizontal_chains = backup_chains
        
        fast_word, track = runtime.predict_next_token(sentence)
        
        shortcut_triggered = "FAST-TRACK" in track
        if shortcut_triggered:
            activated_shortcuts += 1

        # Vergleichen, ob beide Wörter identisch sind
        is_match = slow_word.strip().lower() == fast_word.strip().lower()
        if is_match:
            matched_tokens += 1
            
        status_symbol = "🟢" if is_match else "❌"
        track_symbol = "⚡ [2D-Autobahn]" if shortcut_triggered else "🐌 [Standard-Pfad]"
        
        print(f" [{idx+1:02d}] Pfad: {track_symbol} | {status_symbol} Slow: '{slow_word.strip()}' vs. Fast: '{fast_word.strip()}'")

    # Ergebnis-Bericht ausgeben
    accuracy_percentage = (matched_tokens / total_prompts) * 100
    trigger_rate = (activated_shortcuts / total_prompts) * 100

    print(f"\n==================================================================")
    print(f"📊 AIR ACCURACY VERIFICATION REPORT")
    print(f"==================================================================")
    print(f"🎯 Total Test Prompts:            {total_prompts}")
    print(f"⚡ Activated Fast-Tracks:         {activated_shortcuts} von {total_prompts} ({trigger_rate:.1f}%)")
    print(f"🟢 Identical Token Matches:        {matched_tokens} von {total_prompts}")
    print(f"📈 GLOBAL MATHEMATICAL FIDELITY:   {accuracy_percentage:.2f}% Token Identity")
    print(f"==================================================================\n")

if __name__ == "__main__":
    run_precision_evaluation("gpt2")
