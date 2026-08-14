# app_onthefly.py - TEIL 1 von 3 (Version 2.0 - Zusammenfügen mit Teil 2 und 3)
from flask import Flask, request, jsonify, render_template_string
import threading
import asyncio
import time
import torch

from model_base import AdaptiveTransformerBase
from model_base_llama import AdaptiveLlamaBase
from compiler import OperatorFusionCompiler
from compiler_llama import LlamaOperatorFusionCompiler
from analyzer import LogitLensFlowAnalyzer
from shortcut_engine import ShortcutSynthesizer
from shadow_validator import ShadowValidator
# WICHTIG: Nutzt deine neue, zweidimensionale Inferenz-Engine
from inference_onthefly_v2 import EvolutionaryInferenceEngine

##from config  import GLOBAL_THRESHOLD, EARLY_ABORT_THRESHOLD, STABILITY_WINDOW_TRAINING, STABILITY_WINDOW_PRODUCTION
from config import GLOBAL_THRESHOLD, STABILITY_WINDOW_PRODUCTION, STORAGE_PATH_GPT2, STORAGE_PATH_LLAMA

app = Flask(__name__)

# Globale Datenstruktur für das dynamische On-Demand-Laden (Lazy Loading)
models = {
    "gpt2": {"base": None, "analyzer": None, "engine": None, "compiler": None, "validator": None, "runtime": None},
    "llama3": {"base": None, "analyzer": None, "engine": None, "compiler": None, "validator": None, "runtime": None}
}

# CRITICAL SECURITY REPAIR: Ein threadübergreifendes State-Management 
# verhindert Race Conditions im Flask-Webserver vollständig.
SYSTEM_STATE = {
    "active_model": "gpt2"
}
state_lock = threading.Lock()

system_status = {
    "logs": []
}

def log_message(msg):
    """Gibt Logs in der Konsole aus und puffert sie für das Web-Interface."""
    print(msg)
    system_status["logs"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    if len(system_status["logs"]) > 50:
        system_status["logs"].pop(0)

def ensure_model_is_loaded(model_id):
    """Lädt das ausgewählte Modell erst in den Arbeitsspeicher, wenn es gebraucht wird."""
    if models[model_id]["base"] is not None:
        return
    
    log_message(f"[System] Initialisiere Runtime für {model_id.upper()}...")
    if model_id == "gpt2":
        # GPT-2 Pipeline konfigurieren
        ##abs_storage_path_gpt2 = r"C:\Users\onuri\Desktop\nn-md\adaptive_onthefly_v02\shortcuts_gpt2.pt"
        abs_storage_path_gpt2 = STORAGE_PATH_GPT2
        base = AdaptiveTransformerBase("gpt2")
        analyzer = LogitLensFlowAnalyzer(base)
        ##engine = ShortcutSynthesizer(base, analyzer, convergence_threshold=0.85, storage_file=abs_storage_path_gpt2)
        engine = ShortcutSynthesizer(base, analyzer, convergence_threshold=GLOBAL_THRESHOLD, storage_file=abs_storage_path_gpt2)
        compiler = OperatorFusionCompiler(base, engine)
        ## validator = ShadowValidator(base, engine, compiler, stability_window=1)
        validator = ShadowValidator(base, engine, compiler, stability_window=STABILITY_WINDOW_PRODUCTION)
        runtime = EvolutionaryInferenceEngine(base, engine, compiler, validator)
    else: # llama3
        # Llama 3.2 Pipeline über dein lokales unsloth-Verzeichnis konfigurieren
        base_llama = AdaptiveLlamaBase(model_name="unsloth/Llama-3.2-1B", local_dir="./local_llama_free")
        analyzer = LogitLensFlowAnalyzer(base_llama)
        
        # CRITICAL REPAIR: Absoluter Pfad garantiert, dass die Datei im exakten Ordner landet!
        ##abs_storage_path_llama = r"C:\Users\onuri\Desktop\nn-md\adaptive_onthefly_v02\shortcuts_llama.pt"
        abs_storage_path_llama = STORAGE_PATH_LLAMA

        ### engine = ShortcutSynthesizer(base_llama, analyzer, convergence_threshold=0.80, storage_file=abs_storage_path_llama)
        engine = ShortcutSynthesizer(base_llama, analyzer, convergence_threshold=GLOBAL_THRESHOLD, storage_file=abs_storage_path_llama)
        compiler = LlamaOperatorFusionCompiler(base_llama, engine)
        ## validator = ShadowValidator(base_llama, engine, compiler, stability_window=1)
        validator = ShadowValidator(base_llama, engine, compiler, stability_window=STABILITY_WINDOW_PRODUCTION)
        
        class LlamaRuntimeBridge:
            def __init__(self):
                self.model = base_llama
                self.tokenizer = base_llama.tokenizer
                self.device = base_llama.device
            def run_forward_pass(self, text): return base_llama.run_forward_pass(text)
            def get_captured_data(self): return base_llama.get_captured_data()
            def remove_all_hooks(self): base_llama.remove_all_hooks()
            
        bridge = LlamaRuntimeBridge()
        runtime = EvolutionaryInferenceEngine(bridge, engine, compiler, validator)
        
    models[model_id] = {
        "base": base_llama if model_id == "llama3" else base,
        "analyzer": analyzer, "engine": engine, "compiler": compiler, "validator": validator, "runtime": runtime
    }
    log_message(f"[System] {model_id.upper()}-Infrastruktur erfolgreich hochgefahren.")

# Standardmäßig das erste Modell beim Start hochfahren
ensure_model_is_loaded("gpt2")
# app_onthefly.py - TEIL 2 von 3 (Version 2.0 - Direkt unten an Teil 1 anfügen)

# --- REAKTIVES HTML TEMPLATE (OHNE SCHLAFKNOPF) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Adaptive Multi-Model Router Interface (v2.0)</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #1e1e2e; color: #cdd6f4; margin: 40px; }
        .container { max-width: 1050px; margin: 0 auto; display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 20px; }
        .panel { background: #313244; padding: 20px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.3); }
        h2 { color: #89b4fa; margin-top: 0; }
        select, input[type="text"] { padding: 12px; background: #11111b; border: 1px solid #45475a; color: white; border-radius: 6px; }
        select { width: 100%; font-size: 16px; margin-bottom: 20px; color: #a6e3a1; font-weight: bold; cursor: pointer; }
        input[type="text"] { width: 75%; font-size: 14px; }
        button { padding: 12px 20px; background: #a6e3a1; border: none; color: #11111b; font-weight: bold; border-radius: 6px; cursor: pointer; font-size: 14px; }
        .log-box { background: #11111b; font-family: 'Fira Code', monospace; font-size: 12px; height: 380px; overflow-y: scroll; padding: 12px; border-radius: 6px; color: #a6e3a1; line-height: 1.5; }
        .status-badge { display: inline-block; padding: 6px 12px; border-radius: 20px; font-weight: bold; font-size: 14px; background: #a6e3a1; color: #11111b; }
    </style>
</head>
<body>
    <div class="container">
        <div class="panel">
            <h2>Autonomes Multi-Modell Live-Routing (v2.0)</h2>
            <label style="font-size: 14px; color: #bac2de;" for="model-select">Aktives KI-Modell umschalten:</label>
            <select id="model-select" onchange="changeModel()">
                <option value="gpt2">OpenAI GPT-2 (12 Schichten - Lokal)</option>
                <option value="llama3">Meta Llama 3.2 1B (16 Schichten - Login-Frei)</option>
            </select>
            
            <p>Evolutions-Zustand: <span class="status-badge">Autonomes 2D-Lernen im Flug AKTIV</span></p>
            <p style="font-size: 12px; color: #a6adc8;">Das System optimiert sich jetzt vertikal (Schichten) UND horizontal (Ketten) reaktiv im Flug.</p>
            <br>
            
            <input type="text" id="user-input" placeholder="Schreibe einen Satzanfang...">
            <button onclick="sendMessage()">Senden</button>
            <br><br>
            <h3>Ausgabe der Adaptive Engine:</h3>
            <p id="chat-output" style="font-size: 20px; font-weight: bold; color: #f5c2e7; background: #181825; padding: 15px; border-radius: 6px;"></p>
            <p id="track-output" style="font-size: 13px; color: #b4befe; font-family: monospace;"></p>
        </div>
        
        <div class="panel">
            <h2>Evolutions-Protokoll (Reaktiv)</h2>
            <div id="log-container" class="log-box"></div>
        </div>
    </div>

    <script>
        function changeModel() {
            let model = document.getElementById("model-select").value;
            fetch('/switch_model', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({model: model})
            }).then(res => res.json())
            .then(data => {
                document.getElementById("chat-output").innerText = "";
                document.getElementById("track-output").innerText = "";
            });
        }

        function sendMessage() {
            let input = document.getElementById("user-input").value;
            fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: input})
            })
            .then(res => res.json())
            .then(data => {
                document.getElementById("chat-output").innerText = data.completion;
                document.getElementById("track-output").innerText = "Rechenpfad: [" + data.track + "] | Latenz: " + data.time + " ms";
            });
        }

        setInterval(() => {
            fetch('/logs').then(res => res.json()).then(logs => {
                let box = document.getElementById("log-container");
                box.innerHTML = logs.join("<br>");
                box.scrollTop = box.scrollHeight;
            });
        }, 1000);
    </script>
</body>
</html>
"""
# app_onthefly.py - TEIL 3 von 3 (Version 2.0 - Direkt unten an Teil 2 anfügen)

# --- REAKTIVER HINTERGRUND-THREAD FÜR AUTOMATISCHES LERNEN ---
def continuous_evolution_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    log_message("[Hintergrund-Thread] Autonomer 2D-Evolutions-Wächter im Dauerbetrieb aktiv.")
    while True:
        # Threadsicher das exakt aktive Modell über die Dictionary-Struktur auslesen
        with state_lock:
            active_id = SYSTEM_STATE["active_model"]
            m = models.get(active_id)
        
        # Die Prüfung und Entleerung des Puffers erfolgt nun garantiert isoliert pro Modell-Instanz
        if m and m["runtime"] is not None and len(m["runtime"].background_data_buffer) > 0:
            log_message(f"[Autonomes Lernen] 🧠 Neuer Datenstrom erkannt für {active_id.upper()}. Starte In-Flight-Optimierung...")
            
            # 2D-Evolution im Hintergrund ausführen (Vertikaler Schichtenschnitt + Horizontaler Trie)
            loop.run_until_complete(m["runtime"].run_asynchronous_evolution_cycle())
            
            # Latente Alignierungs-Verifizierung (LAV) direkt im Anschluss ausführen
            sample_prompt = "The capital of Germany is" if active_id == "llama3" else "Die Hauptstadt von Frankreich ist"
            logits = m["base"].run_forward_pass(sample_prompt)
            
            # Sicheres Auslesen der vertikalen Shortcut-Routen aus der Map
            for start_layer in list(m["engine"].shortcut_map.keys()):
                route_info = m["engine"].shortcut_map[start_layer]
                if isinstance(route_info, (tuple, list)):
                    end_layer = int(route_info[0])
                else:
                    end_layer = int(route_info)
                ##end_layer = int(route_info)
                m["validator"].evaluate_shortcut_step(start_layer, end_layer, logits)
                
            # Erkenntnisse unbemerkt auf Festplatte einfrieren
            m["engine"].save_shortcuts()
            log_message(f"[Autonomes Lernen] 🚀 2D-In-Flight-Optimierung für {active_id.upper()} abgeschlossen.")
            
        time.sleep(1) # Ressourcenschonende Taktung (1 Sekunde Abfrage-Intervall)

# Startet den Background-Thread separat abseits des Webservers
threading.Thread(target=continuous_evolution_loop, daemon=True).start()

# --- SERVER ROUTEN ---

@app.route('/')
def home(): 
    return render_template_string(HTML_TEMPLATE)

@app.route('/check_readiness', methods=['GET'])
def check_readiness():
    """Gibt dem Web-Interface Bescheid, welche Modelle fertig geladen sind."""
    with state_lock:
        return jsonify({
            "gpt2": True if models["gpt2"]["base"] is not None else False,
            "llama3": True if models["llama3"]["base"] is not None else False
        })

@app.route('/switch_model', methods=['POST'])
def switch_model():
    target_model = request.json.get("model", "gpt2")
    ensure_model_is_loaded(target_model)
    
    # Threadsicheres Umschalten im permanenten Dictionary
    with state_lock:
        SYSTEM_STATE["active_model"] = target_model
        active_id = SYSTEM_STATE["active_model"]
        
    log_message(f"[Systemsteuerung] Fokus gewechselt auf: {active_id.upper()}")
    return jsonify({"current_model": active_id})

@app.route('/chat', methods=['POST'])
def chat():
    user_text = request.json.get("text", "")
    
    # Threadsicher prüfen, welches Modell JETZT die Inferenz fahren muss
    with state_lock:
        active_id = SYSTEM_STATE["active_model"]
        m = models[active_id]
    
    start_time = time.perf_counter()
    predicted_word, track = m["runtime"].predict_next_token(user_text)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    # --- SPEICHER-ERWEITERUNG FÜR DATEN-EXTRAKTION (read_storage) ---
    clean_token = predicted_word.strip()
    if 1 in m["engine"].shortcut_map:
        route_tuple = m["engine"].shortcut_map
        if len(route_tuple) == 3:
            end, status, sim = route_tuple
            m["engine"].shortcut_map = (end, status, sim, clean_token if clean_token else "Context-Signal")
        elif len(route_tuple) == 4:
            end, status, sim, _ = route_tuple
            m["engine"].shortcut_map = (end, status, sim, clean_token if clean_token else "Context-Signal")

    # CRITICAL SAVE-FORCE: Wir zwingen den Haupt-Thread, das 2D-Wissen JETZT physisch zu sichern!
    try:
        m["engine"].save_shortcuts()
        log_message(f"[System] 💾 Wissensbasis für {active_id.upper()} erfolgreich via Chat-Route auf Festplatte erzwungen.")
    except Exception as e:
        log_message(f"[System] ⚠️ Fehler beim direkten Speichern: {e}")

    log_message(f"[Inferenz] {active_id.upper()} antwortet: '{predicted_word.strip()}' via [{track}]")
    return jsonify({"completion": user_text + predicted_word, "track": track, "time": f"{elapsed_ms:.1f}"})

@app.route('/logs', methods=['GET'])
def get_logs(): 
    return jsonify(system_status["logs"])

if __name__ == '__main__':
    # Startet den integrierten Flask-Server lokal auf Port 5000
    app.run(debug=False, port=5000)
