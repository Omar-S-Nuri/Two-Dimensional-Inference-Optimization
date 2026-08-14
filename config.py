# config.py - ZENTRALE PARAMETER-STEUERUNG FÜR AIR v2.0

import os


# 1. DIE GLOBALE INFERENZ- SCHLEUSE (Elite-Hürde)
# Einsatzort: shortcut_engine.py & inference_onthefly_v2.py
# Bedeutung: Der absolute mathematische Sweet-Spot für 1B-Modelle. 
# Bestimmt, ab welcher Vektor-Ähnlichkeit Schichten fuziert werden (v1)
# und ab welcher Wahrscheinlichkeit Wortketten im Trie scharfschalten (v2).
GLOBAL_THRESHOLD = 0.92

# 2. DER SYSTEMISCHE TÜRESTEHER (Early-Abort)
# Einsatzort: shortcut_engine.py
# Bedeutung: Blockiert semantisches Chaos im Vorfeld. Liegt die Ähnlichkeit 
# zwischen L1 und der mittleren Schicht unter 35%, wird die vertikale Suche 
# sofort abgebrochen. Spart der CPU massenhaft unnötige Rechenschleifen.
EARLY_ABORT_THRESHOLD = 0.45

# 3. DAS STATISTISCHE STABILITÄTS-FENSTER (Shadow-Validation)
# Einsatzort: app_onthefly.py, train_router_backbone.py, run_benchmarks.py
# Bedeutung: Definiert, wie oft eine Brücke im Shadow-Modus über LAV fehlerfrei 
# matchen muss, bevor sie von PENDING auf VALIDATED umspringt.
STABILITY_WINDOW_TRAINING = 1  # Express-Modus für das schnelle Offline-Vortraining
STABILITY_WINDOW_PRODUCTION = 2 # Sicherheits-Modus für den echten Live-Chat-Server




# ==================================================================
# 📂 CENTRALIZED STORAGE PATHS (NEU: Absolut dynamisch via OS-Kapselung)
# ==================================================================
# Ermittelt automatisch das aktuelle Verzeichnis, in dem diese config.py liegt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Krisensichere Pfade für deine .pt-Wissensdatenbanken
STORAGE_PATH_GPT2 = os.path.join(BASE_DIR, "shortcuts_gpt2.pt")
STORAGE_PATH_LLAMA = os.path.join(BASE_DIR, "shortcuts_llama.pt")