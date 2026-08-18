# generate_massive_demo_corpus.py

import os
import re

def clean_text(text):
    """Bereinigt E-Mail-Reste und kryptische Symbole für erstklassiges Sprachfutter."""
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if not re.match(r'^(From|Subject|Lines|Organization|Expires|Reply-To|Distribution|Keywords|Summary|X-Newsreader|In-reply-to|Lines|NNTP-Posting-Host):', line, re.IGNORECASE):
            cleaned_lines.append(line.strip())
    
    text_block = '\n'.join(cleaned_lines)
    text_block = re.sub(r'[\w\.-]+@[\w\.-]+', '', text_block)
    text_block = re.sub(r'^>', '', text_block, flags=re.MULTILINE)
    
    final_lines = [l.strip() for l in text_block.split('\n') if len(l.strip()) > 40]
    return '\n'.join(final_lines)

def generate_giant_corpus(target_count=2000, output_dir="training_corpus"):
    """Kombiniert verschiedene Kategorien zu einem massiven Demo-Datensatz (100% Offline)."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"[Demo Feeder] 📦 Initialisiere massives Multi-Themen-Paket...")
    
    try:
        from sklearn.datasets import fetch_20newsgroups
        
        # Wir laden gezielt Kategorien, die extrem gutes, strukturiertes Englisch enthalten
        categories = [
            'comp.sys.mac.hardware', 'comp.os.ms-windows.misc',  # IT & Hardware-Strukturen
            'sci.space', 'sci.electronics',                     # Wissenschaft & Logik
            'rec.sport.baseball', 'rec.motorcycles',             # Umgangssprache & Fakten
            'talk.politics.misc'                                # Komplexe Satzstrukturen
        ]
        
        newsgroups = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
        documents = newsgroups.data
        
        print(f"[Demo Feeder] 🚀 Generiere {target_count} abwechslungsreiche Textdateien auf der Festplatte...")
        
        saved_count = 0
        for i, doc in enumerate(documents):
            if saved_count >= target_count:
                break
                
            cleaned = clean_text(doc)
            # Nur Dokumente mit echter Substanz sichern
            if len(cleaned.split()) > 50:
                filename = f"demo_text_{5000 + saved_count}.txt"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(cleaned)
                    
                saved_count += 1
                if saved_count % 100 == 0:
                    print(f" -> Progress: [{saved_count}/{target_count}] Dateien offline geschrieben...")
                
        print(f"\n🎉 [READY] {saved_count} High-Quality-Dateien erfolgreich in '{output_dir}/' hinterlegt!")
        print(f"💡 Nächster Schritt: Starte 'python train_router_backbone.py both gpt2 2000' für die Gold-Matrix.")
        
    except ImportError:
        print("\n❌ Fehler: Scikit-Learn ist nicht installiert. Bitte tippe: pip install scikit-learn")
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")

if __name__ == "__main__":
    generate_giant_corpus(target_count=5000)
