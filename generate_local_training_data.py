# generate_local_training_data.py


import os
import re

def clean_text(text):
    """Bereinigt E-Mail-Header und Metadaten für sauberen Text."""
    # Entfernt Zeilen wie "From:", "Subject:", "Lines:", "Organization:"
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if not re.match(r'^(From|Subject|Lines|Organization|Expires|Reply-To|Distribution|Keywords|Summary|X-Newsreader|In-reply-to|Lines|NNTP-Posting-Host):', line, re.IGNORECASE):
            cleaned_lines.append(line.strip())
    
    text_block = '\n'.join(cleaned_lines)
    # Entfernt E-Mail-Adressen und spitze Klammern von Zitaten
    text_block = re.sub(r'[\w\.-]+@[\w\.-]+', '', text_block)
    text_block = re.sub(r'^>', '', text_block, flags=re.MULTILINE)
    
    # In saubere Sätze zerlegen
    final_lines = [l.strip() for l in text_block.split('\n') if len(l.strip()) > 30]
    return '\n'.join(final_lines)

def generate_offline_corpus(count=150, output_dir="training_corpus"):
    """Nutzt das lokale Scikit-Learn Paket, um offline Trainingsdaten zu erzeugen."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"[Data Feeder] 📦 Initialisiere Offline-Datensatz (20 Newsgroups)...")
    
    try:
        # Import erst hier, um Abstürze bei fehlender Installation zu vermeiden
        from sklearn.datasets import fetch_20newsgroups
        
        # Laden der Daten (Scikit-Learn hat ein eingebautes Backup-System, falls offline)
        newsgroups = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        documents = newsgroups.data
        
        print(f"[Data Feeder] 🚀 Generiere {count} lokale Textdateien aus dem Offline-Backbone...")
        
        saved_count = 0
        for i, doc in enumerate(documents):
            if saved_count >= count:
                break
                
            cleaned = clean_text(doc)
            # Nur Dokumente mit ausreichend Substanz sichern
            if len(cleaned.split()) > 40:
                filename = f"local_news_{1000 + saved_count}.txt"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(cleaned)
                    
                saved_count += 1
                print(f" -> [{saved_count}/{count}] Erfolgreich offline generiert: {filename}")
                
        print(f"\n🎉 [READY] {saved_count} Textdateien erfolgreich absolut offline im Ordner '{output_dir}/' hinterlegt.")
        print(f"💡 Du kannst jetzt 'python train_router_backbone.py both gpt2 1000' starten!")
        
    except ImportError:
        print("\n❌ Fehler: Das Paket 'scikit-learn' ist nicht installiert.")
        print("💡 Bitte tippe kurz im Terminal: pip install scikit-learn")
    except Exception as e:
        print(f"❌ Unerwarteter Fehler bei der Offline-Generierung: {e}")

if __name__ == "__main__":
    # Generiert 150 saubere Textdateien vollautomatisch auf deiner Festplatte
    generate_offline_corpus(count=150)
