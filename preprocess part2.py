import re

def clean_text(text):
    """Clean text by removing unwanted characters and formatting."""
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[^A-Za-z0-9.,;!?'\s]", "", text)
    return text.strip()

with open("war_and_peace.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
processed_text = clean_text(raw_text)
 
with open("cleaned_war_and_peace.txt", "w", encoding="utf-8") as f:
    f.write(processed_text)
    
print("Preprocessing Complete: cleaned_war_and_peace.txt saved!")


