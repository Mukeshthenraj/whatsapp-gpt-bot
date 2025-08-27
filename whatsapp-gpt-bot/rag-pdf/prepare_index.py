# rag-pdf/prepare_index.py
import json, re, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

SRC_JSON = Path("../kellen_produkte.json").resolve()
DOCS_JSON = Path("catalog_docs.json")
VECS_NPY  = Path("catalog_vectors.npy")

def norm_id(s: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", (s or "").lower())

def build_docs():
    data = json.loads(Path(SRC_JSON).read_text(encoding="utf-8"))
    docs = []
    for prod in data:
        title = (prod.get("title") or "").strip()
        category = (prod.get("category") or "").strip()
        desc = (prod.get("description") or "").strip()

        for v in prod.get("variants", []):
            bestell = (v.get("bestell_nr") or "").strip()
            ausf = (v.get("ausfuehrung") or "").strip()

            # gather a few useful numbers if present
            extra = []
            for key in ("l_mm","b_mm","h_mm","mm","ø_mm","d_mm"):
                if key in v: extra.append(f"{key.replace('_',' ').upper()}: {v[key]}")
            if "ve" in v: extra.append(f"VE: {v['ve']}")
            if "price_eur" in v: extra.append(f"Preis: {v['price_eur']} €")

            text = " | ".join(filter(None, [title, category, desc, ausf, bestell, " ".join(extra)]))

            docs.append({
                "title": title,
                "category": category,
                "description": desc,
                "ausfuehrung": ausf,
                "bestell_nr": bestell,
                "bestell_nr_norm": norm_id(bestell),
                "extra": extra,
                "text": text
            })
    return docs

def main():
    print("📦 Lade Produkte aus:", SRC_JSON)
    docs = build_docs()
    print(f"🧠 Erzeuge Vektoren für {len(docs)} Varianten …")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    vecs = model.encode([d["text"] for d in docs], normalize_embeddings=True, show_progress_bar=True)
    np.save(VECS_NPY, vecs)
    DOCS_JSON.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Fertig: {DOCS_JSON.name} + {VECS_NPY.name}")

if __name__ == "__main__":
    main()
