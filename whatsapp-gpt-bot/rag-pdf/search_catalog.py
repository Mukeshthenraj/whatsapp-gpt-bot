import json
import re
import unicodedata
import argparse
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import process, fuzz

CATALOG_JSON = "kellen_produkte.json"
DOCS_JSON = "catalog_docs.json"
VECTORS_NPY = "catalog_vectors.npy"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^a-z0-9äöüß\.\s-]", " ", s)  # keep hyphen for readability
    s = re.sub(r"\s+", " ", s).strip()
    return s.replace("-", " ")  # treat hyphen as space for matching

def split_words(s: str):
    return [w for w in s.split() if w]

def norm_bestell(s: str) -> str:
    return s.replace(" ", "").strip() if s else ""

def digits_only(s: str) -> str:
    return re.sub(r"\D", "", s or "")

def _to_float(x):
    if x in (None, "", "-"):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None

def _map_variant_fields(v: Dict[str, Any]) -> Dict[str, Any]:
    bestell = v.get("bestell_nr") or v.get("Bestell-Nr.") or v.get("Bestell-Nr") or ""
    ausf   = v.get("ausfuehrung") or v.get("Ausführung") or v.get("Ausfuehrung") or ""
    l_mm   = v.get("l_mm") or v.get("L mm") or v.get("L")
    b_mm   = v.get("b_mm") or v.get("B mm") or v.get("B")
    h_mm   = v.get("h_mm") or v.get("H mm") or v.get("H")
    staerke_mm = v.get("staerke_mm") or v.get("Stärke") or v.get("Staerke")
    price = v.get("price_eur") or v.get("€") or v.get("EUR") or v.get("Preis")
    ve = v.get("ve") or v.get("VE")
    return {
        "bestell_nr": str(bestell).strip(),
        "ausfuehrung": str(ausf or ""),
        "l_mm": l_mm, "b_mm": b_mm, "h_mm": h_mm, "staerke_mm": staerke_mm,
        "price_eur": _to_float(price), "ve": ve,
    }

SYNONYMS = {
    "herzgriffspachtel": ["herzspachtel", "herzform spachtel", "spachtel herzform"],
    "flächenspachtel": ["flachspachtel", "flächen spachtel"],
}

def expand_synonyms(text: str) -> str:
    norm = normalize_text(text)
    parts = [norm]
    for key, syns in SYNONYMS.items():
        if key in norm:
            parts.extend(syns)
    return " ".join(parts)

def flatten_products(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for prod in data:
        title = (prod.get("title") or prod.get("Title") or "").strip()
        category = (prod.get("category") or prod.get("Kategorie") or "").strip()
        description = (prod.get("description") or prod.get("Beschreibung") or "").strip()
        variants = prod.get("items") if isinstance(prod.get("items"), list) else prod.get("variants") or []
        for v in variants:
            m = _map_variant_fields(v)
            row = {
                "bestell_nr": m["bestell_nr"],
                "title": title,
                "category": category,
                "ausfuehrung": m["ausfuehrung"],
                "description": description,
                "price_eur": m["price_eur"],
                "ve": m["ve"],
            }
            for fld in ("l_mm","b_mm","h_mm","staerke_mm"):
                if m.get(fld) not in (None, ""): row[fld] = m[fld]

            field_blob = " ".join([p for p in [
                title, category, description, row["ausfuehrung"],
                str(row.get("l_mm","")), str(row.get("b_mm","")),
                str(row.get("h_mm","")), str(row.get("staerke_mm","")),
            ] if p])
            row["_blob"] = expand_synonyms(field_blob)
            row["_title_norm"] = normalize_text(title)
            row["_cat_norm"]   = normalize_text(category)
            row["_ausf_norm"]  = normalize_text(row["ausfuehrung"])
            row["_bestell_norm"]   = norm_bestell(row["bestell_nr"])
            row["_bestell_digits"] = digits_only(row["_bestell_norm"])
            rows.append(row)
    return rows

def build_index():
    with open(CATALOG_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = flatten_products(data)
    model = SentenceTransformer(EMBED_MODEL)
    texts = [d["_blob"] for d in docs]
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    with open(DOCS_JSON, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    np.save(VECTORS_NPY, vectors)
    print(f"✅ Index built with {len(docs)} products")

# -------- name-first helpers --------

def title_prefilter(docs, norm_q: str):
    """Prioritize matches where the query fits the TITLE specifically."""
    q_tokens = split_words(norm_q)
    if not q_tokens: return []
    out = []
    for d in docs:
        t = d["_title_norm"]
        # substring hit (very strong)
        if norm_q and (norm_q in t or t in norm_q):
            out.append((1.0, d))
            continue
        # token recall just on title tokens
        t_tokens = set(split_words(t))
        hits = sum(1 for tok in q_tokens if tok in t_tokens)
        if hits == 0: continue
        recall = hits / len(q_tokens)
        if recall >= 0.6:
            out.append((recall, d))
    out.sort(key=lambda x: -x[0])
    return [d for _, d in out]

def prefilter_literal(docs, norm_q: str):
    """Literal recall on the full blob (kept strict)."""
    q_tokens = split_words(norm_q)
    if not q_tokens: return []
    scored = []
    for d in docs:
        blob = d["_blob"]
        hits = sum(1 for t in q_tokens if t in blob)
        if hits == 0: continue
        recall = hits / len(q_tokens)
        if recall >= 0.6:
            scored.append((recall, hits, d))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    return [d for _,__, d in scored]

def fuzzy_multi(docs, norm_q: str, limit: int):
    """Fuzzy across title + ausfuehrung + category."""
    keys = [f"{d['title']} {d.get('ausfuehrung','')} {d.get('category','')}".strip() for d in docs]
    res = process.extract(norm_q, keys, scorer=fuzz.token_set_ratio, limit=limit)
    out, seen = [], set()
    for _, score, idx in res:
        if score < 68:  # a little more forgiving
            continue
        if idx in seen: continue
        seen.add(idx)
        out.append(docs[idx])
    return out

def search(query: str, top_k=25, bestell_only: bool=False):
    with open(DOCS_JSON, "r", encoding="utf-8") as f:
        docs = json.load(f)
    vectors = np.load(VECTORS_NPY)
    model = SentenceTransformer(EMBED_MODEL)

    raw_q = query or ""
    norm_q = normalize_text(raw_q)

    # very short, no digits → let bot (LLM) handle small talk/math/etc.
    if len(norm_q) <= 3 and not re.search(r"\d", norm_q):
        return []

    # 1) exact Bestell-Nr.
    for d in docs:
        if d["bestell_nr"].strip().lower() == raw_q.strip().lower():
            return [d]

    # 2) digits-only Bestell-Nr.
    q_digits = digits_only(raw_q)
    if q_digits:
        for d in docs:
            if d["_bestell_digits"] == q_digits:
                return [d]

    # Bestell-only mode stops here
    if bestell_only:
        return []

    # 3) TITLE-FIRST (fixes “Ersatz-Belag”, etc.)
    t_hits = title_prefilter(docs, norm_q)
    if t_hits:
        return t_hits[:top_k]

    # 4) literal blob
    lit = prefilter_literal(docs, norm_q)
    if lit:
        return lit[:top_k]

    # 5) fuzzy
    fuzzy = fuzzy_multi(docs, norm_q, limit=max(50, top_k))
    if fuzzy:
        return fuzzy[:top_k]

    # 6) semantic
    q_vec = model.encode([norm_q], convert_to_numpy=True)
    hits = util.semantic_search(q_vec, vectors, top_k=top_k)[0]
    return [docs[h["corpus_id"]] for h in hits]

def _print_result(r: Dict[str, Any]):
    print(f"* {r['title']} — {r.get('ausfuehrung','')}".rstrip(" —"))
    print(f"  Kategorie: {r.get('category','')}")
    print(f"  Bestell-Nr.: {r.get('bestell_nr','')}")
    print(f"  Preis: {r.get('price_eur')} € | VE: {r.get('ve')}")
    dims = []
    for fld in ("l_mm","b_mm","h_mm","staerke_mm"):
        if fld in r and r[fld] not in (None, ""):
            dims.append(f"{fld.upper()}={r[fld]}")
    if dims: print("  " + ", ".join(dims))
    print()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="?", help="Search term")
    p.add_argument("--build", action="store_true", help="Build the index")
    p.add_argument("--top", type=int, default=25, help="max results")
    p.add_argument("--bestell-only", action="store_true", help="only return on exact/digits Bestell-Nr. match")
    args = p.parse_args()

    if args.build:
        build_index()
    elif args.query:
        results = search(args.query, top_k=args.top, bestell_only=args.bestell_only)
        if results:
            for r in results: _print_result(r)
        else:
            print("❌ Kein exakter Bestell-Nr.-Treffer." if args.bestell_only else "❌ Keine Treffer.")
    else:
        p.print_help()
