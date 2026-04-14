import os
import json
import re
import time
from collections import defaultdict
from glob import glob

import numpy as np

# ---------------- CONFIG ----------------
CHUNKS_PATH = r"C:\VAST2_UPLOAD\chunks_engineering_ru_rag.jsonl"
PARTS_DIR = r"C:\VAST2_UPLOAD\out\engineering_ru_rag_e5_base_torch\parts"
OUT_DIR = r"C:\VAST2_UPLOAD\out\book_vectors"

TRIM_START = 5
TRIM_END = 10
DTYPE = np.float32

# --- new quality/cleaning config ---
EMBED_DIM = 768
SKIP_FIRST_PAGES = 10           # шаг 2
MAX_GOOD_CHUNKS_PER_BOOK = 20   # шаг 4
EPS = 1e-12

# быстрые эвристики качества текста чанка
MIN_TEXT_LEN = 120
MIN_LETTER_RATIO = 0.75
MAX_DIGIT_RATIO = 0.30
MIN_AVG_TOKEN_LEN = 3.0

BAD_SUBSTRINGS = [
    "card number",
    "intentionally left blank",
    "left blank",
    "author index",
    "table of contents",
    "contents",
    "references",
    "subject index",
    "index",
    "remarks references",
    "page intentionally",
    "rights reserved",
    "представляет собой",
    "в настоящее время",
    "в данной работе",
    "как правило",
    "имеет место",
    "следует отметить",
    "одним из",
    "в целом",
    "written permission",
    "retrieval system",
]

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")

# ---------------------------------------


def tokenize_simple(text: str):
    return TOKEN_RE.findall(text)


def text_quality_metrics(text: str):
    txt = text.strip()
    n = len(txt)
    if n == 0:
        return {
            "text_len": 0,
            "letter_ratio": 0.0,
            "digit_ratio": 0.0,
            "avg_token_len": 0.0,
        }

    letters = sum(ch.isalpha() for ch in txt)
    digits = sum(ch.isdigit() for ch in txt)
    tokens = tokenize_simple(txt)
    avg_token_len = (sum(len(t) for t in tokens) / len(tokens)) if tokens else 0.0

    return {
        "text_len": n,
        "letter_ratio": letters / n,
        "digit_ratio": digits / n,
        "avg_token_len": avg_token_len,
    }


def contains_bad_pattern(text: str) -> bool:
    t = text.lower()
    for p in BAD_SUBSTRINGS:
        if p in t:
            return True
    return False


def is_good_chunk(meta_row: dict) -> tuple[bool, dict]:
    text = str(meta_row.get("text") or "")
    qm = text_quality_metrics(text)

    ok = True
    if qm["text_len"] < MIN_TEXT_LEN:
        ok = False
    if qm["letter_ratio"] < MIN_LETTER_RATIO:
        ok = False
    if qm["digit_ratio"] > MAX_DIGIT_RATIO:
        ok = False
    if qm["avg_token_len"] < MIN_AVG_TOKEN_LEN:
        ok = False
    if contains_bad_pattern(text):
        ok = False

    return ok, qm


def chunk_weight(qm: dict) -> float:
    # шаг 3: вес чанка
    w = qm["letter_ratio"] * (1.0 - qm["digit_ratio"])
    return max(w, 0.0)


os.makedirs(OUT_DIR, exist_ok=True)

print("Loading chunks index...")

chunk_meta = {}
doc_pages = defaultdict(list)
doc_info = {}

# Храним также text для фильтра качества
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        j = json.loads(line)

        cid = j["chunk_id"]
        doc = j["doc_id"]
        page = int(j.get("page_start", 0) or 0)
        txt_rel = j.get("txt_rel")
        year = j.get("year")
        text = j.get("text", "")

        chunk_meta[cid] = {
            "doc_id": doc,
            "page_start": page,
            "text": text,
        }
        doc_pages[doc].append(page)

        if doc not in doc_info:
            doc_info[doc] = (txt_rel, year)

doc_stats = {}
for doc, pages in doc_pages.items():
    doc_stats[doc] = (min(pages), max(pages), len(pages))

print("Docs:", len(doc_stats))

# Старые агрегаторы FULL / TRIM оставляем для совместимости
sum_full = defaultdict(lambda: np.zeros(EMBED_DIM, dtype=np.float64))
cnt_full = defaultdict(int)

sum_trim = defaultdict(lambda: np.zeros(EMBED_DIM, dtype=np.float64))
cnt_trim = defaultdict(int)

used_fallback = defaultdict(bool)

# Новое: собираем хорошие чанки на книгу
good_chunks_by_doc = defaultdict(list)

chunks_total = 0
chunks_trim_used = 0

chunks_skipped_first_pages = 0
chunks_rejected_quality = 0
chunks_good_collected = 0
chunks_used_weighted = 0
books_with_weighted = 0
books_with_weighted_fallback = 0

print("Processing parts...")

npy_files = sorted(glob(os.path.join(PARTS_DIR, "*_emb_part_*.npy")))
total_parts = len(npy_files)

start_time = time.time()

for part_idx, npy_file in enumerate(npy_files):
    meta_file = npy_file.replace("_emb_", "_meta_").replace(".npy", ".npz")

    emb = np.load(npy_file)
    meta = np.load(meta_file, allow_pickle=True)

    if "chunk_ids" in meta:
        ids = meta["chunk_ids"]
    else:
        raise RuntimeError("chunk_ids not found")

    for i, cid in enumerate(ids):
        cid = cid.decode() if isinstance(cid, bytes) else cid

        if cid not in chunk_meta:
            continue

        row = chunk_meta[cid]
        doc = row["doc_id"]
        page = row["page_start"]
        text = row["text"]
        v = emb[i]

        # ---- старый FULL ----
        sum_full[doc] += v
        cnt_full[doc] += 1

        # ---- старый TRIM ----
        page_min, page_max, _ = doc_stats[doc]
        if page > TRIM_START and page <= (page_max - TRIM_END):
            sum_trim[doc] += v
            cnt_trim[doc] += 1
            chunks_trim_used += 1

        # ---- новый отбор для cleaned/weighted FULL ----
        if page <= SKIP_FIRST_PAGES:
            chunks_skipped_first_pages += 1
        else:
            ok, qm = is_good_chunk({"text": text})
            if ok:
                w = chunk_weight(qm)
                if w > 0:
                    good_chunks_by_doc[doc].append((w, v.astype(np.float64, copy=False)))
                    chunks_good_collected += 1
            else:
                chunks_rejected_quality += 1

        chunks_total += 1

    elapsed = time.time() - start_time
    speed = chunks_total / max(elapsed, 1e-6)
    parts_done = part_idx + 1
    parts_left = total_parts - parts_done
    eta = (elapsed / parts_done) * parts_left if parts_done > 0 else 0

    print(f"[{parts_done}/{total_parts}] "
          f"chunks={chunks_total} "
          f"good={chunks_good_collected} "
          f"rej={chunks_rejected_quality} "
          f"skip_first={chunks_skipped_first_pages} "
          f"speed={speed:.1f} ch/s "
          f"eta={eta/60:.1f} min")

print("Aggregating...")

docs = list(doc_stats.keys())
N = len(docs)

vec_full = np.zeros((N, EMBED_DIM), dtype=DTYPE)
vec_trim = np.zeros((N, EMBED_DIM), dtype=DTYPE)

meta_full = []
meta_trim = []

books_fallback = 0

for i, doc in enumerate(docs):
    page_min, page_max, _ = doc_stats[doc]
    txt_rel, year = doc_info.get(doc, (None, None))

    # ============================================================
    # FULL = теперь cleaned + weighted + top-K good chunks
    # fallback -> старый mean по всем чанкам, если good chunks нет
    # ============================================================
    good_items = good_chunks_by_doc.get(doc, [])
    good_items.sort(key=lambda x: x[0], reverse=True)
    good_items = good_items[:MAX_GOOD_CHUNKS_PER_BOOK]

    if good_items:
        vec_sum = np.zeros(EMBED_DIM, dtype=np.float64)
        w_sum = 0.0
        for w, v in good_items:
            vec_sum += v * w
            w_sum += w
            chunks_used_weighted += 1

        v = vec_sum / max(w_sum, EPS)
        books_with_weighted += 1
        full_used_fallback = False
    else:
        v = sum_full[doc] / max(cnt_full[doc], 1)
        books_with_weighted_fallback += 1
        full_used_fallback = True

    v = v / (np.linalg.norm(v) + EPS)
    vec_full[i] = v.astype(DTYPE)

    # ---------------- old TRIM preserved ----------------
    if cnt_trim[doc] == 0:
        v2 = v
        used_fallback[doc] = True
        books_fallback += 1
    else:
        v2 = sum_trim[doc] / cnt_trim[doc]
        v2 = v2 / (np.linalg.norm(v2) + EPS)

    vec_trim[i] = v2.astype(DTYPE)

    meta_full.append({
        "doc_id": doc,
        "txt_rel": txt_rel,
        "year": year,
        "page_min": page_min,
        "page_max": page_max,
        "chunks_count_full": cnt_full[doc],
        "chunks_count_good_after_skip": len(good_chunks_by_doc.get(doc, [])),
        "chunks_count_used_weighted": len(good_items),
        "used_weighted_fallback_full": full_used_fallback,
        "vector_row": i
    })

    meta_trim.append({
        "doc_id": doc,
        "txt_rel": txt_rel,
        "year": year,
        "page_min": page_min,
        "page_max": page_max,
        "chunks_count_full": cnt_full[doc],
        "chunks_count_trimmed": cnt_trim[doc],
        "used_fallback": used_fallback[doc],
        "vector_row": i
    })

print("Saving...")

np.save(os.path.join(OUT_DIR, "book_vectors_full.npy"), vec_full)
np.save(os.path.join(OUT_DIR, "book_vectors_trimmed.npy"), vec_trim)

with open(os.path.join(OUT_DIR, "book_vectors_full.jsonl"), "w", encoding="utf-8") as f:
    for x in meta_full:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

with open(os.path.join(OUT_DIR, "book_vectors_trimmed.jsonl"), "w", encoding="utf-8") as f:
    for x in meta_trim:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

summary = {
    "books_total": N,
    "books_with_fallback_trim": books_fallback,
    "books_with_weighted_full": books_with_weighted,
    "books_with_weighted_fallback_full": books_with_weighted_fallback,
    "chunks_total_seen": chunks_total,
    "chunks_used_trimmed": chunks_trim_used,
    "chunks_removed_by_trim": chunks_total - chunks_trim_used,
    "chunks_skipped_first_pages": chunks_skipped_first_pages,
    "chunks_rejected_quality": chunks_rejected_quality,
    "chunks_good_collected": chunks_good_collected,
    "chunks_used_weighted": chunks_used_weighted,
    "skip_first_pages": SKIP_FIRST_PAGES,
    "max_good_chunks_per_book": MAX_GOOD_CHUNKS_PER_BOOK,
    "min_text_len": MIN_TEXT_LEN,
    "min_letter_ratio": MIN_LETTER_RATIO,
    "max_digit_ratio": MAX_DIGIT_RATIO,
    "min_avg_token_len": MIN_AVG_TOKEN_LEN,
    "trim_start_pages": TRIM_START,
    "trim_end_pages": TRIM_END,
    "dtype": str(DTYPE),
    "embedding_dim": EMBED_DIM,
    "full_vector_method": "weighted_mean_of_good_chunks_after_skip_first_pages",
    "full_fallback_method": "plain_mean_of_all_chunks",
}

with open(os.path.join(OUT_DIR, "book_vectors_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("DONE")
print("Written:", os.path.join(OUT_DIR, "book_vectors_full.npy"))
print("Written:", os.path.join(OUT_DIR, "book_vectors_trimmed.npy"))
print("Written:", os.path.join(OUT_DIR, "book_vectors_full.jsonl"))
print("Written:", os.path.join(OUT_DIR, "book_vectors_trimmed.jsonl"))
print("Written:", os.path.join(OUT_DIR, "book_vectors_summary.json"))
