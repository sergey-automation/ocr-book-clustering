# ============================================
# 03_build_book_vectors_chunk_centered.py
# ============================================
# Задача:
# Построить centered-вектора книг, используя mean_chunk,
# и ОБЯЗАТЕЛЬНО сохранить mapping строк массива к книгам.
#
# Вход:
# - book_vectors_full.npy
# - book_vectors_full.jsonl
# - mean_chunk.npy
#
# Выход:
# - book_vectors_chunk_centered.npy
# - book_vectors_chunk_centered.jsonl
# - book_vectors_chunk_centered_summary.json
#
# Важно:
# - порядок книг НЕ меняется
# - строка i в .npy соответствует строке i в .jsonl
# - mapping сохраняется явно, чтобы потом HDBSCAN и другие
#   этапы могли безошибочно восстановить doc_id / txt_rel / year
# ============================================

import os
import json
import time
import numpy as np

# ---------------- CONFIG ----------------
SRC_VEC_PATH = r"C:\VAST2_UPLOAD\out\book_vectors\book_vectors_full.npy"
SRC_META_PATH = r"C:\VAST2_UPLOAD\out\book_vectors\book_vectors_full.jsonl"
MEAN_CHUNK_PATH = r"C:\VAST2_UPLOAD\out\book_vectors\mean_chunk.npy"

OUT_VEC_PATH = r"C:\VAST2_UPLOAD\out\book_vectors\book_vectors_chunk_centered.npy"
OUT_META_PATH = r"C:\VAST2_UPLOAD\out\book_vectors\book_vectors_chunk_centered.jsonl"
OUT_SUMMARY_PATH = r"C:\VAST2_UPLOAD\out\book_vectors\book_vectors_chunk_centered_summary.json"

DTYPE = np.float32
EPS = 1e-12
# ---------------------------------------


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    t0 = time.time()

    print("Loading book vectors...")
    X = np.load(SRC_VEC_PATH).astype(DTYPE, copy=False)

    print("Loading book meta...")
    meta_rows = load_jsonl(SRC_META_PATH)

    print("Loading mean_chunk...")
    mean_chunk = np.load(MEAN_CHUNK_PATH).astype(DTYPE, copy=False)

    n_books, dim = X.shape
    print(f"Books: {n_books}, dim: {dim}")

    if len(meta_rows) != n_books:
        raise RuntimeError(
            f"Mismatch: vectors rows={n_books}, meta rows={len(meta_rows)}"
        )

    if mean_chunk.shape[0] != dim:
        raise RuntimeError(
            f"Mismatch: mean_chunk dim={mean_chunk.shape[0]}, vectors dim={dim}"
        )

    # ----------------------------------------
    # Проверка, что vector_row согласован с порядком строк
    # ----------------------------------------
    print("Validating meta order...")
    vector_row_errors = 0

    for i, row in enumerate(meta_rows):
        vr = row.get("vector_row")
        if vr is not None and int(vr) != i:
            vector_row_errors += 1

    if vector_row_errors > 0:
        raise RuntimeError(
            f"Meta order error: found {vector_row_errors} rows where vector_row != row index"
        )

    # ----------------------------------------
    # Centering: вычитаем mean_chunk из каждого вектора книги
    # ----------------------------------------
    print("Centering by mean_chunk...")
    X_centered = X - mean_chunk

    # ----------------------------------------
    # Нормализация каждой строки до длины 1
    # ----------------------------------------
    print("Normalizing...")
    norms = np.linalg.norm(X_centered, axis=1, keepdims=True)
    norms = np.maximum(norms, EPS)
    X_centered = X_centered / norms
    X_centered = X_centered.astype(DTYPE, copy=False)

    # ----------------------------------------
    # Готовим meta для centered-файла
    # Порядок строк сохраняем тот же
    # ----------------------------------------
    print("Preparing centered meta...")
    centered_meta_rows = []

    for i, row in enumerate(meta_rows):
        new_row = dict(row)

        # гарантируем правильный индекс строки
        new_row["vector_row"] = i

        # фиксируем, что это centered-вектора по mean_chunk
        new_row["vector_kind"] = "book_vectors_chunk_centered"
        new_row["centering_source"] = "mean_chunk.npy"

        centered_meta_rows.append(new_row)

    # ----------------------------------------
    # Сохраняем результаты
    # ----------------------------------------
    print("Saving vectors...")
    np.save(OUT_VEC_PATH, X_centered)

    print("Saving meta...")
    write_jsonl(OUT_META_PATH, centered_meta_rows)

    # ----------------------------------------
    # Summary
    # ----------------------------------------
    summary = {
        "books_total": int(n_books),
        "embedding_dim": int(dim),
        "dtype": str(DTYPE),
        "source_vectors": SRC_VEC_PATH,
        "source_meta": SRC_META_PATH,
        "source_mean_chunk": MEAN_CHUNK_PATH,
        "out_vectors": OUT_VEC_PATH,
        "out_meta": OUT_META_PATH,
        "meta_rows": int(len(centered_meta_rows)),
        "vector_row_validation_errors": int(vector_row_errors),
        "order_preserved": True,
        "normalization": "l2",
        "centering": "subtract mean_chunk then normalize",
        "time_sec": float(time.time() - t0),
    }

    with open(OUT_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("DONE")
    print(f"Written: {OUT_VEC_PATH}")
    print(f"Written: {OUT_META_PATH}")
    print(f"Written: {OUT_SUMMARY_PATH}")
    print(f"Time: {time.time() - t0:.2f} sec")


if __name__ == "__main__":
    main()