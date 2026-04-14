# ============================================
# STEP 3 — SPLIT LARGE LEVEL-2 BRANCHES
# ============================================
# Задача:
# Разбить только крупные ВЕТКИ уровня 2 на уровень 3.
#
# Вход:
# - book_vectors_chunk_centered.npy
# - book_clusters_level2.jsonl
#
# Выход:
# - book_clusters_level3.jsonl
#
# Логика:
# 1. Загружаем векторы книг
# 2. Загружаем результаты уровня 2:
#    book_id, cluster_id, subcluster_id
# 3. Находим только реальные ветки уровня 2, которые:
#    - принадлежат cluster_id != -1
#    - имеют subcluster_id != -1
#    - находятся внутри cluster_id, который реально делился на уровне 2
#    - имеют size > LEVEL3_THRESHOLD
# 4. Для каждой такой ветки:
#    - берем ее книги
#    - UMAP -> HDBSCAN
#    - сохраняем subsubcluster_id
# 5. Для остальных:
#    - subsubcluster_id = 0
#
# Важно:
# - НЕ делим шум уровня 1
# - НЕ делим шум уровня 2
# - НЕ делим кластеры, которые на уровне 2 вообще не дробились
# ============================================

import json
import time
from collections import Counter, defaultdict

import numpy as np
import umap
import hdbscan


# -------------------------
# ПУТИ
# -------------------------
VEC_PATH = r"C:\VAST2_UPLOAD\out\book_vectors\book_vectors_chunk_centered.npy"
LEVEL2_PATH = r"C:\VAST2_UPLOAD\out\book_vectors\book_clusters_level2.jsonl"
OUT_PATH = r"C:\VAST2_UPLOAD\out\book_vectors\book_clusters_level3.jsonl"


# -------------------------
# ПАРАМЕТРЫ
# -------------------------
LEVEL3_THRESHOLD = 120

UMAP_DIM = 10
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.05
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42

HDBSCAN_MIN_CLUSTER_SIZE = 12
HDBSCAN_MIN_SAMPLES = 6
HDBSCAN_METRIC = "euclidean"


def load_level2_rows(path):
    """
    Загружает JSONL формата:
    {
      "book_id": int,
      "cluster_id": int,
      "subcluster_id": int
    }
    """
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            for key in ("book_id", "cluster_id", "subcluster_id"):
                if key not in obj:
                    raise ValueError(f"Line {line_no}: missing key {key}")

            rows.append({
                "book_id": int(obj["book_id"]),
                "cluster_id": int(obj["cluster_id"]),
                "subcluster_id": int(obj["subcluster_id"]),
            })

    if not rows:
        raise ValueError("Level2 file is empty")

    rows.sort(key=lambda x: x["book_id"])

    # Проверка непрерывности book_id
    for i, row in enumerate(rows):
        if row["book_id"] != i:
            raise ValueError(
                f"book_id mismatch at sorted position {i}: got {row['book_id']}"
            )

    return rows


def build_maps(level2_rows):
    """
    Строит:
    - cluster_id -> list(book_id)
    - (cluster_id, subcluster_id) -> list(book_id)
    - cluster_id -> set(subcluster_id)
    """
    cluster_map = defaultdict(list)
    branch_map = defaultdict(list)
    cluster_to_subs = defaultdict(set)

    for row in level2_rows:
        book_id = row["book_id"]
        cid = row["cluster_id"]
        sid = row["subcluster_id"]

        cluster_map[cid].append(book_id)
        branch_map[(cid, sid)].append(book_id)
        cluster_to_subs[cid].add(sid)

    return cluster_map, branch_map, cluster_to_subs


def is_cluster_really_split(cluster_id, sub_ids):
    """
    Кластер считаем реально разделенным на уровне 2,
    если внутри него есть хотя бы 2 не-шумовых subcluster_id.

    Примеры:
    [0] -> не делился
    [-1, 0, 1, 2, 3] -> делился
    [-1, 1] -> формально делился, но очень грубо; тоже считаем делившимся
    """
    if cluster_id == -1:
        return False

    non_noise = [x for x in sub_ids if x != -1]
    return len(non_noise) >= 2


def select_level3_candidates(branch_map, cluster_to_subs, threshold):
    """
    Выбираем ветки для уровня 3.

    Берем только:
    - cluster_id != -1
    - subcluster_id != -1
    - cluster_id реально делился на уровне 2
    - размер ветки > threshold
    """
    candidates = []

    for (cid, sid), ids in branch_map.items():
        if cid == -1:
            continue

        if sid == -1:
            continue

        if not is_cluster_really_split(cid, cluster_to_subs[cid]):
            continue

        size = len(ids)
        if size > threshold:
            candidates.append((cid, sid, size))

    candidates.sort(key=lambda x: (-x[2], x[0], x[1]))
    return candidates


def run_subclustering(X_sub):
    """
    UMAP -> HDBSCAN
    """
    umap_model = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=UMAP_DIM,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
    )

    X_umap = umap_model.fit_transform(X_sub)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
    )

    labels = clusterer.fit_predict(X_umap)
    return labels


def save_level3(path, level2_rows, subsubcluster_labels):
    """
    Сохраняет:
    {
      "book_id": ...,
      "cluster_id": ...,
      "subcluster_id": ...,
      "subsubcluster_id": ...
    }
    """
    with open(path, "w", encoding="utf-8") as f:
        for row in level2_rows:
            book_id = row["book_id"]
            out = {
                "book_id": book_id,
                "cluster_id": int(row["cluster_id"]),
                "subcluster_id": int(row["subcluster_id"]),
                "subsubcluster_id": int(subsubcluster_labels[book_id]),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def main():
    t0 = time.time()

    print("=== STEP 3: SPLIT LARGE LEVEL-2 BRANCHES ===")

    # -------------------------
    # 1. Векторы
    # -------------------------
    print("Loading vectors...")
    X = np.load(VEC_PATH).astype(np.float32)
    n_books, dim = X.shape
    print(f"Books: {n_books}, dim: {dim}")

    # -------------------------
    # 2. Level 2
    # -------------------------
    print("Loading level2 clusters...")
    level2_rows = load_level2_rows(LEVEL2_PATH)

    if len(level2_rows) != n_books:
        raise ValueError(
            f"Mismatch: vectors={n_books}, level2_rows={len(level2_rows)}"
        )

    # -------------------------
    # 3. Карты
    # -------------------------
    cluster_map, branch_map, cluster_to_subs = build_maps(level2_rows)

    print(f"Unique cluster_id count: {len(cluster_map)}")
    print(f"Unique level2 branches: {len(branch_map)}")

    # -------------------------
    # 4. Кандидаты уровня 3
    # -------------------------
    candidates = select_level3_candidates(
        branch_map=branch_map,
        cluster_to_subs=cluster_to_subs,
        threshold=LEVEL3_THRESHOLD,
    )

    print(f"Level3 candidate branches (size > {LEVEL3_THRESHOLD}): {len(candidates)}")
    if candidates:
        print("List:")
        for cid, sid, size in candidates:
            print(f"  cluster {cid}, subcluster {sid}: {size} books")
    else:
        print("No valid candidate branches found.")

    # -------------------------
    # 5. По умолчанию:
    #    subsubcluster_id = 0
    # -------------------------
    subsubcluster_labels = np.zeros(n_books, dtype=np.int32)

    total_processed = 0

    # -------------------------
    # 6. Деление кандидатов
    # -------------------------
    for idx, (cid, sid, size) in enumerate(candidates, 1):
        ids = branch_map[(cid, sid)]
        total_processed += len(ids)

        print()
        print(
            f"[{idx}/{len(candidates)}] Processing "
            f"cluster {cid}, subcluster {sid} ({len(ids)} books)"
        )

        X_sub = X[ids]

        t_branch = time.time()
        labels = run_subclustering(X_sub)

        for local_i, book_id in enumerate(ids):
            subsubcluster_labels[book_id] = int(labels[local_i])

        counter = Counter(labels)
        n_subsub = len([x for x in counter if x != -1])
        n_noise = counter.get(-1, 0)

        print(f"  subsubclusters found: {n_subsub}")
        print(f"  subsubcluster noise: {n_noise}")
        print("  top subsubclusters:")
        for subsub_id, subsub_size in sorted(counter.items(), key=lambda x: (-x[1], x[0]))[:10]:
            print(f"    subsubcluster {subsub_id}: {subsub_size}")

        print(f"  done in {round(time.time() - t_branch, 2)} sec")

    # -------------------------
    # 7. Сохранение
    # -------------------------
    print()
    print("Saving level3 results...")
    save_level3(OUT_PATH, level2_rows, subsubcluster_labels)

    # -------------------------
    # 8. Финальная статистика
    # -------------------------
    final_counter = Counter(subsubcluster_labels.tolist())

    print("DONE")
    print(f"Saved: {OUT_PATH}")
    print(f"Processed books on level 3: {total_processed}")
    print("Global subsubcluster_id distribution (top 15):")
    for sid, cnt in sorted(final_counter.items(), key=lambda x: (-x[1], x[0]))[:15]:
        print(f"  subsubcluster_id {sid}: {cnt}")

    print(f"Total time: {round(time.time() - t0, 2)} sec")


if __name__ == "__main__":
    main()