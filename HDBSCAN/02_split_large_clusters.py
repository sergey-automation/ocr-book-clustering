# ============================================
# STEP 2 — SPLIT LARGE CLUSTERS (уровень 2)
# ============================================
# Задача:
# Разбить только КРУПНЫЕ кластеры уровня 1 на подкластеры.
#
# Вход:
# - book_vectors_chunk_centered.npy
# - book_clusters_level1.jsonl
#
# Выход:
# - book_clusters_level2.jsonl
#
# Логика:
# 1. Загружаем векторы книг
# 2. Загружаем cluster_id уровня 1
# 3. Находим крупные кластеры: size > LARGE_THRESHOLD
# 4. Для каждого крупного кластера:
#    - берём только его книги
#    - UMAP: 768 -> 10
#    - HDBSCAN
#    - сохраняем subcluster_id
# 5. Для малых кластеров:
#    - subcluster_id = 0
# 6. Шум HDBSCAN внутри крупного кластера:
#    - subcluster_id = -1
# ============================================

import os
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
LEVEL1_PATH = r"C:\VAST2_UPLOAD\out\book_vectors\book_clusters_level1.jsonl"
OUT_PATH = r"C:\VAST2_UPLOAD\out\book_vectors\book_clusters_level2.jsonl"


# -------------------------
# ПАРАМЕТРЫ
# -------------------------
LARGE_THRESHOLD = 150

UMAP_DIM = 10
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.05
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42

HDBSCAN_MIN_CLUSTER_SIZE = 12
HDBSCAN_MIN_SAMPLES = 6
HDBSCAN_METRIC = "euclidean"


# -------------------------
# ВСПОМОГАТЕЛЬНОЕ
# -------------------------
def load_level1_clusters(path):
    """
    Загружает JSONL с полями:
    {"book_id": int, "cluster_id": int}

    Возвращает:
    - level1_labels: list[int] длиной N, индекс = book_id
    """
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            if "book_id" not in obj or "cluster_id" not in obj:
                raise ValueError(
                    f"Bad JSONL at line {line_no}: no book_id / cluster_id"
                )

            rows.append((int(obj["book_id"]), int(obj["cluster_id"])))

    if not rows:
        raise ValueError("Level1 file is empty")

    rows.sort(key=lambda x: x[0])

    max_book_id = rows[-1][0]
    labels = [None] * (max_book_id + 1)

    for book_id, cluster_id in rows:
        if labels[book_id] is not None:
            raise ValueError(f"Duplicate book_id in level1 file: {book_id}")
        labels[book_id] = cluster_id

    missing = [i for i, x in enumerate(labels) if x is None]
    if missing:
        raise ValueError(f"Missing book_id(s) in level1 file, example: {missing[:10]}")

    return labels


def build_cluster_map(level1_labels):
    """
    cluster_id -> list[book_id]
    """
    mp = defaultdict(list)
    for book_id, cluster_id in enumerate(level1_labels):
        mp[cluster_id].append(book_id)
    return mp


def run_subclustering(X_sub):
    """
    Для векторов одного крупного кластера:
    - UMAP
    - HDBSCAN

    Возвращает:
    - labels_sub: np.ndarray[int], длина = len(X_sub)
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

    labels_sub = clusterer.fit_predict(X_umap)
    return labels_sub


def save_level2(path, level1_labels, subcluster_labels):
    """
    Сохраняет:
    {"book_id": i, "cluster_id": ..., "subcluster_id": ...}
    """
    with open(path, "w", encoding="utf-8") as f:
        for book_id in range(len(level1_labels)):
            row = {
                "book_id": book_id,
                "cluster_id": int(level1_labels[book_id]),
                "subcluster_id": int(subcluster_labels[book_id]),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# -------------------------
# ОСНОВНАЯ ПРОГРАММА
# -------------------------
def main():
    t0 = time.time()

    print("=== STEP 2: SPLIT LARGE CLUSTERS ===")

    # 1. Загрузка векторов
    print("Loading vectors...")
    X = np.load(VEC_PATH).astype(np.float32)
    n_books, dim = X.shape
    print(f"Books: {n_books}, dim: {dim}")

    # 2. Загрузка level1
    print("Loading level1 clusters...")
    level1_labels = load_level1_clusters(LEVEL1_PATH)

    if len(level1_labels) != n_books:
        raise ValueError(
            f"Mismatch: vectors={n_books}, level1_labels={len(level1_labels)}"
        )

    # 3. cluster_id -> список book_id
    cluster_map = build_cluster_map(level1_labels)
    cluster_sizes = {cid: len(ids) for cid, ids in cluster_map.items()}

    print(f"Unique level1 labels (including noise): {len(cluster_sizes)}")

    # 4. Найти крупные кластеры
    large_clusters = [
        cid for cid, size in sorted(cluster_sizes.items(), key=lambda x: (-x[1], x[0]))
        if cid != -1 and size > LARGE_THRESHOLD
    ]

    print(f"Large clusters to split (size > {LARGE_THRESHOLD}): {len(large_clusters)}")
    if large_clusters:
        print("List:")
        for cid in large_clusters:
            print(f"  cluster {cid}: {cluster_sizes[cid]} books")
    else:
        print("No large clusters found.")

    # 5. По умолчанию:
    #    для всех книг subcluster_id = 0
    #    потом для крупных кластеров заменим на реальные значения
    subcluster_labels = np.zeros(n_books, dtype=np.int32)

    # Отдельно можно сохранить шум уровня 1 как 0.
    # Здесь сознательно НЕ делаем -1 для шума уровня 1,
    # потому что subcluster_id относится к уровню 2.
    # Для cluster_id = -1 оставляем subcluster_id = 0.

    total_large_books = 0

    # 6. Обработка крупных кластеров
    for idx, cid in enumerate(large_clusters, 1):
        ids = cluster_map[cid]
        size = len(ids)
        total_large_books += size

        print()
        print(f"[{idx}/{len(large_clusters)}] Processing cluster {cid} ({size} books)")

        X_sub = X[ids]

        t_cluster = time.time()
        labels_sub = run_subclustering(X_sub)

        # Сохранить labels_sub обратно в общий массив
        for local_i, book_id in enumerate(ids):
            subcluster_labels[book_id] = int(labels_sub[local_i])

        # Статистика по подкластерам
        counter_sub = Counter(labels_sub)
        n_subclusters = len([x for x in counter_sub if x != -1])
        n_noise = counter_sub.get(-1, 0)

        print(f"  subclusters found: {n_subclusters}")
        print(f"  subcluster noise: {n_noise}")

        print("  top subclusters:")
        for sub_id, sub_size in sorted(counter_sub.items(), key=lambda x: (-x[1], x[0]))[:10]:
            print(f"    subcluster {sub_id}: {sub_size}")

        print(f"  done in {round(time.time() - t_cluster, 2)} sec")

    # 7. Сохранение
    print()
    print("Saving level2 results...")
    save_level2(OUT_PATH, level1_labels, subcluster_labels)

    # 8. Общая статистика
    counter_level2 = Counter(subcluster_labels.tolist())

    print("DONE")
    print(f"Saved: {OUT_PATH}")
    print(f"Large-cluster books processed: {total_large_books}")
    print(f"Subcluster label distribution (global top 15):")
    for sub_id, cnt in sorted(counter_level2.items(), key=lambda x: (-x[1], x[0]))[:15]:
        print(f"  subcluster_id {sub_id}: {cnt}")

    print(f"Total time: {round(time.time() - t0, 2)} sec")


if __name__ == "__main__":
    main()