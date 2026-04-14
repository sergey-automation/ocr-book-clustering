# ============================================
# STEP 1 — HDBSCAN LEVEL 1 (весь корпус)
# ============================================
# Задача:
# Разбить ВСЕ книги на первичные кластеры (темы)
#
# Вход:
# - book_vectors_chunk_centered.npy (N книг × 768)
#
# Выход:
# - book_clusters_level1.jsonl
# - статистика кластеров
#
# Логика:
# 1. Загружаем векторы книг
# 2. UMAP: сжимаем пространство 768 -> 10
# 3. HDBSCAN: находим плотные кластеры
# 4. Сохраняем cluster_id для каждой книги
#
# Важно:
# - cluster_id = -1 означает шум / выброс
# - random_state зафиксирован для воспроизводимости UMAP
# ============================================

import json
import time
from collections import Counter

import numpy as np
import umap
import hdbscan


# -------------------------
# ПУТИ
# -------------------------
VEC_PATH = r"C:\VAST2_UPLOAD\out\book_vectors\book_vectors_chunk_centered.npy"
OUT_PATH = r"C:\VAST2_UPLOAD\out\book_vectors\book_clusters_level1.jsonl"


# -------------------------
# ПАРАМЕТРЫ
# -------------------------
# Размерность после UMAP
UMAP_DIM = 10

# Параметры HDBSCAN
MIN_CLUSTER_SIZE = 20
MIN_SAMPLES = 10

# Фиксация seed для воспроизводимости UMAP
RANDOM_STATE = 42


def main():
    # -------------------------
    # СТАРТ
    # -------------------------
    t0 = time.time()

    print("=== STEP 1: HDBSCAN LEVEL 1 ===")

    # -------------------------
    # 1. Загрузка векторов
    # -------------------------
    print("Loading vectors...")
    X = np.load(VEC_PATH).astype(np.float32)

    # Ожидаем форму: [число_книг, 768]
    print(f"Books: {X.shape[0]}, dim: {X.shape[1]}")

    # -------------------------
    # 2. UMAP (сжатие пространства)
    # -------------------------
    print("UMAP reduction...")

    # UMAP переводит исходные 768-мерные векторы
    # в более компактное пространство из 10 измерений.
    #
    # metric="cosine":
    # для эмбеддингов обычно логично использовать косинусную близость.
    #
    # random_state=42:
    # фиксируем случайность, чтобы результат между запусками
    # был воспроизводимым.
    umap_model = umap.UMAP(
        n_neighbors=15,
        min_dist=0.05,
        n_components=UMAP_DIM,
        metric="cosine",
        random_state=RANDOM_STATE,
    )

    X_umap = umap_model.fit_transform(X)

    print(f"UMAP done: {X_umap.shape}")

    # -------------------------
    # 3. HDBSCAN
    # -------------------------
    print("Running HDBSCAN...")

    # После UMAP обычно используем евклидову метрику
    # уже в сжатом пространстве.
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
    )

    labels = clusterer.fit_predict(X_umap)

    # -------------------------
    # 4. Статистика
    # -------------------------
    print("Analyzing clusters...")

    counter = Counter(labels)

    # Число реальных кластеров без шума
    n_clusters = len([c for c in counter if c != -1])

    # Число шумовых точек
    n_noise = counter.get(-1, 0)

    print(f"Clusters found: {n_clusters}")
    print(f"Noise points: {n_noise}")

    # Показываем 10 самых крупных групп
    print("\nTop clusters:")
    for cid, size in sorted(counter.items(), key=lambda x: -x[1])[:10]:
        print(f"Cluster {cid}: {size}")

    # -------------------------
    # 5. Сохранение
    # -------------------------
    print("Saving results...")

    # Для каждой книги сохраняем ее индекс и cluster_id
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for i, label in enumerate(labels):
            f.write(
                json.dumps(
                    {
                        "book_id": i,
                        "cluster_id": int(label),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # -------------------------
    # ФИНАЛ
    # -------------------------
    print("DONE")
    print(f"Saved: {OUT_PATH}")
    print(f"Time: {round(time.time() - t0, 2)} sec")


if __name__ == "__main__":
    main()