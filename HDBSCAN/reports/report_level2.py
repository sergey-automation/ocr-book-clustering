# ============================================
# STEP 2 REPORT
# ============================================
# Задача:
# Построить отчет по результатам level 2.
#
# Вход:
# - book_clusters_level2.jsonl
#
# Выход:
# - step2_report.txt
# - step2_report.csv
#
# Что анализируем:
# 1. Размер каждой пары (cluster_id, subcluster_id)
# 2. Сколько книг в каждом cluster_id всего
# 3. Сколько шума внутри каждого cluster_id
# 4. Какие ветки самые крупные
#
# Важно:
# - subcluster_id = 0 может означать:
#   а) малый cluster_id, который не делили
#   б) реальный подкластер 0 внутри большого cluster_id
# Поэтому анализируем только пару:
#   (cluster_id, subcluster_id)
# ============================================

import csv
import json
import time
from collections import Counter, defaultdict


# -------------------------
# ПУТИ
# -------------------------
IN_PATH = r"C:\VAST2_UPLOAD\out\book_vectors\book_clusters_level2.jsonl"
OUT_TXT = r"C:\VAST2_UPLOAD\out\book_vectors\step2_report.txt"
OUT_CSV = r"C:\VAST2_UPLOAD\out\book_vectors\step2_report.csv"


# -------------------------
# ПАРАМЕТРЫ ОТЧЕТА
# -------------------------
TOP_BRANCHES = 100
LARGE_BRANCH_THRESHOLD = 120


def load_rows(path):
    """
    Загружает JSONL с полями:
    - book_id
    - cluster_id
    - subcluster_id
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
                    raise ValueError(f"Line {line_no}: missing key: {key}")

            rows.append({
                "book_id": int(obj["book_id"]),
                "cluster_id": int(obj["cluster_id"]),
                "subcluster_id": int(obj["subcluster_id"]),
            })

    if not rows:
        raise ValueError("Input file is empty")

    return rows


def main():
    t0 = time.time()

    print("=== STEP 2 REPORT ===")
    print("Loading level2 file...")

    rows = load_rows(IN_PATH)
    n_books = len(rows)

    print(f"Rows loaded: {n_books}")

    # ---------------------------------
    # Проверка уникальности book_id
    # ---------------------------------
    book_ids = [r["book_id"] for r in rows]
    uniq_book_ids = set(book_ids)

    if len(uniq_book_ids) != n_books:
        raise ValueError("Duplicate book_id detected")

    # ---------------------------------
    # Агрегации
    # ---------------------------------
    # Размеры cluster_id
    cluster_counts = Counter()

    # Размеры точных веток (cluster_id, subcluster_id)
    branch_counts = Counter()

    # Шум внутри cluster_id
    cluster_noise_counts = Counter()

    # Список subcluster_id внутри каждого cluster_id
    cluster_to_subs = defaultdict(set)

    for r in rows:
        cid = r["cluster_id"]
        sid = r["subcluster_id"]

        cluster_counts[cid] += 1
        branch_counts[(cid, sid)] += 1
        cluster_to_subs[cid].add(sid)

        if sid == -1:
            cluster_noise_counts[cid] += 1

    # ---------------------------------
    # Сортировки
    # ---------------------------------
    clusters_sorted = sorted(cluster_counts.items(), key=lambda x: (-x[1], x[0]))
    branches_sorted = sorted(branch_counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))

    # ---------------------------------
    # Подготовка строк CSV
    # ---------------------------------
    csv_rows = []

    for (cid, sid), size in branches_sorted:
        cluster_total = cluster_counts[cid]
        noise_count = cluster_noise_counts.get(cid, 0)
        noise_pct = (100.0 * noise_count / cluster_total) if cluster_total else 0.0

        # Сколько "реальных" subcluster_id кроме -1
        real_subs = sorted(x for x in cluster_to_subs[cid] if x != -1)
        real_sub_count = len(real_subs)

        csv_rows.append({
            "cluster_id": cid,
            "subcluster_id": sid,
            "branch_size": size,
            "cluster_total": cluster_total,
            "branch_share_pct": round(100.0 * size / cluster_total, 2),
            "cluster_noise_count": noise_count,
            "cluster_noise_pct": round(noise_pct, 2),
            "cluster_unique_sub_ids_non_noise": real_sub_count,
        })

    # ---------------------------------
    # Сохранение CSV
    # ---------------------------------
    print("Writing CSV...")

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cluster_id",
                "subcluster_id",
                "branch_size",
                "cluster_total",
                "branch_share_pct",
                "cluster_noise_count",
                "cluster_noise_pct",
                "cluster_unique_sub_ids_non_noise",
            ],
            delimiter=";",
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    # ---------------------------------
    # Подготовка TXT-отчета
    # ---------------------------------
    print("Writing TXT report...")

    lines = []

    lines.append("=== STEP 2 REPORT ===")
    lines.append("")
    lines.append(f"Input file: {IN_PATH}")
    lines.append(f"Books total: {n_books}")
    lines.append(f"Unique cluster_id count: {len(cluster_counts)}")
    lines.append(f"Unique (cluster_id, subcluster_id) branches: {len(branch_counts)}")
    lines.append("")

    # 1. Крупнейшие cluster_id
    lines.append("=== TOP CLUSTERS ===")
    for cid, size in clusters_sorted[:50]:
        noise_count = cluster_noise_counts.get(cid, 0)
        noise_pct = (100.0 * noise_count / size) if size else 0.0
        sub_ids_sorted = sorted(cluster_to_subs[cid])
        lines.append(
            f"cluster_id={cid:>4} | size={size:>5} | "
            f"noise={noise_count:>4} ({noise_pct:>6.2f}%) | "
            f"sub_ids={sub_ids_sorted}"
        )

    lines.append("")

    # 2. Крупнейшие ветки
    lines.append("=== TOP BRANCHES (cluster_id, subcluster_id) ===")
    for (cid, sid), size in branches_sorted[:TOP_BRANCHES]:
        cluster_total = cluster_counts[cid]
        share_pct = (100.0 * size / cluster_total) if cluster_total else 0.0
        lines.append(
            f"cluster_id={cid:>4} | subcluster_id={sid:>4} | "
            f"branch_size={size:>5} | cluster_total={cluster_total:>5} | "
            f"share={share_pct:>6.2f}%"
        )

    lines.append("")

    # 3. Крупные ветки — кандидаты на level 3
    lines.append(f"=== CANDIDATES FOR LEVEL 3 (branch_size > {LARGE_BRANCH_THRESHOLD}) ===")
    found_large = False
    for (cid, sid), size in branches_sorted:
        if size > LARGE_BRANCH_THRESHOLD and sid != -1:
            cluster_total = cluster_counts[cid]
            share_pct = (100.0 * size / cluster_total) if cluster_total else 0.0
            lines.append(
                f"cluster_id={cid:>4} | subcluster_id={sid:>4} | "
                f"branch_size={size:>5} | cluster_total={cluster_total:>5} | "
                f"share={share_pct:>6.2f}%"
            )
            found_large = True

    if not found_large:
        lines.append("No branches above threshold.")

    lines.append("")

    # 4. Кластеры с большим шумом
    lines.append("=== CLUSTERS WITH HIGH NOISE ===")
    noise_rank = []
    for cid, total in cluster_counts.items():
        noise_count = cluster_noise_counts.get(cid, 0)
        noise_pct = (100.0 * noise_count / total) if total else 0.0
        if noise_count > 0:
            noise_rank.append((cid, total, noise_count, noise_pct))

    noise_rank.sort(key=lambda x: (-x[3], -x[2], x[0]))

    if noise_rank:
        for cid, total, noise_count, noise_pct in noise_rank[:50]:
            lines.append(
                f"cluster_id={cid:>4} | total={total:>5} | "
                f"noise={noise_count:>4} | noise_pct={noise_pct:>6.2f}%"
            )
    else:
        lines.append("No noise branches found.")

    lines.append("")

    # 5. Сколько книг имеют subcluster_id = 0
    zero_total = sum(1 for r in rows if r["subcluster_id"] == 0)
    zero_pct = 100.0 * zero_total / n_books if n_books else 0.0

    lines.append("=== GLOBAL ZERO SUBCLUSTER ===")
    lines.append(
        f"Books with subcluster_id = 0: {zero_total} ({zero_pct:.2f}%)"
    )
    lines.append(
        "Note: this includes both:"
    )
    lines.append("- clusters that were not split")
    lines.append("- real subcluster 0 inside split clusters")
    lines.append("")

    # 6. Краткий итог
    lines.append("=== SUMMARY ===")
    lines.append(f"Books total: {n_books}")
    lines.append(f"Clusters total: {len(cluster_counts)}")
    lines.append(f"Branches total: {len(branch_counts)}")
    lines.append(f"CSV saved: {OUT_CSV}")
    lines.append(f"TXT saved: {OUT_TXT}")
    lines.append(f"Time: {round(time.time() - t0, 2)} sec")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("DONE")
    print(f"TXT: {OUT_TXT}")
    print(f"CSV: {OUT_CSV}")
    print(f"Time: {round(time.time() - t0, 2)} sec")


if __name__ == "__main__":
    main()