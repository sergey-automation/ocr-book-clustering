# ============================================
# STEP 3 REPORT
# ============================================
# Задача:
# Построить отчет по результатам level 3.
#
# Вход:
# - book_clusters_level3.jsonl
#
# Выход:
# - step3_report.txt
# - step3_report.csv
#
# Что анализируем:
# 1. Размер cluster_id
# 2. Размер веток level 2: (cluster_id, subcluster_id)
# 3. Размер веток level 3: (cluster_id, subcluster_id, subsubcluster_id)
# 4. Шум level 3 внутри каждой ветки level 2
#
# Важно:
# - subcluster_id = 0 может быть:
#   а) неделившийся cluster_id уровня 1
#   б) реальная ветка 0 внутри split cluster
#
# - subsubcluster_id = 0 может быть:
#   а) ветка уровня 2 не делилась дальше
#   б) реальная ветка 0 после level 3
#
# Поэтому анализируем не отдельные id, а полные ключи:
#   (cluster_id, subcluster_id)
#   (cluster_id, subcluster_id, subsubcluster_id)
# ============================================

import csv
import json
import time
from collections import Counter, defaultdict


# -------------------------
# ПУТИ
# -------------------------
IN_PATH = r"C:\VAST2_UPLOAD\out\book_vectors\book_clusters_level3.jsonl"
OUT_TXT = r"C:\VAST2_UPLOAD\out\book_vectors\step3_report.txt"
OUT_CSV = r"C:\VAST2_UPLOAD\out\book_vectors\step3_report.csv"


# -------------------------
# ПАРАМЕТРЫ ОТЧЕТА
# -------------------------
TOP_CLUSTERS = 50
TOP_LEVEL2_BRANCHES = 100
TOP_LEVEL3_BRANCHES = 150

# Кандидаты на возможный следующий уровень
NEXT_LEVEL_THRESHOLD = 120


def load_rows(path):
    """
    Загружает JSONL формата:
    {
      "book_id": int,
      "cluster_id": int,
      "subcluster_id": int,
      "subsubcluster_id": int
    }
    """
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            for key in ("book_id", "cluster_id", "subcluster_id", "subsubcluster_id"):
                if key not in obj:
                    raise ValueError(f"Line {line_no}: missing key: {key}")

            rows.append({
                "book_id": int(obj["book_id"]),
                "cluster_id": int(obj["cluster_id"]),
                "subcluster_id": int(obj["subcluster_id"]),
                "subsubcluster_id": int(obj["subsubcluster_id"]),
            })

    if not rows:
        raise ValueError("Input file is empty")

    return rows


def is_level2_branch_really_split(subsub_ids):
    """
    Ветка level 2 считается реально разделенной на level 3,
    если внутри нее есть хотя бы 2 не-шумовых subsubcluster_id.

    Примеры:
    [0] -> не делилась
    [-1, 0, 1, 2] -> делилась
    [-1, 1] -> тоже считаем делившейся
    """
    non_noise = [x for x in subsub_ids if x != -1]
    return len(non_noise) >= 2


def main():
    t0 = time.time()

    print("=== STEP 3 REPORT ===")
    print("Loading level3 file...")

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
    level1_counts = Counter()   # cluster_id
    level2_counts = Counter()   # (cluster_id, subcluster_id)
    level3_counts = Counter()   # (cluster_id, subcluster_id, subsubcluster_id)

    # Шум level 3 внутри ветки level 2
    level2_noise_counts = Counter()  # key = (cluster_id, subcluster_id)

    # Какие subcluster_id есть внутри cluster_id
    cluster_to_subs = defaultdict(set)

    # Какие subsubcluster_id есть внутри ветки level 2
    level2_to_subsubs = defaultdict(set)

    for r in rows:
        cid = r["cluster_id"]
        sid = r["subcluster_id"]
        ssid = r["subsubcluster_id"]

        level1_counts[cid] += 1
        level2_counts[(cid, sid)] += 1
        level3_counts[(cid, sid, ssid)] += 1

        cluster_to_subs[cid].add(sid)
        level2_to_subsubs[(cid, sid)].add(ssid)

        if ssid == -1:
            level2_noise_counts[(cid, sid)] += 1

    # ---------------------------------
    # Сортировки
    # ---------------------------------
    level1_sorted = sorted(level1_counts.items(), key=lambda x: (-x[1], x[0]))
    level2_sorted = sorted(level2_counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
    level3_sorted = sorted(level3_counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1], x[0][2]))

    # ---------------------------------
    # CSV
    # ---------------------------------
    csv_rows = []

    for (cid, sid, ssid), size in level3_sorted:
        level1_total = level1_counts[cid]
        level2_total = level2_counts[(cid, sid)]

        level2_noise = level2_noise_counts.get((cid, sid), 0)
        level2_noise_pct = (100.0 * level2_noise / level2_total) if level2_total else 0.0

        real_subsubs = sorted(x for x in level2_to_subsubs[(cid, sid)] if x != -1)
        real_subsub_count = len(real_subsubs)

        csv_rows.append({
            "cluster_id": cid,
            "subcluster_id": sid,
            "subsubcluster_id": ssid,
            "level3_branch_size": size,
            "level2_branch_total": level2_total,
            "level1_cluster_total": level1_total,
            "share_of_level2_pct": round(100.0 * size / level2_total, 2) if level2_total else 0.0,
            "share_of_level1_pct": round(100.0 * size / level1_total, 2) if level1_total else 0.0,
            "level2_noise_count": level2_noise,
            "level2_noise_pct": round(level2_noise_pct, 2),
            "level2_unique_subsub_ids_non_noise": real_subsub_count,
        })

    print("Writing CSV...")

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cluster_id",
                "subcluster_id",
                "subsubcluster_id",
                "level3_branch_size",
                "level2_branch_total",
                "level1_cluster_total",
                "share_of_level2_pct",
                "share_of_level1_pct",
                "level2_noise_count",
                "level2_noise_pct",
                "level2_unique_subsub_ids_non_noise",
            ],
            delimiter=";",
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    # ---------------------------------
    # TXT
    # ---------------------------------
    print("Writing TXT report...")

    lines = []

    lines.append("=== STEP 3 REPORT ===")
    lines.append("")
    lines.append(f"Input file: {IN_PATH}")
    lines.append(f"Books total: {n_books}")
    lines.append(f"Unique cluster_id count: {len(level1_counts)}")
    lines.append(f"Unique level2 branches: {len(level2_counts)}")
    lines.append(f"Unique level3 branches: {len(level3_counts)}")
    lines.append("")

    # 1. Топ level 1
    lines.append("=== TOP LEVEL 1 CLUSTERS ===")
    for cid, size in level1_sorted[:TOP_CLUSTERS]:
        sub_ids_sorted = sorted(cluster_to_subs[cid])
        lines.append(
            f"cluster_id={cid:>4} | size={size:>5} | sub_ids={sub_ids_sorted}"
        )

    lines.append("")

    # 2. Топ level 2 веток
    lines.append("=== TOP LEVEL 2 BRANCHES (cluster_id, subcluster_id) ===")
    for (cid, sid), size in level2_sorted[:TOP_LEVEL2_BRANCHES]:
        level1_total = level1_counts[cid]
        share_pct = (100.0 * size / level1_total) if level1_total else 0.0

        noise_count = level2_noise_counts.get((cid, sid), 0)
        noise_pct = (100.0 * noise_count / size) if size else 0.0

        subsubs_sorted = sorted(level2_to_subsubs[(cid, sid)])

        lines.append(
            f"cluster_id={cid:>4} | subcluster_id={sid:>4} | "
            f"branch_size={size:>5} | cluster_total={level1_total:>5} | "
            f"share={share_pct:>6.2f}% | "
            f"level3_noise={noise_count:>4} ({noise_pct:>6.2f}%) | "
            f"subsub_ids={subsubs_sorted}"
        )

    lines.append("")

    # 3. Топ level 3 веток
    lines.append("=== TOP LEVEL 3 BRANCHES (cluster_id, subcluster_id, subsubcluster_id) ===")
    for (cid, sid, ssid), size in level3_sorted[:TOP_LEVEL3_BRANCHES]:
        level2_total = level2_counts[(cid, sid)]
        level1_total = level1_counts[cid]

        share_l2 = (100.0 * size / level2_total) if level2_total else 0.0
        share_l1 = (100.0 * size / level1_total) if level1_total else 0.0

        lines.append(
            f"cluster_id={cid:>4} | subcluster_id={sid:>4} | subsubcluster_id={ssid:>4} | "
            f"size={size:>5} | level2_total={level2_total:>5} | level1_total={level1_total:>5} | "
            f"share_of_level2={share_l2:>6.2f}% | share_of_level1={share_l1:>6.2f}%"
        )

    lines.append("")

    # 4. Ветки level 2, реально разделенные на level 3
    lines.append("=== LEVEL 2 BRANCHES REALLY SPLIT ON LEVEL 3 ===")
    really_split_found = False
    for (cid, sid), size in level2_sorted:
        if is_level2_branch_really_split(level2_to_subsubs[(cid, sid)]):
            noise_count = level2_noise_counts.get((cid, sid), 0)
            noise_pct = (100.0 * noise_count / size) if size else 0.0
            subsubs_sorted = sorted(level2_to_subsubs[(cid, sid)])
            lines.append(
                f"cluster_id={cid:>4} | subcluster_id={sid:>4} | "
                f"size={size:>5} | level3_noise={noise_count:>4} ({noise_pct:>6.2f}%) | "
                f"subsub_ids={subsubs_sorted}"
            )
            really_split_found = True

    if not really_split_found:
        lines.append("No branches were really split on level 3.")

    lines.append("")

    # 5. Кандидаты на следующий уровень
    lines.append(f"=== CANDIDATES FOR NEXT LEVEL (level3_branch_size > {NEXT_LEVEL_THRESHOLD}) ===")
    found_next = False
    for (cid, sid, ssid), size in level3_sorted:
        if ssid == -1:
            continue
        if size > NEXT_LEVEL_THRESHOLD:
            level2_total = level2_counts[(cid, sid)]
            level1_total = level1_counts[cid]
            share_l2 = (100.0 * size / level2_total) if level2_total else 0.0
            share_l1 = (100.0 * size / level1_total) if level1_total else 0.0
            lines.append(
                f"cluster_id={cid:>4} | subcluster_id={sid:>4} | subsubcluster_id={ssid:>4} | "
                f"size={size:>5} | level2_total={level2_total:>5} | level1_total={level1_total:>5} | "
                f"share_of_level2={share_l2:>6.2f}% | share_of_level1={share_l1:>6.2f}%"
            )
            found_next = True

    if not found_next:
        lines.append("No valid candidates above threshold.")

    lines.append("")

    # 6. Ветки level 2 с большим шумом level 3
    lines.append("=== LEVEL 2 BRANCHES WITH HIGH LEVEL 3 NOISE ===")
    noise_rank = []
    for (cid, sid), total in level2_counts.items():
        noise_count = level2_noise_counts.get((cid, sid), 0)
        noise_pct = (100.0 * noise_count / total) if total else 0.0
        if noise_count > 0:
            noise_rank.append((cid, sid, total, noise_count, noise_pct))

    noise_rank.sort(key=lambda x: (-x[4], -x[3], x[0], x[1]))

    if noise_rank:
        for cid, sid, total, noise_count, noise_pct in noise_rank[:100]:
            lines.append(
                f"cluster_id={cid:>4} | subcluster_id={sid:>4} | "
                f"total={total:>5} | noise={noise_count:>4} | noise_pct={noise_pct:>6.2f}%"
            )
    else:
        lines.append("No level 3 noise found.")

    lines.append("")

    # 7. Глобальные нули
    zero_total = sum(1 for r in rows if r["subsubcluster_id"] == 0)
    zero_pct = 100.0 * zero_total / n_books if n_books else 0.0

    lines.append("=== GLOBAL ZERO SUBSUBCLUSTER ===")
    lines.append(f"Books with subsubcluster_id = 0: {zero_total} ({zero_pct:.2f}%)")
    lines.append("Note: this includes both:")
    lines.append("- branches that were not split on level 3")
    lines.append("- real subsubcluster 0 inside split branches")
    lines.append("")

    # 8. Итог
    lines.append("=== SUMMARY ===")
    lines.append(f"Books total: {n_books}")
    lines.append(f"Clusters total: {len(level1_counts)}")
    lines.append(f"Level2 branches total: {len(level2_counts)}")
    lines.append(f"Level3 branches total: {len(level3_counts)}")
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