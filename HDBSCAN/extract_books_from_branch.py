# ============================================
# STEP EXTRACT BOOKS FROM BRANCH
# ============================================
# Задача:
# Вытащить список книг из выбранной ветки дерева кластеризации.
#
# Поддерживаемые уровни:
# - level 1: cluster_id
# - level 2: cluster_id + subcluster_id
# - level 3: cluster_id + subcluster_id + subsubcluster_id
#
# Вход:
# - JSONL с кластерами
# - JSONL с метаданными книг
#
# Выход:
# - TXT со списком книг
# - CSV со списком книг
#
# Особенности:
# - если в метаданных нет book_id, используется номер строки
# - подходит для manifest_engineering_ru_rag.jsonl
# - полный путь к txt берется из поля txt_abs
#
# Примеры:
#
# Level 1:
# py step_extract_books_from_branch.py ^
#   --clusters C:\VAST2_UPLOAD\out\book_vectors\book_clusters_hdbscan.jsonl ^
#   --meta C:\VAST2_UPLOAD\manifest_engineering_ru_rag.jsonl ^
#   --cluster-id 0
#
# Level 2:
# py step_extract_books_from_branch.py ^
#   --clusters C:\VAST2_UPLOAD\out\book_vectors\book_clusters_level2.jsonl ^
#   --meta C:\VAST2_UPLOAD\manifest_engineering_ru_rag.jsonl ^
#   --cluster-id 0 ^
#   --subcluster-id 3
#
# Level 3:
# py step_extract_books_from_branch.py ^
#   --clusters C:\VAST2_UPLOAD\out\book_vectors\book_clusters_level3.jsonl ^
#   --meta C:\VAST2_UPLOAD\manifest_engineering_ru_rag.jsonl ^
#   --cluster-id 0 ^
#   --subcluster-id 0 ^
#   --subsubcluster-id 1
# ============================================

import os
import csv
import json
import argparse
from typing import Dict, List


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Bad JSON at line {line_no} in {path}: {e}")
            rows.append(obj)
    return rows


def detect_level(cluster_rows: List[dict]) -> int:
    """
    Определяем уровень файла кластеров.
    """
    has_sub = any("subcluster_id" in r for r in cluster_rows)
    has_subsub = any("subsubcluster_id" in r for r in cluster_rows)

    if has_subsub:
        return 3
    if has_sub:
        return 2
    return 1


def detect_title(row: dict) -> str:
    """
    Пытаемся вытащить максимально полезное название книги.
    """
    candidates = [
        "title",
        "book_title",
        "name",
        "file_name",
        "filename",
        "source_name",
        "doc_title",
        "basename",
    ]
    for key in candidates:
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    path_candidates = ["txt_abs", "txt_rel", "source_path", "path", "file_path"]
    for key in path_candidates:
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return os.path.basename(v.strip())

    return ""


def detect_doc_id(row: dict) -> str:
    candidates = ["doc_id", "book_doc_id", "docid", "id16", "doc_id16", "sid"]
    for key in candidates:
        v = row.get(key)
        if v is not None:
            return str(v)
    return ""


def detect_txt_abs(row: dict) -> str:
    v = row.get("txt_abs")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return ""


def detect_txt_rel(row: dict) -> str:
    v = row.get("txt_rel")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return ""


def build_meta_index(meta_rows: List[dict]) -> Dict[int, dict]:
    """
    Строим индекс book_id -> metadata row

    Поддержка двух режимов:
    1. В файле явно есть book_id
    2. book_id нет, тогда используем номер строки
    """
    has_book_id = any("book_id" in row for row in meta_rows)

    index = {}

    if has_book_id:
        for row in meta_rows:
            if "book_id" not in row:
                continue
            book_id = int(row["book_id"])
            index[book_id] = row
    else:
        for i, row in enumerate(meta_rows):
            index[i] = row

    return index


def filter_cluster_rows(
    cluster_rows: List[dict],
    cluster_id: int,
    subcluster_id=None,
    subsubcluster_id=None
) -> List[dict]:
    """
    Универсальный фильтр для level1 / level2 / level3.
    """
    out = []

    for row in cluster_rows:
        if int(row.get("cluster_id")) != cluster_id:
            continue

        if subcluster_id is not None:
            if "subcluster_id" not in row:
                continue
            if int(row.get("subcluster_id")) != subcluster_id:
                continue

        if subsubcluster_id is not None:
            if "subsubcluster_id" not in row:
                continue
            if int(row.get("subsubcluster_id")) != subsubcluster_id:
                continue

        out.append(row)

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Extract books from selected clustering branch"
    )

    parser.add_argument("--clusters", required=True, help="Path to cluster JSONL")
    parser.add_argument("--meta", required=True, help="Path to books metadata JSONL")
    parser.add_argument("--cluster-id", required=True, type=int, help="cluster_id")
    parser.add_argument("--subcluster-id", type=int, default=None, help="subcluster_id")
    parser.add_argument("--subsubcluster-id", type=int, default=None, help="subsubcluster_id")
    parser.add_argument(
        "--out-prefix",
        default=r"C:\VAST2_UPLOAD\out\book_vectors\extract_branch",
        help="Output prefix without extension"
    )
    parser.add_argument(
        "--sort-by",
        choices=["book_id", "title"],
        default="book_id",
        help="Sorting of output books"
    )

    args = parser.parse_args()

    print("=== EXTRACT BOOKS FROM BRANCH ===")
    print(f"Clusters file: {args.clusters}")
    print(f"Metadata file: {args.meta}")

    # -------------------------
    # 1. Загрузка
    # -------------------------
    print("Loading cluster rows...")
    cluster_rows = load_jsonl(args.clusters)
    print(f"Cluster rows loaded: {len(cluster_rows)}")

    print("Loading metadata rows...")
    meta_rows = load_jsonl(args.meta)
    print(f"Metadata rows loaded: {len(meta_rows)}")

    level = detect_level(cluster_rows)
    print(f"Detected cluster file level: {level}")

    # -------------------------
    # 2. Проверка аргументов
    # -------------------------
    if level == 1:
        if args.subcluster_id is not None or args.subsubcluster_id is not None:
            raise ValueError("Level 1 file supports only --cluster-id")
    elif level == 2:
        if args.subsubcluster_id is not None:
            raise ValueError("Level 2 file does not support --subsubcluster-id")

    # -------------------------
    # 3. Индекс метаданных
    # -------------------------
    meta_index = build_meta_index(meta_rows)

    # -------------------------
    # 4. Фильтр ветки
    # -------------------------
    selected = filter_cluster_rows(
        cluster_rows=cluster_rows,
        cluster_id=args.cluster_id,
        subcluster_id=args.subcluster_id,
        subsubcluster_id=args.subsubcluster_id
    )

    print(f"Selected books: {len(selected)}")

    if not selected:
        print("No books found for selected branch.")
        return

    # -------------------------
    # 5. Сбор строк вывода
    # -------------------------
    result_rows = []

    for row in selected:
        book_id = int(row["book_id"])
        meta = meta_index.get(book_id, {})

        out_row = {
            "book_id": book_id,
            "cluster_id": int(row.get("cluster_id", 0)),
            "subcluster_id": int(row["subcluster_id"]) if "subcluster_id" in row else "",
            "subsubcluster_id": int(row["subsubcluster_id"]) if "subsubcluster_id" in row else "",
            "title": detect_title(meta),
            "doc_id": detect_doc_id(meta),
            "txt_abs": detect_txt_abs(meta),
            "txt_rel": detect_txt_rel(meta),
        }

        result_rows.append(out_row)

    # -------------------------
    # 6. Сортировка
    # -------------------------
    if args.sort_by == "title":
        result_rows.sort(key=lambda x: (str(x["title"]).lower(), x["book_id"]))
    else:
        result_rows.sort(key=lambda x: x["book_id"])

    # -------------------------
    # 7. Имя файлов
    # -------------------------
    suffix_parts = [f"c{args.cluster_id}"]

    if args.subcluster_id is not None:
        suffix_parts.append(f"s{args.subcluster_id}")

    if args.subsubcluster_id is not None:
        suffix_parts.append(f"ss{args.subsubcluster_id}")

    suffix = "_".join(suffix_parts)

    out_txt = f"{args.out_prefix}_{suffix}.txt"
    out_csv = f"{args.out_prefix}_{suffix}.csv"

    out_dir = os.path.dirname(out_txt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # 8. CSV
    # -------------------------
    print("Writing CSV...")
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "book_id",
                "cluster_id",
                "subcluster_id",
                "subsubcluster_id",
                "title",
                "doc_id",
                "txt_abs",
                "txt_rel",
            ],
            delimiter=";",
        )
        writer.writeheader()
        writer.writerows(result_rows)

    # -------------------------
    # 9. TXT
    # -------------------------
    print("Writing TXT...")

    lines = []
    lines.append("=== EXTRACT BOOKS FROM BRANCH ===")
    lines.append("")
    lines.append(f"Clusters file: {args.clusters}")
    lines.append(f"Metadata file: {args.meta}")
    lines.append(f"Detected level: {level}")
    lines.append("")
    lines.append(f"cluster_id = {args.cluster_id}")

    if args.subcluster_id is not None:
        lines.append(f"subcluster_id = {args.subcluster_id}")

    if args.subsubcluster_id is not None:
        lines.append(f"subsubcluster_id = {args.subsubcluster_id}")

    lines.append(f"Books found: {len(result_rows)}")
    lines.append("")
    lines.append("=== BOOKS ===")

    for i, row in enumerate(result_rows, 1):
        lines.append(
            f"{i:>4}. "
            f"book_id={row['book_id']} | "
            f"title={row['title'] or '[NO TITLE]'} | "
            f"doc_id={row['doc_id'] or '[NO DOC_ID]'} | "
            f"txt_abs={row['txt_abs'] or '[NO TXT_ABS]'}"
        )

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # -------------------------
    # 10. Финал
    # -------------------------
    print("DONE")
    print(f"TXT: {out_txt}")
    print(f"CSV: {out_csv}")


if __name__ == "__main__":
    main()