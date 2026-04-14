import os
import csv
import json
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Bad JSON at line {line_no} in {path}: {e}")
    if not rows:
        raise ValueError(f"Input file is empty: {path}")
    return rows


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def normalize_path_sep(s: str) -> str:
    return s.replace("\\", os.sep).replace("/", os.sep)


def strip_known_extensions(filename: str) -> str:
    name = filename
    known = [
        ".pdf", ".djvu", ".tif", ".tiff", ".txt", ".doc", ".docx", ".rtf", ".fb2",
        ".epub", ".html", ".htm", ".md", ".csv", ".json", ".jsonl", ".xml",
    ]
    changed = True
    while changed:
        changed = False
        lower = name.lower()
        for ext in known:
            if lower.endswith(ext):
                name = name[:-len(ext)]
                changed = True
                break
    return name


def detect_doc_id(row: dict) -> str:
    for key in ("doc_id", "book_doc_id", "docid", "id16", "doc_id16", "sid"):
        v = row.get(key)
        if v is not None:
            return str(v)
    return ""


def detect_txt_rel(row: dict) -> str:
    v = row.get("txt_rel")
    return v.strip() if isinstance(v, str) else ""


def detect_txt_abs(row: dict, txt_root: str = "") -> str:
    v = row.get("txt_abs")
    if isinstance(v, str) and v.strip():
        return v.strip()
    rel = detect_txt_rel(row)
    if rel and txt_root:
        return os.path.normpath(os.path.join(txt_root, normalize_path_sep(rel)))
    return ""


def detect_title(row: dict) -> str:
    for key in ("title", "book_title", "name", "file_name", "filename", "source_name", "doc_title", "basename"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def filename_from_paths(row: dict, txt_root: str = "") -> str:
    txt_abs = detect_txt_abs(row, txt_root=txt_root)
    if txt_abs:
        return os.path.basename(txt_abs.rstrip("\\/"))
    txt_rel = detect_txt_rel(row)
    if txt_rel:
        return os.path.basename(txt_rel.rstrip("\\/"))
    title = detect_title(row)
    if title:
        return title
    doc_id = detect_doc_id(row)
    return doc_id


def load_level3_rows(path: str) -> List[dict]:
    rows = []
    for row in load_jsonl(path):
        for key in ("book_id", "cluster_id", "subcluster_id", "subsubcluster_id"):
            if key not in row:
                raise ValueError(f"Missing key {key} in level3 file")
        rows.append({
            "book_id": int(row["book_id"]),
            "cluster_id": int(row["cluster_id"]),
            "subcluster_id": int(row["subcluster_id"]),
            "subsubcluster_id": int(row["subsubcluster_id"]),
        })
    rows.sort(key=lambda x: x["book_id"])
    for i, row in enumerate(rows):
        if row["book_id"] != i:
            raise ValueError(f"Expected continuous book_id sequence. At position {i} got {row['book_id']}")
    return rows


def load_book_meta_by_book_id(path: str) -> Dict[int, dict]:
    rows = load_jsonl(path)
    by_book_id: Dict[int, dict] = {}
    for i, row in enumerate(rows):
        if "book_id" in row:
            bid = int(row["book_id"])
        elif "vector_row" in row:
            bid = int(row["vector_row"])
        else:
            bid = i
        by_book_id[bid] = row
    return by_book_id


def load_manifest_by_doc_id(path: str) -> Dict[str, dict]:
    rows = load_jsonl(path)
    by_doc_id: Dict[str, dict] = {}
    for row in rows:
        doc_id = detect_doc_id(row)
        if not doc_id:
            continue
        by_doc_id[doc_id] = row
    if not by_doc_id:
        raise ValueError("No doc_id values found in manifest")
    return by_doc_id


def level_quality(label: int) -> str:
    return "noise" if int(label) == -1 else "good"


def overall_book_quality(row: dict) -> str:
    labels = [row["cluster_id"], row["subcluster_id"], row["subsubcluster_id"]]
    return "noise_or_mixed" if any(x == -1 for x in labels) else "good_hit"


def branch_quality_from_counter(counter: Counter) -> Dict[str, Any]:
    total = sum(counter.values())
    noise = counter.get(-1, 0)
    noise_pct = round(100.0 * noise / total, 2) if total else 0.0
    if noise_pct >= 35.0:
        quality = "mostly_noise"
    elif noise_pct >= 15.0:
        quality = "mixed"
    else:
        quality = "good"
    return {
        "noise_count": noise,
        "noise_pct": noise_pct,
        "quality": quality,
    }


def group_rows(level3_rows: List[dict]) -> Tuple[Dict[int, List[dict]], Dict[Tuple[int, int], List[dict]], Dict[Tuple[int, int, int], List[dict]]]:
    lvl1: Dict[int, List[dict]] = defaultdict(list)
    lvl2: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    lvl3: Dict[Tuple[int, int, int], List[dict]] = defaultdict(list)
    for r in level3_rows:
        cid = r["cluster_id"]
        sid = r["subcluster_id"]
        ssid = r["subsubcluster_id"]
        lvl1[cid].append(r)
        lvl2[(cid, sid)].append(r)
        lvl3[(cid, sid, ssid)].append(r)
    return lvl1, lvl2, lvl3


def merge_meta(book_meta_row: dict, manifest_row: dict) -> dict:
    out = {}
    if manifest_row:
        out.update(manifest_row)
    if book_meta_row:
        out.update(book_meta_row)
    return out


def build_book_record(cluster_row: dict, book_meta_row: dict, manifest_row: dict, txt_root: str = "") -> dict:
    merged = merge_meta(book_meta_row, manifest_row)
    filename = filename_from_paths(merged, txt_root=txt_root)
    title = detect_title(manifest_row) or detect_title(book_meta_row) or strip_known_extensions(filename)
    authors = manifest_row.get("authors", "") if manifest_row else ""
    year = (manifest_row.get("year") if manifest_row else None) or book_meta_row.get("year") if book_meta_row else None
    return {
        "book_id": int(cluster_row["book_id"]),
        "doc_id": detect_doc_id(merged),
        "cluster_id": int(cluster_row["cluster_id"]),
        "subcluster_id": int(cluster_row["subcluster_id"]),
        "subsubcluster_id": int(cluster_row["subsubcluster_id"]),
        "title": title,
        "authors": authors,
        "year": year,
        "filename": filename,
        "filename_no_ext": strip_known_extensions(filename),
        "txt_rel": detect_txt_rel(merged),
        "txt_abs": detect_txt_abs(merged, txt_root=txt_root),
        "level1_quality": level_quality(cluster_row["cluster_id"]),
        "level2_quality": level_quality(cluster_row["subcluster_id"]),
        "level3_quality": level_quality(cluster_row["subsubcluster_id"]),
        "assignment_quality": overall_book_quality(cluster_row),
    }


def build_tree(level3_rows: List[dict], book_meta_by_book_id: Dict[int, dict], manifest_by_doc_id: Dict[str, dict], txt_root: str = "") -> dict:
    lvl1, lvl2, lvl3 = group_rows(level3_rows)

    root = {
        "tree_kind": "book_cluster_tree",
        "books_total": len(level3_rows),
        "clusters_total_level1": len(lvl1),
        "branches_total_level2": len(lvl2),
        "branches_total_level3": len(lvl3),
        "children": [],
    }

    for cid in sorted(lvl1.keys(), key=lambda x: (x == -1, x)):
        rows1 = lvl1[cid]
        sid_counter = Counter(r["subcluster_id"] for r in rows1)
        q1 = branch_quality_from_counter(sid_counter)
        node1 = {
            "level": 1,
            "cluster_id": int(cid),
            "node_key": f"c{cid}",
            "size": len(rows1),
            "quality_against_level2": q1,
            "children": [],
        }

        sids = sorted({r["subcluster_id"] for r in rows1}, key=lambda x: (x == -1, x))
        for sid in sids:
            rows2 = lvl2[(cid, sid)]
            ssid_counter = Counter(r["subsubcluster_id"] for r in rows2)
            q2 = branch_quality_from_counter(ssid_counter)
            node2 = {
                "level": 2,
                "cluster_id": int(cid),
                "subcluster_id": int(sid),
                "node_key": f"c{cid}_s{sid}",
                "size": len(rows2),
                "share_of_cluster_pct": round(100.0 * len(rows2) / len(rows1), 2) if rows1 else 0.0,
                "quality_against_level3": q2,
                "children": [],
            }

            ssids = sorted({r["subsubcluster_id"] for r in rows2}, key=lambda x: (x == -1, x))
            for ssid in ssids:
                rows3 = lvl3[(cid, sid, ssid)]
                books = []
                for r in rows3:
                    book_meta_row = book_meta_by_book_id.get(r["book_id"], {})
                    doc_id = detect_doc_id(book_meta_row)
                    manifest_row = manifest_by_doc_id.get(doc_id, {}) if doc_id else {}
                    books.append(build_book_record(r, book_meta_row, manifest_row, txt_root=txt_root))
                node3 = {
                    "level": 3,
                    "cluster_id": int(cid),
                    "subcluster_id": int(sid),
                    "subsubcluster_id": int(ssid),
                    "node_key": f"c{cid}_s{sid}_ss{ssid}",
                    "size": len(rows3),
                    "share_of_level2_pct": round(100.0 * len(rows3) / len(rows2), 2) if rows2 else 0.0,
                    "node_quality": "noise" if ssid == -1 else "good",
                    "books": books,
                }
                node2["children"].append(node3)
            node1["children"].append(node2)
        root["children"].append(node1)

    return root


def render_tree_names_only(tree: dict) -> str:
    lines: List[str] = []
    lines.append("TREE: books only (names without path and extension)")
    lines.append("")
    for n1 in tree["children"]:
        lines.append(f"cluster {n1['cluster_id']} | size={n1['size']}")
        for n2 in n1["children"]:
            lines.append(f"  subcluster {n2['subcluster_id']} | size={n2['size']}")
            for n3 in n2["children"]:
                lines.append(f"    subsubcluster {n3['subsubcluster_id']} | size={n3['size']}")
                for book in n3["books"]:
                    lines.append(f"      - {book['filename_no_ext']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_llm_tree(tree: dict) -> dict:
    out = {
        "tree_kind": "cluster_tree_for_llm_branch_naming",
        "instructions": (
            "Propose short, practical names for branches using only listed book names. "
            "Do not use paths or file extensions. Return names for level 1, level 2 and level 3 branches where possible."
        ),
        "children": [],
    }
    for n1 in tree["children"]:
        x1 = {
            "cluster_id": n1["cluster_id"],
            "size": n1["size"],
            "books_preview": [],
            "children": [],
        }
        preview1 = []
        for n2 in n1["children"]:
            for n3 in n2["children"]:
                for b in n3["books"]:
                    preview1.append(b["filename_no_ext"])
        x1["books_preview"] = preview1[:40]

        for n2 in n1["children"]:
            x2 = {
                "subcluster_id": n2["subcluster_id"],
                "size": n2["size"],
                "books_preview": [],
                "children": [],
            }
            preview2 = []
            for n3 in n2["children"]:
                for b in n3["books"]:
                    preview2.append(b["filename_no_ext"])
            x2["books_preview"] = preview2[:40]

            for n3 in n2["children"]:
                x3 = {
                    "subsubcluster_id": n3["subsubcluster_id"],
                    "size": n3["size"],
                    "book_names": [b["filename_no_ext"] for b in n3["books"]],
                }
                x2["children"].append(x3)
            x1["children"].append(x2)
        out["children"].append(x1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cluster tree exports from level3 + book meta + manifest")
    parser.add_argument("--level3", required=True, help="Path to book_clusters_level3.jsonl")
    parser.add_argument("--book-meta", required=True, help="Path to book_vectors_chunk_centered.jsonl or book_vectors_full.jsonl")
    parser.add_argument("--manifest", required=True, help="Path to manifest_engineering_ru_rag.jsonl")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--txt-root", default="", help="Optional root to convert txt_rel into absolute path when txt_abs is absent")
    parser.add_argument("--base-name", default="book_cluster_tree", help="Base file name prefix")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading level3...")
    level3_rows = load_level3_rows(args.level3)
    print("Loading book meta...")
    book_meta_by_book_id = load_book_meta_by_book_id(args.book_meta)
    print("Loading manifest...")
    manifest_by_doc_id = load_manifest_by_doc_id(args.manifest)

    missing_book_meta = [r["book_id"] for r in level3_rows if r["book_id"] not in book_meta_by_book_id]
    if missing_book_meta:
        raise ValueError(f"Book meta missing for some book_id values, example: {missing_book_meta[:10]}")

    missing_manifest = []
    for r in level3_rows:
        book_meta_row = book_meta_by_book_id[r["book_id"]]
        doc_id = detect_doc_id(book_meta_row)
        if doc_id and doc_id not in manifest_by_doc_id:
            missing_manifest.append(doc_id)
    if missing_manifest:
        print(f"WARNING: manifest rows not found for some doc_id values, example: {missing_manifest[:10]}")

    print("Building tree...")
    tree = build_tree(level3_rows, book_meta_by_book_id, manifest_by_doc_id, txt_root=args.txt_root)
    tree["source_level3_file"] = args.level3
    tree["source_book_meta_file"] = args.book_meta
    tree["source_manifest_file"] = args.manifest
    tree["txt_root_used"] = args.txt_root

    out_tree_json = os.path.join(args.out_dir, f"{args.base_name}.json")
    out_tree_txt = os.path.join(args.out_dir, f"{args.base_name}_titles_only.txt")
    out_llm_json = os.path.join(args.out_dir, f"{args.base_name}_for_llm.json")
    out_summary_json = os.path.join(args.out_dir, f"{args.base_name}_summary.json")

    write_json(out_tree_json, tree)
    write_text(out_tree_txt, render_tree_names_only(tree))
    write_json(out_llm_json, render_llm_tree(tree))

    summary = {
        "level3_input": args.level3,
        "book_meta_input": args.book_meta,
        "manifest_input": args.manifest,
        "out_tree_json": out_tree_json,
        "out_titles_txt": out_tree_txt,
        "out_llm_json": out_llm_json,
        "books_total": len(level3_rows),
        "clusters_total_level1": tree["clusters_total_level1"],
        "branches_total_level2": tree["branches_total_level2"],
        "branches_total_level3": tree["branches_total_level3"],
    }
    write_json(out_summary_json, summary)

    print("DONE")
    print(f"JSON tree: {out_tree_json}")
    print(f"TXT tree:  {out_tree_txt}")
    print(f"LLM tree:  {out_llm_json}")
    print(f"Summary:   {out_summary_json}")


if __name__ == "__main__":
    main()
