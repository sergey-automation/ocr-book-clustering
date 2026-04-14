import os
import json
import argparse
from collections import defaultdict

import numpy as np
import umap


DEFAULT_VEC_PATH = r"C:\VAST2_UPLOAD\out\book_vectors\book_vectors_chunk_centered.npy"
DEFAULT_META_PATH = r"C:\VAST2_UPLOAD\out\book_vectors\book_vectors_full.jsonl"
DEFAULT_CLUSTERS_PATH = r"C:\VAST2_UPLOAD\out\book_vectors\book_clusters_level3.jsonl"
DEFAULT_OUT_PATH = r"C:\VAST2_UPLOAD\out\book_vectors\PLOT\umap_clusters_stats.html"


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Bad JSON at line {line_no} in {path}: {e}")
    return rows


def safe_get(row: dict, *keys, default=""):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def escape_html(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def main():
    parser = argparse.ArgumentParser(description="Interactive UMAP by cluster with compact stats table")
    parser.add_argument("--vec", default=DEFAULT_VEC_PATH, help="Path to .npy book vectors")
    parser.add_argument("--meta", default=DEFAULT_META_PATH, help="Path to .jsonl metadata")
    parser.add_argument("--clusters", default=DEFAULT_CLUSTERS_PATH, help="Path to cluster jsonl")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="Output .html path")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--min-dist", type=float, default=0.05, help="UMAP min_dist")
    parser.add_argument("--metric", default="cosine", help="UMAP metric")
    parser.add_argument("--random-state", type=int, default=42, help="UMAP random_state")
    parser.add_argument("--point-size", type=int, default=4, help="Scatter point size")
    parser.add_argument("--max-title-len", type=int, default=120, help="Trim title in hover")
    parser.add_argument("--hide-noise-default", type=int, choices=[0, 1], default=1, help="Hide cluster -1 initially")
    args = parser.parse_args()

    try:
        import plotly.graph_objects as go
        from plotly.offline import plot as plot_offline
    except Exception as e:
        raise RuntimeError("plotly is not installed. Install it with: py -m pip install plotly") from e

    print("Loading vectors...")
    if not os.path.exists(args.vec):
        raise FileNotFoundError(f"Vectors file not found: {args.vec}")
    X = np.load(args.vec).astype(np.float32, copy=False)
    n_books, _ = X.shape

    print("Loading meta...")
    if not os.path.exists(args.meta):
        raise FileNotFoundError(f"Meta file not found: {args.meta}")
    meta_rows = load_jsonl(args.meta)
    if len(meta_rows) != n_books:
        raise RuntimeError(f"Mismatch: vectors rows={n_books}, meta rows={len(meta_rows)}")

    print("Loading clusters...")
    if not os.path.exists(args.clusters):
        raise FileNotFoundError(f"Clusters file not found: {args.clusters}")
    cluster_rows = load_jsonl(args.clusters)
    if len(cluster_rows) != n_books:
        raise RuntimeError(f"Mismatch: vectors rows={n_books}, cluster rows={len(cluster_rows)}")

    print("UMAP reduction...")
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=2,
        metric=args.metric,
        random_state=args.random_state,
    )
    X2 = reducer.fit_transform(X)

    print("Preparing cluster stats...")
    ids_by_cluster = defaultdict(list)
    sub_ids_by_cluster = defaultdict(list)
    for i, row in enumerate(cluster_rows):
        cid = safe_int(row.get("cluster_id"), -999999)
        ids_by_cluster[cid].append(i)
        sid = row.get("subcluster_id")
        if sid is not None:
            sub_ids_by_cluster[cid].append(safe_int(sid, 0))

    stats_rows = []
    dist_to_center = np.zeros(n_books, dtype=np.float32)
    rank_from_far = np.zeros(n_books, dtype=np.int32)

    for cid, ids in ids_by_cluster.items():
        Xc = X[ids]
        centroid = Xc.mean(axis=0)
        dists = np.linalg.norm(Xc - centroid, axis=1)
        for local_idx, book_idx in enumerate(ids):
            dist_to_center[book_idx] = float(dists[local_idx])
        order_far = np.argsort(-dists)
        for rank_pos, local_idx in enumerate(order_far, 1):
            rank_from_far[ids[int(local_idx)]] = rank_pos

        n = len(ids)
        pct = 100.0 * n / max(n_books, 1)
        if cid in sub_ids_by_cluster and len(sub_ids_by_cluster[cid]) > 0:
            sids = sub_ids_by_cluster[cid]
            sg = len({x for x in sids if x != -1})
            nz = 100.0 * sum(1 for x in sids if x == -1) / max(len(sids), 1)
        else:
            sg = 0
            nz = 0.0
        spr = float(dists.mean()) if len(dists) else 0.0
        mx = float(dists.max()) if len(dists) else 0.0
        stats_rows.append({"cid": int(cid), "n": int(n), "pct": pct, "sg": int(sg), "nz": nz, "spr": spr, "max": mx})

    print("Preparing traces...")
    xs_by_cluster = {}
    ys_by_cluster = {}
    text_by_cluster = {}

    def make_hover(i, meta, cluster_row):
        title = safe_get(meta, "title", "book_title", "name", "file_name", "filename", "source_name", "doc_title", "basename", default="")
        if not title:
            txt_rel = str(safe_get(meta, "txt_rel", default=""))
            title = os.path.basename(txt_rel) if txt_rel else f"book_{i}"
        title = str(title)
        if len(title) > args.max_title_len:
            title = title[: args.max_title_len - 3] + "..."
        parts = [
            f"book_id={i}",
            f"title={title}",
            f"doc_id={safe_get(meta, 'doc_id', 'sid', default='')}",
            f"year={safe_get(meta, 'year', default='')}",
            f"txt_rel={safe_get(meta, 'txt_rel', default='')}",
            f"vector_row={safe_get(meta, 'vector_row', default='')}",
            f"cluster_id={safe_get(cluster_row, 'cluster_id', default='')}",
        ]
        if "subcluster_id" in cluster_row:
            parts.append(f"subcluster_id={safe_get(cluster_row, 'subcluster_id', default='')}")
        if "subsubcluster_id" in cluster_row:
            parts.append(f"subsubcluster_id={safe_get(cluster_row, 'subsubcluster_id', default='')}")
        parts.append(f"dist_to_cluster_center={dist_to_center[i]:.6f}")
        parts.append(f"rank_by_distance_in_cluster={int(rank_from_far[i])}")
        return "<br>".join(parts)

    for i, meta in enumerate(meta_rows):
        cluster_row = cluster_rows[i]
        cid = safe_int(cluster_row.get("cluster_id"), -999999)
        cluster_key = str(cid)
        xs_by_cluster.setdefault(cluster_key, []).append(float(X2[i, 0]))
        ys_by_cluster.setdefault(cluster_key, []).append(float(X2[i, 1]))
        text_by_cluster.setdefault(cluster_key, []).append(make_hover(i, meta, cluster_row))

    fig = go.Figure()
    for cluster_key in sorted(xs_by_cluster.keys(), key=lambda x: (x == "-1", int(x) if x.lstrip("-").isdigit() else x)):
        visible = "legendonly" if (cluster_key == "-1" and args.hide_noise_default == 1) else True
        fig.add_trace(
            go.Scattergl(
                x=xs_by_cluster[cluster_key],
                y=ys_by_cluster[cluster_key],
                mode="markers",
                name=f"cluster {cluster_key}",
                text=text_by_cluster[cluster_key],
                hovertemplate="%{text}<extra></extra>",
                marker={"size": args.point_size},
                visible=visible,
            )
        )

    fig.update_layout(
        title="Books UMAP Interactive",
        width=1400,
        height=950,
        template="plotly_white",
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        margin=dict(l=60, r=30, t=80, b=40),
    )

    plot_div = plot_offline(fig, output_type="div", include_plotlyjs="cdn")

    stats_rows_sorted = sorted(stats_rows, key=lambda r: r["cid"])
    header_defs = [
        ("cid", "cluster_id"),
        ("n", "число книг в кластере"),
        ("pct", "доля от корпуса, %"),
        ("sg", "число подгрупп"),
        ("nz", "доля шума, %"),
        ("spr", "средняя дистанция до центра"),
        ("max", "максимальная дистанция до центра"),
    ]
    table_rows_html = []
    for r in stats_rows_sorted:
        table_rows_html.append(
            "<tr>"
            f"<td data-value=\"{r['cid']}\">{r['cid']}</td>"
            f"<td data-value=\"{r['n']}\">{r['n']}</td>"
            f"<td data-value=\"{r['pct']:.6f}\">{r['pct']:.2f}</td>"
            f"<td data-value=\"{r['sg']}\">{r['sg']}</td>"
            f"<td data-value=\"{r['nz']:.6f}\">{r['nz']:.2f}</td>"
            f"<td data-value=\"{r['spr']:.9f}\">{r['spr']:.4f}</td>"
            f"<td data-value=\"{r['max']:.9f}\">{r['max']:.4f}</td>"
            "</tr>"
        )
    headers_html = "".join(
        f'<th title="{escape_html(title)}" data-col="{idx}">{escape_html(short)}</th>'
        for idx, (short, title) in enumerate(header_defs)
    )

    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>UMAP Clusters Stats</title>
<style>
body {{ font-family: Arial, Helvetica, sans-serif; margin: 0; padding: 0; color: #222; }}
.wrap {{ padding: 10px 14px 18px 14px; }}
.note {{ font-size: 12px; color: #555; margin: 0 0 8px 0; }}
.table-wrap {{ margin-top: 12px; border-top: 1px solid #ddd; padding-top: 10px; }}
table {{ border-collapse: collapse; width: 100%; table-layout: fixed; font-size: 12px; }}
thead th {{ position: sticky; top: 0; background: #f7f7f7; border: 1px solid #ddd; padding: 6px 4px; text-align: right; cursor: pointer; user-select: none; }}
tbody td {{ border: 1px solid #e3e3e3; padding: 4px 4px; text-align: right; }}
tbody tr:nth-child(even) {{ background: #fafafa; }}
.small {{ font-size: 11px; color: #666; }}
</style>
</head>
<body>
<div class=\"wrap\">
  <p class=\"note\">Legend keeps standard visibility toggle. Cluster -1 starts hidden by default.</p>
  {plot_div}
  <div class=\"table-wrap\">
    <table id=\"clusterTable\">
      <thead><tr>{headers_html}</tr></thead>
      <tbody>{''.join(table_rows_html)}</tbody>
    </table>
    <div class=\"small\" style=\"margin-top:6px;\">Click a column header to sort.</div>
  </div>
</div>
<script>
(function() {{
  const table = document.getElementById('clusterTable');
  const tbody = table.querySelector('tbody');
  const headers = table.querySelectorAll('thead th');
  let sortState = {{ col: 0, asc: true }};
  function sortTable(colIndex) {{
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const asc = (sortState.col === colIndex) ? !sortState.asc : true;
    sortState = {{ col: colIndex, asc: asc }};
    rows.sort((a, b) => {{
      const av = a.children[colIndex].getAttribute('data-value');
      const bv = b.children[colIndex].getAttribute('data-value');
      const an = Number(av); const bn = Number(bv);
      let cmp;
      if (!Number.isNaN(an) && !Number.isNaN(bn)) cmp = an - bn;
      else cmp = String(av).localeCompare(String(bv));
      return asc ? cmp : -cmp;
    }});
    for (const row of rows) tbody.appendChild(row);
  }}
  headers.forEach((th, idx) => th.addEventListener('click', () => sortTable(idx)));
}})();
</script>
</body>
</html>"""

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print("Saving HTML...")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print("DONE")
    print(f"Written: {args.out}")


if __name__ == "__main__":
    main()
