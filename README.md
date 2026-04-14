# OCR Book Clustering

Tool for reorganizing large unstructured OCR book collections into clean, semantic libraries using embeddings and hierarchical clustering.

---

## Problem

Large OCR book collections are usually:

- unstructured  
- poorly named  
- mixed across topics  
- noisy (indexes, tables, garbage OCR)

Building RAG or search directly on this data leads to:

- low relevance  
- high noise  
- wasted compute  

---

## Solution

This project provides a **pre-RAG pipeline** that:

1. Converts page-level embeddings into **book-level embeddings**
2. Cleans and filters noisy OCR content
3. Clusters books into **semantic groups**
4. Builds a **hierarchical topic tree**
5. Allows extracting books into **clean thematic folders**
6. Provides interactive visualization (UMAP)

---

## Pipeline
```
OCR books
↓
page embeddings
↓
book embeddings (filtered + weighted)
↓
hierarchical clustering (UMAP + HDBSCAN)
↓
cluster tree / branches
↓
clean thematic library
↓
RAG (on selected subsets only)
```


---

## Features

### 1. Book-level embeddings (not page-level)

- filters low-quality OCR chunks  
- removes boilerplate (index, contents, etc.)  
- weighted aggregation of best chunks  
- fallback for bad data  

---

### 2. Hierarchical clustering

- Level 1 → broad topics  
- Level 2 → subtopics (only large clusters)  
- Level 3 → fine-grained branches  

---

### 3. Practical output

- extract books by cluster  
- build folder structure  
- generate cluster tree  

---

### 4. Interactive visualization

- UMAP projection  
- cluster separation visible  
- noise detection  
- cluster statistics  

---

## Project structure
```
ocr-book-clustering/

data/
chunks.jsonl
manifest.jsonl

out/
parts/
book_vectors/
PLOT/
EXPORT/

PREP_BOOKS/
HDBSCAN/
PLOT/
EXPORT/
```

---

## Installation

```bash
pip install -r requirements.txt
```
Usage
Step 1 — Build book embeddings
```
PREP_BOOKS/run_prep_books.bat
```
Step 2 — Cluster books
```
HDBSCAN/run_hdbscan_pipeline.bat
```
Output:
```
book_clusters_level1.jsonl
book_clusters_level2.jsonl
book_clusters_level3.jsonl
```

Step 3 — Visualize
```
PLOT/run_plot_umap_clusters_stats.bat
```
Output:
```
out/PLOT/umap_clusters_stats.html
```
Open in browser.

Step 4 — Extract books from cluster

Example:
```
python HDBSCAN/extract_books_from_branch.py ^
  --clusters out/book_vectors/book_clusters_level3.jsonl ^
  --meta out/book_vectors/book_vectors_full.jsonl ^
  --cluster-id 0 ^
  --subcluster-id 2
  ```
 Output:

JSON tree
TXT structure
LLM-ready data
Why this matters

Instead of building RAG on noisy OCR dumps:
```
 BAD:
OCR → embeddings → RAG
```
You get:
```
GOOD:
OCR → clustering → clean subsets → embeddings → RAG
```

Benefits:

higher relevance
lower noise
faster search
less token usage
Example use case

Input:
```
5000 OCR books (mixed topics)
```
Output:
```
/library/
  /mechanics/
  /electronics/
  /chemistry/
  /junk/
  ```
Then build RAG only on:
```
/mechanics/
```
Requirements
```
numpy
umap-learn
hdbscan
plotly
```

Notes
Designed for large OCR corpora
-Works without metadata
-Robust to noisy text
-Scales to millions of chunks

License

MIT