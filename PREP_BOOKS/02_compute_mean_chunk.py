import os
import numpy as np
import time

PARTS_DIR = r"C:\VAST2_UPLOAD\out\engineering_ru_rag_e5_base_torch\parts"
OUT_PATH = r"C:\VAST2_UPLOAD\out\book_vectors\mean_chunk.npy"

files = sorted([f for f in os.listdir(PARTS_DIR) if f.endswith(".npy")])

total_sum = None
total_count = 0

t0 = time.time()

print("Processing parts...")

for i, fname in enumerate(files, 1):
    path = os.path.join(PARTS_DIR, fname)

    X = np.load(path)  # [N, 768]
    X = X.astype(np.float32)

    if total_sum is None:
        total_sum = np.zeros(X.shape[1], dtype=np.float64)

    total_sum += X.sum(axis=0)
    total_count += X.shape[0]

    if i % 5 == 0 or i == len(files):
        elapsed = time.time() - t0
        speed = total_count / max(elapsed, 1e-6)
        eta = (len(files) - i) * (elapsed / i)

        print(f"[{i}/{len(files)}] chunks={total_count} speed={speed:.0f} ch/s eta={eta/60:.1f} min")

mean_vec = total_sum / total_count
mean_vec = mean_vec.astype(np.float32)

np.save(OUT_PATH, mean_vec)

t1 = time.time()

print("DONE")
print(f"Chunks total: {total_count}")
print(f"Saved: {OUT_PATH}")
print(f"Time: {t1 - t0:.2f} sec")