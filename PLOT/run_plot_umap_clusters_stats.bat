@echo off
setlocal

cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  set "PY=py"
) else (
  set "PY=python"
)

set "ROOT=%~dp0.."

set "VEC=%ROOT%\out\book_vectors\book_vectors_chunk_centered.npy"
set "META=%ROOT%\out\book_vectors\book_vectors_full.jsonl"
set "CLUSTERS=%ROOT%\out\book_vectors\book_clusters_level3.jsonl"
set "OUT=%ROOT%\out\PLOT\umap_clusters_stats.html"

echo === BUILD HTML ===

%PY% plot_umap_clusters_stats.py ^
  --vec "%VEC%" ^
  --meta "%META%" ^
  --clusters "%CLUSTERS%" ^
  --out "%OUT%"

if %errorlevel% neq 0 goto :error

echo DONE
pause
exit /b 0

:error
echo ERROR
pause
exit /b 1