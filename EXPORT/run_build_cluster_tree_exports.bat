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

set "LEVEL3=%ROOT%\out\book_vectors\book_clusters_level3.jsonl"
set "BOOK_META=%ROOT%\out\book_vectors\book_vectors_chunk_centered.jsonl"
set "MANIFEST=%ROOT%\data\manifest.jsonl"
set "OUTDIR=%ROOT%\out\EXPORT"

echo === BUILD TREE ===

%PY% build_cluster_tree_exports.py ^
  --level3 "%LEVEL3%" ^
  --book-meta "%BOOK_META%" ^
  --manifest "%MANIFEST%" ^
  --out-dir "%OUTDIR%" ^
  --base-name "book_cluster_tree"

if %errorlevel% neq 0 goto :error

echo DONE
pause
exit /b 0

:error
echo ERROR
pause
exit /b 1