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

set "CHUNKS=%ROOT%\data\chunks.jsonl"
set "PARTS=%ROOT%\out\parts"
set "OUT=%ROOT%\out\book_vectors"

echo === BUILD BOOK VECTORS ===
%PY% 01_build_book_vectors.py

if %errorlevel% neq 0 goto :error

echo === MEAN CHUNK ===
%PY% 02_compute_mean_chunk.py

if %errorlevel% neq 0 goto :error

echo === CENTER BOOK VECTORS ===
%PY% 03_center_book_vectors.py

if %errorlevel% neq 0 goto :error

echo DONE
pause
exit /b 0

:error
echo ERROR
pause
exit /b 1