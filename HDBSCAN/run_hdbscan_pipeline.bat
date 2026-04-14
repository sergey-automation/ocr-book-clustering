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

echo === LEVEL 1 ===
%PY% 01_cluster_level1.py
if %errorlevel% neq 0 goto :error

echo === LEVEL 2 ===
%PY% 02_split_large_clusters.py
if %errorlevel% neq 0 goto :error

echo === LEVEL 3 ===
%PY% 03_split_large_branches.py
if %errorlevel% neq 0 goto :error

echo === REPORTS ===
%PY% reports\report_level2.py
%PY% reports\report_level3.py

echo DONE
pause
exit /b 0

:error
echo ERROR
pause
exit /b 1