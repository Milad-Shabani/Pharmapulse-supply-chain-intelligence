@echo off
REM ============================================================
REM Publish this folder as a new public repo on github.com/Milad-Shabani
REM Requires: git, and GitHub CLI (gh) installed + logged in (gh auth login)
REM ============================================================

set REPO_NAME=pharmapulse-demand-planning
set REPO_DESC=PharmaPulse: synthetic pharmaceutical manufacturing & distribution dataset (42 SKUs, 14 distribution centers, ~430K demand rows) with a global LightGBM quantile demand-forecasting model, ABC/XYZ segmentation, safety-stock planning, expiry-risk analysis, and Excel + interactive HTML dashboard reporting.

REM --- adjust this to wherever you unzipped/cloned the project locally
cd /d "C:\Users\MILAD\Desktop\pharmapulse-demand-planning"

REM --- set your git identity (safe to run every time)
git config --global user.name "Milad Shabani"
git config --global user.email "MILAD.SHABANI6515@GMAIL.COM"

REM --- init only if not already a repo
if not exist ".git" (
    git init
    git branch -M main
)

REM --- remove any leftover remote from a previous attempt
git remote remove origin 2>nul

git add .
git commit -m "Initial commit: PharmaPulse - demand planning & inventory intelligence platform"
git branch -M main

gh repo create %REPO_NAME% --public --source=. --remote=origin --push --description "%REPO_DESC%"

gh repo edit Milad-Shabani/%REPO_NAME% --add-topic data-science --add-topic demand-forecasting --add-topic lightgbm --add-topic supply-chain --add-topic inventory-management --add-topic python --add-topic pharma --add-topic dashboard --add-topic excel

echo.
echo Done. Repo should now be live at:
echo https://github.com/Milad-Shabani/%REPO_NAME%
pause
