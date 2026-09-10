#!/usr/bin/env bash
#
# One-shot publish script: initializes a git repo here (if not already
# one), commits everything, and pushes it to a GitHub remote.
#
# Usage:
#   ./scripts/publish_to_github.sh https://github.com/<you>/<repo>.git
#   ./scripts/publish_to_github.sh git@github.com:<you>/<repo>.git
#
# Prerequisites:
#   1. You've already created an EMPTY repository on github.com
#      (do not initialize it with a README/license/.gitignore there).
#   2. `git` is installed and you're authenticated for the URL you pass
#      in (HTTPS -> a GitHub personal access token or the GitHub CLI's
#      cached credentials; SSH -> your SSH key is added to GitHub).
#
# This script only touches the local repo and your `origin` remote -
# it never stores or transmits credentials itself.

set -euo pipefail

REMOTE_URL="${1:-}"
BRANCH="${2:-main}"

if [[ -z "${REMOTE_URL}" ]]; then
  echo "Usage: $0 <github-remote-url> [branch-name]" >&2
  echo "Example: $0 https://github.com/Milad-Shabani/pharmapulse-demand-planning.git" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

echo "==> Working directory: ${PROJECT_ROOT}"

if [[ ! -d .git ]]; then
  echo "==> Initializing a new git repository..."
  git init
  git checkout -b "${BRANCH}"
else
  echo "==> Existing git repository detected."
  git checkout -B "${BRANCH}"
fi

echo "==> Staging files..."
git add -A

if git diff --cached --quiet; then
  echo "==> Nothing to commit (working tree already matches the last commit)."
else
  echo "==> Committing..."
  git commit -m "PharmaPulse: synthetic pharma demand-planning dataset + global LightGBM quantile forecasting + Excel/HTML reporting"
fi

if git remote get-url origin >/dev/null 2>&1; then
  echo "==> Updating existing 'origin' remote to ${REMOTE_URL}"
  git remote set-url origin "${REMOTE_URL}"
else
  echo "==> Adding 'origin' remote: ${REMOTE_URL}"
  git remote add origin "${REMOTE_URL}"
fi

echo "==> Pushing to origin/${BRANCH}..."
git push -u origin "${BRANCH}"

echo ""
echo "Done. Your repository should now be live at:"
echo "  ${REMOTE_URL%.git}"
