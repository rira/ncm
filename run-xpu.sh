#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CACHE_ROOT="$REPO_DIR"
VENV_PYTHON="$REPO_DIR/.venv/Scripts/python.exe"

if [[ ! -f "$VENV_PYTHON" ]]; then
    COMMON_GIT_DIR="$(git -C "$REPO_DIR" rev-parse --git-common-dir 2>/dev/null || true)"
    if [[ -n "$COMMON_GIT_DIR" ]]; then
        if [[ "$COMMON_GIT_DIR" != /* ]]; then
            COMMON_GIT_DIR="$REPO_DIR/$COMMON_GIT_DIR"
        fi
        COMMON_ROOT="$(cd "$(dirname "$COMMON_GIT_DIR")" && pwd -P)"
        if [[ -f "$COMMON_ROOT/.venv/Scripts/python.exe" ]]; then
            VENV_PYTHON="$COMMON_ROOT/.venv/Scripts/python.exe"
            if [[ ! -d "$REPO_DIR/.hf-cache" ]]; then
                CACHE_ROOT="$COMMON_ROOT"
            fi
        fi
    fi
fi

if [[ ! -f "$VENV_PYTHON" ]]; then
    echo "error: Intel XPU virtual environment not found at '$VENV_PYTHON' (see README_INTEL_XPU.md)" >&2
    exit 1
fi

export HF_HOME="$(wslpath -w "$CACHE_ROOT/.hf-cache")"
export NCM_ROOT="$(wslpath -w "$REPO_DIR")"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"

for var in HF_HOME NCM_ROOT NCM_DEVICE ONEAPI_DEVICE_SELECTOR PYTHONIOENCODING; do
    case ":${WSLENV:-}:" in
        *":${var}:"*|*":${var}/"*) ;;
        *) WSLENV="${WSLENV:+$WSLENV:}${var}" ;;
    esac
done
export WSLENV

cd "$REPO_DIR"
exec "$VENV_PYTHON" "$@"
