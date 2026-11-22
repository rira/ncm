$ErrorActionPreference = "Stop"

$RepoRoot = $PSScriptRoot
$CacheRoot = $RepoRoot
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    $GitPointer = Join-Path $RepoRoot ".git"
    if (Test-Path -LiteralPath $GitPointer -PathType Leaf) {
        $GitLine = (Get-Content -LiteralPath $GitPointer -TotalCount 1).Trim()
        if ($GitLine -match '^gitdir:\s*(.+)$') {
            $GitDir = $matches[1].Trim()
            if ($GitDir -match '^/mnt/([a-zA-Z])/(.+)$') {
                $GitDir = ($matches[1].ToUpper() + ":\" + ($matches[2] -replace '/', '\'))
            } else {
                $GitDir = $GitDir -replace '/', '\'
            }
            if (-not [System.IO.Path]::IsPathRooted($GitDir)) {
                $GitDir = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $GitDir))
            }
            $WorktreeParent = Split-Path -Parent $GitDir
            if ((Split-Path -Leaf $WorktreeParent) -eq "worktrees") {
                $CommonGitDir = Split-Path -Parent $WorktreeParent
                $CommonRoot = Split-Path -Parent $CommonGitDir
                $CommonPython = Join-Path $CommonRoot ".venv\Scripts\python.exe"
                if (Test-Path -LiteralPath $CommonPython) {
                    $PythonExe = $CommonPython
                    if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot ".hf-cache"))) {
                        $CacheRoot = $CommonRoot
                    }
                }
            }
        }
    }
}

if (-not (Test-Path -LiteralPath $PythonExe)) {
    Write-Error "Intel XPU virtual environment not found at '$PythonExe'. See README_INTEL_XPU.md for setup."
    exit 1
}

$env:HF_HOME = Join-Path $CacheRoot ".hf-cache"
$env:PYTHONIOENCODING = "utf-8"
if (-not $env:ONEAPI_DEVICE_SELECTOR) {
    $env:ONEAPI_DEVICE_SELECTOR = "level_zero:gpu"
}
$env:NCM_ROOT = $RepoRoot

& $PythonExe @args
exit $LASTEXITCODE
