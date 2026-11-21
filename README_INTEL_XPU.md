# Intel Arc XPU Runtime Guide (Windows 11 + WSL2)

Reference setup and launchers for running NCM on a Windows 11 Intel Arc GPU from either Windows PowerShell or WSL2.

## Verified Environment

- **Host OS**: Windows 11 (`win_amd64`)
- **Python**: `3.12.16` in `.venv\` (managed via `.tools\uv.exe`)
- **PyTorch XPU**: `torch==2.15.0+xpu`, `torchvision==0.29.2+xpu`, `torchaudio==2.11.1+xpu`, `triton-xpu==3.9.0`
- **Detected XPU Device (`xpu:0`)**: `Intel(R) Arc(TM) A770 Graphics` (`15932 MB` / ~15.6 GiB VRAM, 512 EUs, Level-Zero driver `1.15.39183+3`)

## Provisioning from Scratch (Windows PowerShell)

1. Optional standalone `uv` (no admin/UAC required):

   ```powershell
   $ProgressPreference = 'SilentlyContinue'
   New-Item -ItemType Directory -Force -Path .tools | Out-Null
   Invoke-WebRequest -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile '.tools\uv.zip'
   Expand-Archive -Path '.tools\uv.zip' -DestinationPath '.tools' -Force
   Remove-Item '.tools\uv.zip'
   ```

2. Create a Python 3.12 virtual environment and install PyTorch XPU wheels **first** (so PyPI never overwrites the `+xpu` build):

   ```powershell
   .\.tools\uv.exe venv .venv --python 3.12
   .\.tools\uv.exe pip install --python .venv\Scripts\python.exe torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
   .\.tools\uv.exe pip install --python .venv\Scripts\python.exe -r requirements-xpu.txt
   ```

3. Verify the runtime and detected XPU device:

   ```powershell
   .\run-xpu.ps1 -c "import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.get_device_name(0))"
   ```

## Launchers

Both launchers set `HF_HOME` (`.hf-cache`), `NCM_ROOT`, `PYTHONIOENCODING=utf-8`, and `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` (unless already overridden), propagate exit codes, and automatically resolve the main checkout's `.venv` and `.hf-cache` when invoked inside a `git worktree`.

### Windows PowerShell (`run-xpu.ps1`)

```powershell
.\run-xpu.ps1 -m pytest
.\run-xpu.ps1 evaluate.py --config config/eval/sudoku-easy.yaml --checkpoint-dir artifacts/sudoku-easy --data-dir data/sudoku-easy
```

### WSL2 Bash (`run-xpu.sh`)

Resolves symlinks (e.g. `~/projects/ncm` -> `/mnt/d/projects/ncm`), translates `HF_HOME` and `NCM_ROOT` via `wslpath -w`, exports `HF_HOME`, `NCM_ROOT`, `NCM_DEVICE`, `ONEAPI_DEVICE_SELECTOR`, and `PYTHONIOENCODING` via `WSLENV`, and executes `.venv/Scripts/python.exe`:

```bash
./run-xpu.sh -c "import torch; print(torch.__version__, torch.xpu.is_available())"
./run-xpu.sh -m pytest
```

When calling `powershell.exe` non-interactively from WSL2 scripts or automation, pass `-NoProfile -NonInteractive -InputFormat None` so PowerShell 5.1 does not block waiting on piped stdin.

## Dual-GPU Handling (`Arc A770` vs `UHD 770`)

`ONEAPI_DEVICE_SELECTOR=level_zero:gpu` restricts PyTorch XPU to Level-Zero GPU devices. If both an integrated `Intel(R) UHD Graphics 770` and a discrete `Intel(R) Arc(TM) A770 Graphics` are active, list device indices and pin to the Arc A770:

```bash
./run-xpu.sh -c "import torch; [print(i, torch.xpu.get_device_name(i), torch.xpu.get_device_properties(i).total_memory) for i in range(torch.xpu.device_count())]"
```

Then select the Arc A770 explicitly via `ONEAPI_DEVICE_SELECTOR` or `NCM_DEVICE`, for example:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 NCM_DEVICE=xpu:0 ./run-xpu.sh evaluate.py ...
```

## VRAM Contention (Ollama)

If Ollama has a model resident in Arc A770 VRAM, unload it before running evaluations to keep the full ~15.6 GiB available:

```bash
ollama ps
ollama stop <model>
```

## Windows WDDM `TdrDelay` Note (Optional)

If Windows Timeout Detection and Recovery (TDR) resets the display driver during long first-run SYCL or Triton JIT kernel compilations, increase `TdrDelay` to `20` seconds (requires elevated PowerShell and a reboot):

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" -Name "TdrDelay" -Value 20 -PropertyType DWord -Force
```
