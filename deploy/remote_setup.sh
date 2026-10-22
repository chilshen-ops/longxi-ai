set -uo pipefail
# ASCII only: this file is shipped over ssh/scp and must not depend on any encoding.
echo "=== 1) unpack code ==="
mkdir -p ~/ksr-moe
tar xzf ~/ksr-deploy.tgz -C ~/ksr-moe || { echo "unpack FAILED"; echo "__KSR_DONE__ rc=2"; exit 2; }
cd ~/ksr-moe || { echo "__KSR_DONE__ rc=2"; exit 2; }
find . -maxdepth 3 -type f -name '*.py' -o -maxdepth 3 -type f -name '*.json' | sort | sed 's/^/  /' | head -30

echo "=== 2) python env ==="
PY=python3
PIPFLAGS="--user"
if python3 -m venv .venv 2>/dev/null && [ -x .venv/bin/python ]; then
  PY="$HOME/ksr-moe/.venv/bin/python"
  PIPFLAGS=""
  echo "  using venv: $PY"
else
  echo "  venv unavailable, falling back to pip --user"
fi
$PY -V
$PY -m pip --version || { echo "__KSR_DONE__ rc=3"; exit 3; }
$PY -m pip install --disable-pip-version-check -q $PIPFLAGS --upgrade pip 2>/dev/null || true

install_try() {
  desc="$1"; shift
  echo "--- install $desc: $* ---"
  if $PY -m pip install --disable-pip-version-check -q $PIPFLAGS "$@" 2>/tmp/piperr; then
    echo "  OK (default index)"; return 0
  fi
  echo "  default index failed: $(tail -3 /tmp/piperr | tr '\n' ' ')"
  echo "  retry via tuna PyPI mirror"
  if $PY -m pip install --disable-pip-version-check -q $PIPFLAGS \
       -i https://pypi.tuna.tsinghua.edu.cn/simple "$@" 2>/tmp/piperr2; then
    echo "  OK (tuna mirror)"; return 0
  fi
  echo "  still failing: $(tail -3 /tmp/piperr2 | tr '\n' ' ')"
  return 1
}

install_try numpy numpy || { echo "__KSR_DONE__ rc=3"; exit 3; }
install_try faiss-cpu faiss-cpu || { echo "__KSR_DONE__ rc=3"; exit 3; }
if ! $PY -c "import torch" 2>/dev/null; then
  if ! install_try torch-cpu torch --index-url https://download.pytorch.org/whl/cpu; then
    echo "  last resort: torch from PyPI (large, but runs on CPU too)"
    install_try torch-pypi torch || { echo "__KSR_DONE__ rc=3"; exit 3; }
  fi
fi
$PY -c "import torch,numpy,faiss;print('  torch',torch.__version__,'| numpy',numpy.__version__,'| faiss',faiss.__version__)" \
  || { echo "__KSR_DONE__ rc=3"; exit 3; }
echo "  nproc=$(nproc) mem_gb=$(free -g | awk 'NR==2{print $2}')"

echo "=== 3) spec calculator selfcheck ==="
$PY tools/ksr_calc.py --selfcheck && echo "  calculator OK" || { echo "__KSR_DONE__ rc=4"; exit 4; }

echo "=== 4) model core tests ==="
$PY tests/test_model_core.py || echo "  (some tests failed, see above)"

echo "=== 5) four-claims experiment (KSR_THREADS=8) ==="
KSR_THREADS=8 PYTHONIOENCODING=utf-8 $PY -u -m ksr.experiment
rc=$?
echo "=== experiment exit code: $rc ==="
echo "__KSR_DONE__ rc=$rc"
