"""Fresh design: parallel simulated annealing + local search for ±1 polynomials."""
# EVOLVE-BLOCK-START
import time
import math
import threading
import numpy as np

N = 24
DEADLINE = 110.0
_START = time.time()

# Precompute dense sampling grid on |z|=1
_M = 720
_THETA = np.linspace(0.0, 2.0 * np.pi, _M, endpoint=False)
_Z = np.exp(1j * _THETA)
_ZT = _Z.T  # (M, 1)

# Known-good (Golay-Rudin-Shapiro) fallback
def _fallback():
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    return np.array(p[:N], dtype=np.int8)

_FALLBACK = _fallback()

def _peak(coeffs):
    """Return max |P(z)| over |z|=1 via dense grid + golden-section refinement."""
    v = _Z @ coeffs
    m = np.abs(v)
    i = int(np.argmax(m))
    best = float(m[i])
    # golden-section refinement in the local interval
    t0 = _THETA[(i - 1) % _M]
    t2 = _THETA[(i + 1) % _M]
    if t2 < t0:
        t2 += 2.0 * np.pi
    # handle wrap for i==0
    if i == 0:
        t0 = -_THETA[1]
    # ensure interval contains theta_i
    a, b = t0, t2
    # golden section
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = float(np.abs(np.exp(1j * c) @ coeffs))
    fd = float(np.exp(1j * d) @ coeffs)
    for _ in range(40):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = float(np.exp(1j * c) @ coeffs)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = float(np.exp(1j * d) @ coeffs)
    local = max(fc, fd)
    if local > best:
        best = local
    return best

def _score(coeffs):
    return _peak(coeffs) / math.sqrt(N)

def _local_search(s, rng):
    """Hill-climb / steepest descent on single-bit flips."""
    best = s.copy()
    best_val = _peak(best)
    improved = True
    while improved and (time.time() - _START) < DEADLINE:
        improved = False
        vals = np.empty(N)
        for k in range(N):
            t = best.copy()
            t[k] = -t[k]
            vals[k] = _peak(t)
        kmin = int(np.argmin(vals))
        if vals[kmin] < best_val - 1e-12:
            best[kmin] = -best[kmin]
            best_val = vals[kmin]
            improved = True
    return best, best_val

def _worker(seed, stop_event, out):
    rng = np.random.default_rng(seed)
    s = rng.choice([-1, 1], size=N).astype(np.int8)
    cur_val = _peak(s)
    best = s.copy()
    best_val = cur_val
    T = 1.0
    Tmin = 0.02
    alpha = 0.999
    while not stop_event.is_set() and (time.time() - _START) < DEADLINE:
        # propose a flip
        k = int(rng.integers(0, N))
        t = s.copy()
        t[k] = -t[k]
        new_val = _peak(t)
        delta = new_val - cur_val
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
            s = t
            cur_val = new_val
            if cur_val < best_val:
                best_val = cur_val
                best = s.copy()
        T *= alpha
        if T < Tmin:
            T = Tmin
    # final local polish
    best, best_val = _local_search(best, rng)
    out[seed] = (best, best_val)

def construct():
    global _START
    _START = time.time()
    deadline = _START + DEADLINE
    results = {}
    stop = threading.Event()
    n_workers = 4
    threads = []
    for i in range(n_workers):
        t = threading.Thread(target=_worker, args=(1000 + i, stop, results))
        t.start()
        threads.append(t)
    # wait until just before deadline
    while (time.time() < deadline - 5.0) and any(t.is_alive() for t in threads):
        time.sleep(0.2)
    stop.set()
    for t in threads:
        t.join(timeout=2.0)

    # pick best among workers
    best = None
    best_val = float('inf')
    for k, (s, v) in results.items():
        if v < best_val:
            best_val = v
            best = s
    # include fallback
    fb_val = _peak(_FALLBACK)
    if fb_val < best_val:
        best = _FALLBACK
        best_val = fb_val
    if best is None:
        best = _FALLBACK
    # final polish
    best, _ = _local_search(best, np.random.default_rng(0))
    return list(best.astype(int))
# EVOLVE-BLOCK-END
