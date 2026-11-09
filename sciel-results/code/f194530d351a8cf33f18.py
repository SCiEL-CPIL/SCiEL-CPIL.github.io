"""Rewritten program: minimize max|sidelobe autocorrelation| / (sum x)^2 for signed step heights."""
import time
import numpy as np

N = 400
_K = np.arange(1, N)  # lags 1..N-1

def _ac_pos(x):
    n = len(x)
    f = np.fft.fft(x, n=2*n)
    ac = np.fft.ifft(f*np.conj(f)).real
    return ac[1:n].copy()

def _metric(x, ac, s):
    if s == 0:
        return float('inf')
    return float(np.max(np.abs(ac)) / (s*s))

def construct():
    deadline = time.time() + 100.0  # safety margin under 120s
    n = N
    best_x = np.ones(n)
    best_m = float('inf')

    def evaluate_state(x):
        s = x.sum()
        ac = _ac_pos(x)
        m = _metric(x, ac, s)
        return x, ac, s, m

    def try_restart(bias):
        nonlocal best_x, best_m
        # build initial x
        if bias is None:
            x = np.ones(n)
        else:
            x = np.where(np.random.rand(n) < bias, -1.0, 1.0)
        x = x.copy()
        s = x.sum()
        ac = _ac_pos(x)
        m = _metric(x, ac, s)
        if m < best_m:
            best_m = m; best_x = x.copy()
        # SA
        T = 1e-7
        T_min = 1e-12
        cool = 0.9999
        stuck = 0
        it = 0
        while time.time() < deadline - 2.0:
            j = np.random.randint(n)
            v = x[j]
            km = _K
            li = np.clip(j - km, 0, n-1)
            ri = np.clip(j + km, 0, n-1)
            left = np.where(km <= j, x[li], 0.0)
            right = np.where(km + j < n, x[ri], 0.0)
            delta = -2.0*v*(left+right)
            new_ac = ac + delta
            new_s = s - 2.0*v
            if new_s == 0:
                new_m = float('inf')
            else:
                new_m = float(np.max(np.abs(new_ac)) / (new_s*new_s))
            accept = False
            if new_m < m:
                accept = True
            else:
                d = new_m - m
                if d < 1e-15:
                    accept = True
                else:
                    # annealing acceptance
                    if T > T_min and np.random.rand() < np.exp(-d/T):
                        accept = True
            if accept:
                x[j] = -v
                ac = new_ac
                s = new_s
                m = new_m
                stuck = 0
                if m < best_m:
                    best_m = m; best_x = x.copy()
            else:
                stuck += 1
            it += 1
            T *= cool
            # kick to escape local minima
            if stuck > 2000:
                # random multi-flip kick
                r = np.random.randint(3, 25)
                idx = np.random.randint(0, n, r)
                x[idx] = -x[idx]
                s = x.sum()
                ac = _ac_pos(x)
                m = _metric(x, ac, s)
                stuck = 0
                T = 1e-7
                if m < best_m:
                    best_m = m; best_x = x.copy()
    # initial run from all +1
    try_restart(None)
    # additional biased restarts if time remains
    for bias in (0.1, 0.2, 0.05, 0.15):
        if time.time() > deadline - 10.0:
            break
        try_restart(bias)
    if not np.isfinite(best_m):
        return np.ones(n)
    return best_x
