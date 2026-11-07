"""Erdos minimum overlap: smooth surrogate + low-dim parametric + SA polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 142.0  # safety margin before 150s kill

def _score(h):
    if not np.all(np.isfinite(h)):
        return np.inf
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _is_valid(h):
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
        return False
    if abs(np.sum(h) - N / 2) > 1e-9 * N:
        return False
    return True

def _fix_sum(h):
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(200):
        diff = target - np.sum(h)
        if abs(diff) < 1e-13:
            break
        if diff > 0:
            mask = h < 1.0 - 1e-10
            if not np.any(mask):
                break
            room = 1.0 - h[mask]
            total_room = np.sum(room)
            if total_room < 1e-10:
                break
            f = min(1.0, diff / total_room)
            h[mask] += room * f
        else:
            mask = h > 1e-10
            if not np.any(mask):
                break
            val = h[mask]
            total_val = np.sum(val)
            if total_val < 1e-10:
                break
            f = min(1.0, -diff / total_val)
            h[mask] -= val * f
    diff = target - np.sum(h)
    if abs(diff) > 1e-13:
        if diff > 0:
            idx = np.argmin(h)
            if h[idx] < 1.0:
                h[idx] = min(1.0, h[idx] + diff)
        else:
            idx = np.argmax(h)
            if h[idx] > 0.0:
                h[idx] = max(0.0, h[idx] + diff)
        h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    start = time.time()
    def deadline():
        return time.time() > start + DEADLINE

    rng = np.random.RandomState(12345)
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)

    def try_update(h):
        nonlocal best_h, best_score
        if _is_valid(h):
            s = _score(h)
            if s < best_score - 1e-12:
                best_score = s
                best_h = h.copy()

    # --- Phase 1: generate diverse structured candidates ---
    candidates = []

    # Alternating (block size 2): 1,0,1,0,...
    h_alt = np.zeros(N)
    h_alt[0::2] = 1.0
    candidates.append(_fix_sum(h_alt.copy()))

    # Block patterns with various sizes
    for bs in [2, 3, 4, 5, 7, 10, 14, 25]:
        h_b = np.zeros(N)
        for i in range(N):
            h_b[i] = 1.0 if (i // bs) % 2 == 0 else 0.0
        candidates.append(_fix_sum(h_b.copy()))

    # Sinusoidal patterns
    for freq in [1, 2, 3, 4, 5, 7, 10, 20, 25, 50]:
        for phase in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]:
            x = np.arange(N)
            h_s = 0.5 + 0.5 * np.sin(2 * np.pi * freq * x / N + phase)
            candidates.append(_fix_sum(h_s.copy()))

    # Sum of two sinusoids
    for f1 in [2, 3, 5]:
        for f2 in [7, 13, 25]:
            x = np.arange(N)
            h_t = 0.5 + 0.3 * np.sin(2 * np.pi * f1 * x / N) + 0.3 * np.cos(2 * np.pi * f2 * x / N)
            candidates.append(_fix_sum(h_t.copy()))

    # Random
    for _ in range(20):
        candidates.append(_fix_sum(rng.uniform(0, 1, N)))

    # Gradient
    candidates.append(_fix_sum(np.linspace(0, 1, N).copy()))
    candidates.append(_fix_sum(np.linspace(1, 0, N).copy()))

    for c in candidates:
        if deadline():
            break
        try_update(c)

    # --- Phase 2: low-dim parametric optimization (Fourier coefficients) ---
    # h[i] = 0.5 + sum_{k=1}^{K} (a_k cos(2pi k i/N) + b_k sin(2pi k i/N))
    # Optimize over the 2K parameters to minimize sum(h^2) as smooth surrogate
    # while keeping h in [0,1] via penalty
    K = 5
    n_params = 2 * K

    def _fourier_params_to_h(p):
        x = np.arange(N)
        h = np.full(N, 0.5)
        for k in range(1, K + 1):
            h += p[2*(k-1)] * np.cos(2 * np.pi * k * x / N)
            h += p[2*(k-1)+1] * np.sin(2 * np.pi * k * x / N)
        return h

    def _obj_fourier(p, mu):
        h = _fourier_params_to_h(p)
        # smooth surrogate: minimize sum(h^2) which drives center correlation down
        surrogate = np.sum(h ** 2)
        # actual max correlation (soft-max via log-sum-exp for smoothness)
        corr = np.correlate(h, 1 - h, mode="full")
        # soft max
        beta = 50.0
        soft_max = (1.0 / beta) * np.log(np.sum(np.exp(beta * corr / N * 2)))
        # penalties
        h_clipped_viol = np.maximum(0, h - 1) ** 2 + np.maximum(0, -h) ** 2
        sum_err = np.sum(h) - N / 2.0
        pen = mu * (np.sum(h_clipped_viol) + sum_err ** 2)
        return soft_max + 0.5 * surrogate + pen

    # Multi-start over Fourier params
    for trial in range(30):
        if deadline():
            break
        p0 = rng.uniform(-0.5, 0.5, n_params)
        # Bias toward alternating for some trials
        if trial % 3 == 0:
            p0 = np.zeros(n_params)
            # Set k=2 to approximate alternating (cos(4pi i/N) = cos(2pi*2*i/N))
            p0[0] = 0.5  # a_1... wait k=1: cos(2pi*i/N)
            # For alternating we want k=50 (period 2): too high. Use k=2 with high freq
            # Actually alternating = cos(2pi * 50 * i / N) = cos(pi*i) = (-1)^i
            # That's k=50, beyond our K. Instead use direct alternating candidate already.
            # For parametric, let's try k=2, k=3
            p0[2] = 0.4  # a_2
            p0[4] = 0.4  # a_3

        for mu in [50, 500, 5000]:
            if deadline():
                break
            try:
                res = minimize(
                    _obj_fourier, p0, args=(mu,),
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * n_params,
                    options={'maxiter': 80, 'ftol': 1e-12}
                )
                if res.success:
                    p0 = res.x
                    h_cand = _fix_sum(_fourier_params_to_h(p0))
                    try_update(h_cand)
            except Exception:
                pass

    # --- Phase 3: direct 100D polish with L-BFGS-B on smooth objective ---
    def _obj_direct(x, mu):
        h = x
        corr = np.correlate(h, 1 - h, mode="full")
        beta = 50.0
        soft_max = (1.0 / beta) * np.log(np.sum(np.exp(beta * corr / N * 2)))
        surrogate = np.sum(h ** 2)
        h_viol = np.maximum(0, h - 1) ** 2 + np.maximum(0, -h) ** 2
        sum_err = np.sum(h) - N / 2.0
        pen = mu * (np.sum(h_viol) + sum_err ** 2)
        return soft_max + 0.3 * surrogate + pen

    for mu in [100, 1000, 10000]:
        if deadline():
            break
        try:
            res = minimize(
                _obj_direct, best_h.copy(), args=(mu,),
                method='L-BFGS-B',
                bounds=[(0, 1)] * N,
                options={'maxiter': 60, 'ftol': 1e-12}
            )
            if res.success:
                h_cand = _fix_sum(res.x)
                try_update(h_cand)
        except Exception:
            pass

    # --- Phase 4: Simulated Annealing polish ---
    h_sa = best_h.copy()
    cur_score = _score(h_sa)
    T = 0.008
    step = 0.04
    iters = 0
    max_sa = 12000
    no_improve = 0
    while iters < max_sa and not deadline():
        iters += 1
        nch = rng.randint(1, 5)
        idx = rng.choice(N, nch, replace=False)
        deltas = rng.uniform(-step, step, nch)
        h_new = h_sa.copy()
        for i, d in zip(idx, deltas):
            h_new[i] += d
        h_new = _fix_sum(h_new)
        if not _is_valid(h_new):
            continue
        s_new = _score(h_new)
        ds = s_new - cur_score
        if ds < 0 or rng.random() < np.exp(-ds / max(T, 1e-6)):
            h_sa = h_new
            cur_score = s_new
            if cur_score < best_score - 1e-12:
                best_score = cur_score
                best_h = h_sa.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        T *= 0.998
        # adapt step
        if no_improve > 300:
            step *= 0.9
            no_improve = 0
        if T < 1e-5:
            break

    if not _is_valid(best_h):
        best_h = np.full(N, 0.5)
    return best_h
