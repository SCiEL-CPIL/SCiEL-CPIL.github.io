import numpy as np
import time

N = 50
DEADLINE = 143.0

def _ratio_fast(h):
    """Fast ratio computation for inner loops."""
    h = np.maximum(h, 0.0)
    s1 = float(np.sum(h))
    if s1 < 1e-15:
        return 0.0
    s2 = float(np.sum(h * h))
    n = len(h)
    pad = 2 * n - 1
    a = np.zeros(pad)
    a[:n] = h
    c = np.fft.ifft(np.fft.fft(a) ** 2).real
    sup = float(np.max(c)) * (1.0 / n)
    if sup < 1e-15:
        return 0.0
    return s2 / (s1 * sup)


def _ratio(h):
    return _ratio_fast(h)


def _sidon_set(max_size, seed=0):
    """Greedy Sidon-like set with randomization."""
    rng = np.random.default_rng(seed)
    s = [0]
    diffs = set()
    order = list(range(1, N))
    rng.shuffle(order)
    for c in order:
        new_diffs = set()
        ok = True
        for i in s:
            d = abs(c - i)
            if d in diffs or d in new_diffs:
                ok = False
                break
            new_diffs.add(d)
        if ok:
            s.append(c)
            diffs.update(new_diffs)
        if len(s) >= max_size:
            break
    return sorted(s)


def _de_full_space(deadline, pop_size=60, seed=42):
    """Differential evolution over the full 50-dim space."""
    from scipy.optimize import differential_evolution
    rng = np.random.default_rng(seed)
    
    def objective(x):
        x = np.maximum(x, 0.0)
        s = x.sum()
        if s < 1e-15:
            return 0.0
        x = x / s
        return -_ratio_fast(x)
    
    bounds = [(0.0, 1.0)] * N
    t_start = time.time()
    max_iter_budget = int((deadline - t_start) * 0.8 / 0.05)  # rough per-iter time estimate
    
    try:
        res = differential_evolution(
            objective, bounds,
            maxiter=max(50, min(max_iter_budget, 200)),
            popsize=pop_size,
            seed=seed,
            tol=1e-8,
            mutation=(0.5, 1.2),
            recombination=0.8,
            polish=True,
            update_interval=1,
            callback=None
        )
        x = np.maximum(res.x, 0.0)
        s = x.sum()
        if s < 1e-15:
            return None
        x = x / s
        return x
    except Exception:
        return None


def _de_sparse_support(deadline, pop_size=40, seed=123):
    """
    Differential evolution where the genotype is:
    - First K bits: binary mask for support (K = 50)
    - Remaining bits: continuous weights for each position
    We decode by taking positions where mask > 0.5 and using corresponding weights.
    """
    from scipy.optimize import differential_evolution
    rng = np.random.default_rng(seed)
    
    # Genotype: [mask_0, ..., mask_49, weight_0, ..., weight_49]
    dim = 2 * N
    
    def decode(x):
        mask = (x[:N] > 0.5).astype(float)
        weights = np.maximum(x[N:], 0.0)
        h = mask * weights
        s = h.sum()
        if s < 1e-15:
            return None
        return h / s
    
    def objective(x):
        h = decode(x)
        if h is None:
            return 0.0
        return -_ratio_fast(h)
    
    bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)] * N
    t_start = time.time()
    
    try:
        res = differential_evolution(
            objective, bounds,
            maxiter=80,
            popsize=pop_size,
            seed=seed,
            tol=1e-7,
            mutation=(0.6, 1.2),
            recombination=0.8,
            polish=False
        )
        x = res.x
        h = decode(x)
        if h is None:
            return None
        return h
    except Exception:
        return None


def _local_polish(h, deadline):
    """Fast local polish: coordinate-wise adjustments with normalization."""
    cur = h.copy()
    best_r = _ratio_fast(cur)
    
    for _ in range(200):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            for frac in (0.3, 0.5, 0.7, 1.0):
                for sign in (1.0, -1.0):
                    trial = cur.copy()
                    trial[i] = max(0.0, cur[i] * (1.0 + sign * frac))
                    s = trial.sum()
                    if s < 1e-15:
                        continue
                    trial = trial / s
                    sc = _ratio_fast(trial)
                    if sc > best_r + 1e-12:
                        cur = trial
                        best_r = sc
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            break
    
    return cur


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    def remaining(deadline):
        return max(0.0, deadline - time.time())
    
    # ---- Generate diverse initial candidates ----
    candidates = []
    
    # 1. Sidon sets of various sizes
    for sz in (6, 8, 10, 12, 14, 16, 18, 20):
        for seed in (0, 1, 2, 3, 4):
            s = _sidon_set(sz, seed)
            if len(s) >= 3:
                h = np.zeros(N)
                for i in s:
                    h[i] = 1.0
                candidates.append(h / h.sum())
    
    # 2. Random sparse sets
    for _ in range(40):
        k = int(rng.integers(5, 25))
        h = np.zeros(N)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.uniform(0.3, 1.5, k)
        candidates.append(h / h.sum())
    
    # 3. Structured patterns
    for period in (2, 3, 4, 5, 7, 11):
        h = np.zeros(N)
        h[::period] = 1.0
        candidates.append(h / h.sum())
    
    # 4. Gaussian-like shapes
    t = np.linspace(-1, 1, N)
    for width in (0.15, 0.25, 0.4, 0.6, 1.0):
        h = np.exp(-t**2 / width)
        candidates.append(h / h.sum())
    
    # 5. Two-peak patterns
    for d in (0.2, 0.35, 0.5, 0.7, 0.9):
        h = np.exp(-((t - d)**2) / 0.15) + np.exp(-((t + d)**2) / 0.15)
        candidates.append(h / h.sum())
    
    # Evaluate all candidates
    best_h = candidates[0].copy()
    best_score = _ratio_fast(best_h)
    for c in candidates[1:]:
        sc = _ratio_fast(c)
        if sc > best_score:
            best_score = sc
            best_h = c.copy()
    
    # ---- Phase 1: Differential Evolution on full space ----
    de_deadline = t0 + 50.0
    if time.time() < de_deadline - 2.0:
        try:
            h_de = _de_full_space(de_deadline, pop_size=50, seed=42)
            if h_de is not None:
                sc = _ratio_fast(h_de)
                if sc > best_score:
                    best_score, best_h = sc, h_de
        except Exception:
            pass
    
    # ---- Phase 2: Differential Evolution on sparse support representation ----
    de2_deadline = t0 + 90.0
    if time.time() < de2_deadline - 2.0:
        try:
            h_de2 = _de_sparse_support(de2_deadline, pop_size=40, seed=123)
            if h_de2 is not None:
                sc = _ratio_fast(h_de2)
                if sc > best_score:
                    best_score, best_h = sc, h_de2
        except Exception:
            pass
    
    # ---- Phase 3: Local polish from best so far ----
    polish_deadline = t0 + 110.0
    if time.time() < polish_deadline - 1.0:
        try:
            h_pol = _local_polish(best_h.copy(), polish_deadline)
            sc = _ratio_fast(h_pol)
            if sc > best_score:
                best_score, best_h = sc, h_pol
        except Exception:
            pass
    
    # ---- Phase 4: Another DE run with different seed on full space ----
    de3_deadline = t0 + 130.0
    if time.time() < de3_deadline - 2.0:
        try:
            h_de3 = _de_full_space(de3_deadline, pop_size=40, seed=777)
            if h_de3 is not None:
                sc = _ratio_fast(h_de3)
                if sc > best_score:
                    best_score, best_h = sc, h_de3
        except Exception:
            pass
    
    # ---- Phase 5: Final local polish ----
    final_deadline = t0 + 140.0
    if time.time() < final_deadline - 1.0:
        try:
            h_final = _local_polish(best_h.copy(), final_deadline)
            sc = _ratio_fast(h_final)
            if sc > best_score:
                best_score, best_h = sc, h_final
        except Exception:
            pass
    
    # Safety: ensure we return before the hard deadline
    if time.time() > DEADLINE - 2.0:
        pass  # just return what we have
    
    return best_h.tolist()
