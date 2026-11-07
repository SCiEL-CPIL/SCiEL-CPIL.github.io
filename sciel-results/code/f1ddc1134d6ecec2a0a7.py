import numpy as np
from scipy.optimize import minimize
import time

N = 100
HALF = N / 2.0

def _value(h):
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (N,):
        return np.inf
    if np.any(~np.isfinite(h)):
        return np.inf
    h_clipped = np.clip(h, 0.0, 1.0)
    corr = np.correlate(h_clipped, 1.0 - h_clipped, mode="full")
    return float(np.max(corr)) / N * 2.0

def _max_corr(h):
    h = np.asarray(h, dtype=np.float64)
    corr = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(corr))

def _smooth_max(corr, lam):
    """Log-sum-exp approximation to max(corr)."""
    m = np.max(corr)
    if m > 1e10:
        return m
    scaled = corr / lam
    scaled = np.clip(scaled, -700, 700)
    return m + lam * np.log(np.sum(np.exp(scaled - np.max(scaled))))

def _local_binary_search(h_bin, max_iters=200):
    """Try swapping 1->0 and 0->1 pairs to reduce max correlation."""
    h = h_bin.astype(np.float64)
    current = _max_corr(h)
    ones = np.where(h_bin == 1)[0]
    zeros = np.where(h_bin == 0)[0]
    if len(ones) == 0 or len(zeros) == 0:
        return h
    
    improved = True
    it = 0
    while improved and it < max_iters:
        improved = False
        it += 1
        best_gain = 1e-12
        best_i = -1
        best_j = -1
        for i in ones:
            for j in zeros:
                h[i] = 0.0
                h[j] = 1.0
                new_val = _max_corr(h)
                gain = current - new_val
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_j = j
                h[i] = 1.0
                h[j] = 0.0
        if best_i >= 0:
            h[best_i] = 0.0
            h[best_j] = 1.0
            current = current - best_gain
            # Update index sets
            ones = np.where(h > 0.5)[0]
            zeros = np.where(h < 0.5)[0]
            improved = True
    return h

def _fix_sum(h):
    """Adjust h to satisfy sum == N/2, keeping in [0,1]."""
    h = h.copy()
    for _ in range(20):
        diff = HALF - np.sum(h)
        if abs(diff) < 1e-12:
            break
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    t_start = time.time()
    deadline = t_start + 135.0  # safety margin before 150s kill
    
    best_h = np.full(N, 0.5)
    best_val = _value(best_h)
    
    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - HALF}]
    
    def make_objective(lam):
        def objective(x):
            h = np.clip(x, 0.0, 1.0)
            corr = np.correlate(h, 1.0 - h, mode="full")
            return _smooth_max(corr, lam)
        return objective
    
    def try_optimize(start_h, lam, maxiter=200):
        nonlocal best_h, best_val
        try:
            res = minimize(
                make_objective(lam), start_h,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-14, 'iprint': 0}
            )
            h_candidate = np.clip(res.x, 0.0, 1.0)
            h_candidate = _fix_sum(h_candidate)
            val = _value(h_candidate)
            if val < best_val:
                best_h = h_candidate.copy()
                best_val = val
        except Exception:
            pass
    
    def try_binary_refine(h_cont):
        nonlocal best_h, best_val
        try:
            h_bin = (h_cont > 0.5).astype(np.int8)
            # Ensure exactly 50 ones
            if np.sum(h_bin) != 50:
                sorted_idx = np.argsort(h_cont)
                if np.sum(h_bin) > 50:
                    for idx in sorted_idx:
                        if np.sum(h_bin) == 50:
                            break
                        h_bin[idx] = 0
                else:
                    for idx in sorted_idx[::-1]:
                        if np.sum(h_bin) == 50:
                            break
                        h_bin[idx] = 1
            h_refined = _local_binary_search(h_bin, max_iters=100)
            val = _value(h_refined)
            if val < best_val:
                best_h = h_refined.copy()
                best_val = val
        except Exception:
            pass
    
    # Generate diverse starting points
    rng = np.random.default_rng(42)
    starts = []
    
    # Seed
    starts.append(np.full(N, 0.5))
    
    # Sine waves with various frequencies and phases
    for freq in range(1, 25):
        for phase in np.linspace(0, 2*np.pi, 4, endpoint=False):
            h_s = 0.5 + 0.45 * np.sin(2 * np.pi * freq * np.arange(N) / N + phase)
            h_s = np.clip(h_s, 0, 1)
            h_s = _fix_sum(h_s)
            starts.append(h_s)
    
    # Random
    for _ in range(40):
        h_r = rng.uniform(0, 1, N)
        h_r = _fix_sum(h_r)
        starts.append(h_r)
    
    # Multi-frequency
    for _ in range(30):
        h_m = np.full(N, 0.5)
        for _ in range(4):
            f = rng.integers(1, 18)
            p = rng.uniform(0, 2*np.pi)
            a = rng.uniform(0.08, 0.35)
            h_m += a * np.sin(2 * np.pi * f * np.arange(N) / N + p)
        h_m = np.clip(h_m, 0, 1)
        h_m = _fix_sum(h_m)
        starts.append(h_m)
    
    # Binary-like starts
    for _ in range(20):
        h_b = (rng.uniform(0, 1, N) > 0.5).astype(np.float64)
        if np.sum(h_b) != 50:
            idx = np.argsort(rng.uniform(0, 1, N))
            if np.sum(h_b) > 50:
                for i in idx:
                    if np.sum(h_b) == 50:
                        break
                    h_b[i] = 0
            else:
                for i in idx[::-1]:
                    if np.sum(h_b) == 50:
                        break
                    h_b[i] = 1
        starts.append(h_b)
    
    # Phase-shifted patterns (Erdos-style)
    for period in [3, 4, 5, 6, 7, 8, 10, 12]:
        for offset in range(period):
            h_p = np.zeros(N)
            for i in range(N):
                if (i + offset) % period < period // 2:
                    h_p[i] = 1.0
                else:
                    h_p[i] = 0.0
            h_p = _fix_sum(h_p)
            starts.append(h_p)
    
    # Annealed lambda schedule: start smooth, end sharp
    lambda_schedule = [10.0, 5.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]
    
    # Process starts with multiple lambda values (annealing)
    for si, start_h in enumerate(starts):
        if time.time() > deadline - 10:
            break
        # Try a few lambda values for each start
        for lam in [5.0, 1.0, 0.1, 0.01]:
            if time.time() > deadline - 10:
                break
            try_optimize(start_h, lam, maxiter=150)
    
    # Refine best solution with binary local search
    if time.time() < deadline - 5:
        try_binary_refine(best_h)
    
    # Additional refinement rounds on best solution
    for _ in range(3):
        if time.time() > deadline - 5:
            break
        try_optimize(best_h, 0.01, maxiter=200)
        try_binary_refine(best_h)
    
    return best_h.tolist()
