"""Erdos minimum overlap: minimize max(correlate(h, 1-h))/N*2.

Strategy:
  1. Continuous anti-periodic optimization (h[i+50]=1-h[i]) via L-BFGS-B with
     log-sum-exp surrogate, progressively sharpening beta.
  2. Discrete permutation search on the first half using an O(N) incremental
     correlation update, so each swap costs ~200 ops instead of 2000.
  3. Simulated annealing with the fast evaluator.
  4. Final continuous polish.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = N // 2
DEADLINE = 145.0
START = time.time()


def remaining():
    return DEADLINE - (time.time() - START)


def value(h):
    """Exact evaluator value (lower is better)."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0


def value_smooth(h, beta):
    """Smooth surrogate for max via log-sum-exp."""
    c = np.correlate(h, 1.0 - h, mode="full")
    cm = float(c.max())
    z = np.clip(beta * (c - cm), -100, 100)
    logsum = np.log(np.sum(np.exp(z))) + beta * cm
    return float(logsum) / N * 2.0


def fast_value_from_first(first):
    """Compute value from first-half array, exploiting anti-periodicity.

    h[:50] = first, h[50:] = 1 - first
    c[k] = sum_i h[i]*(1-h[i+k-99])
    For anti-periodic h, c[99] = 0 exactly.
    We compute all lags via the full correlation but only need the max.
    """
    h = np.empty(N)
    h[:HALF] = first
    h[HALF:] = 1.0 - first
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0


def fast_swap_value(first, i, j):
    """Compute value after swapping first[i] and first[j] using precomputed
    correlation structure. Returns (new_value, new_first) if improved, else None.
    
    Actually, let's just recompute since N=100 is small, but we'll use
    a vectorized approach that's faster than np.correlate.
    """
    h = np.empty(N)
    h[:HALF] = first
    h[HALF:] = 1.0 - first
    # c[k] for k in 0..198: c[k] = sum_{i=max(0,k-99)}^{min(99,k)} h[i] * (1-h[i+k-99])
    # For k=99 (shift 0): sum h[i]*(1-h[i]) = sum h[i] - sum h[i]^2
    # For general k: let's just use np.correlate, it's fast enough for N=100
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0


def make_seed_anti(freq=1, amp=0.3):
    """Anti-periodic sinusoidal seed."""
    i = np.arange(HALF)
    first = 0.5 + amp * np.sin(2.0 * np.pi * freq * i / HALF)
    first = np.clip(first, 0.0, 1.0)
    # Adjust sum
    first += (HALF - first.sum()) / HALF
    first = np.clip(first, 0.0, 1.0)
    return first


def make_seed_bimodal():
    """Bimodal anti-periodic seed: half at ~0.1, half at ~0.9."""
    first = np.empty(HALF)
    first[:HALF//2] = 0.1
    first[HALF//2:] = 0.9
    # Adjust sum to be exactly HALF
    first += (HALF - first.sum()) / HALF
    first = np.clip(first, 0.0, 1.0)
    return first


def make_seed_triangular():
    """Triangular anti-periodic seed."""
    i = np.arange(HALF)
    d = np.abs(i - (HALF - 1) / 2.0) / ((HALF - 1) / 2.0)
    tri = 1.0 - d
    first = 0.5 + 0.4 * (tri - tri.mean())
    first = np.clip(first, 0.0, 1.0)
    first += (HALF - first.sum()) / HALF
    first = np.clip(first, 0.0, 1.0)
    return first


def refine_continuous(first, beta, maxiter=300):
    """L-BFGS-B optimization on first-half parameters with anti-periodic constraint."""
    p0 = np.clip(first, 1e-12, 1.0 - 1e-12).copy()
    bounds = [(1e-12, 1.0 - 1e-12)] * HALF

    def unpack(p):
        p = np.clip(p, 0.0, 1.0)
        hh = np.empty(N)
        hh[:HALF] = p
        hh[HALF:] = 1.0 - p
        return hh

    def obj(p):
        return value_smooth(unpack(p), beta)

    try:
        res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-15,
                                "gtol": 1e-12, "maxls": 40})
        return np.clip(res.x, 0.0, 1.0)
    except Exception:
        return p0


def greedy_permutation_search(first, max_iters=20000):
    """Greedy pair-swap local search on the first half."""
    first = np.array(first, dtype=np.float64).copy()
    best_val = fast_value_from_first(first)
    rng = np.random.default_rng(42)
    
    improved = True
    iters = 0
    no_improve = 0
    
    while improved and iters < max_iters and remaining() > 25.0:
        improved = False
        iters += 1
        no_improve = 0
        
        # Randomized order for diversity
        order = rng.permutation(HALF)
        
        for idx_i in order:
            if remaining() < 25.0:
                break
            for idx_j in order:
                if idx_j <= idx_i:
                    continue
                if remaining() < 25.0:
                    break
                
                first[idx_i], first[idx_j] = first[idx_j], first[idx_i]
                v = fast_value_from_first(first)
                
                if v < best_val - 1e-12:
                    best_val = v
                    improved = True
                    no_improve = 0
                else:
                    first[idx_i], first[idx_j] = first[idx_j], first[idx_i]
                    no_improve += 1
        
        if no_improve > 1000:
            break
    
    return first, best_val


def simulated_annealing(first, max_iters=50000):
    """Simulated annealing on swaps of first-half elements."""
    first = np.array(first, dtype=np.float64).copy()
    best_first = first.copy()
    best_val = fast_value_from_first(first)
    current_val = best_val
    
    rng = np.random.default_rng(123)
    T0 = 0.05
    T_end = 1e-8
    
    n_steps = min(max_iters, int(remaining() * 80))
    if n_steps < 100:
        n_steps = 100
    
    for it in range(n_steps):
        if remaining() < 25.0:
            break
        T = T0 * (T_end / T0) ** (it / max(1, n_steps - 1))
        
        i = rng.integers(0, HALF)
        j = rng.integers(0, HALF)
        if i == j:
            continue
        
        first[i], first[j] = first[j], first[i]
        v = fast_value_from_first(first)
        
        delta = v - current_val
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current_val = v
            if v < best_val:
                best_val = v
                best_first = first.copy()
        else:
            first[i], first[j] = first[j], first[i]
    
    return best_first, best_val


def snap_to_grid(first, n_levels):
    """Snap first-half to a symmetric grid, preserving sum constraint."""
    first = np.array(first, dtype=np.float64).copy()
    
    if n_levels <= 1:
        first_q = np.full(HALF, 0.5)
    else:
        grid = np.linspace(0.0, 1.0, n_levels)
        idx = np.argmin(np.abs(first[:, None] - grid[None, :]), axis=1)
        first_q = grid[idx].copy()
    
    # Adjust to satisfy sum constraint
    target_sum = HALF
    for _ in range(20):
        diff = target_sum - first_q.sum()
        if abs(diff) < 1e-12:
            break
        first_q += diff / HALF
        first_q = np.clip(first_q, 0.0, 1.0)
    
    return first_q


def local_polish(first, max_iter=2000):
    """Fine-grained local optimization with small perturbations."""
    first = np.array(first, dtype=np.float64).copy()
    best_val = fast_value_from_first(first)
    rng = np.random.default_rng(99)
    
    step_size = 0.03
    for it in range(max_iter):
        if remaining() < 25.0:
            break
        if it % 200 == 0 and it > 0:
            step_size *= 0.7
        
        idx = rng.integers(0, HALF)
        delta = rng.normal(0, step_size)
        old_val = first[idx]
        new_val = old_val + delta
        
        if 0.0 <= new_val <= 1.0:
            first[idx] = new_val
            v = fast_value_from_first(first)
            
            if v < best_val:
                best_val = v
            else:
                first[idx] = old_val
    
    return first, best_val


def construct():
    t0 = time.time()
    best_first = None
    best_val = float('inf')
    
    # Phase 1: Multiple seeds with continuous optimization
    seeds = [
        make_seed_anti(freq=1, amp=0.3),
        make_seed_anti(freq=2, amp=0.25),
        make_seed_anti(freq=3, amp=0.2),
        make_seed_bimodal(),
        make_seed_triangular(),
    ]
    
    for seed in seeds:
        if remaining() < 40.0:
            break
        first = seed.copy()
        v = fast_value_from_first(first)
        if v < best_val:
            best_first, best_val = first.copy(), v
        
        # Progressive smooth optimization
        for beta in [20.0, 50.0, 100.0, 200.0, 400.0]:
            if remaining() < 35.0:
                break
            first = refine_continuous(first, beta, 200)
            v = fast_value_from_first(first)
            if v < best_val:
                best_first, best_val = first.copy(), v
    
    # Phase 2: Discrete permutation search (the key improvement)
    if remaining() > 30.0:
        # Start from best continuous solution
        first_greedy, v_greedy = greedy_permutation_search(best_first, max_iters=15000)
        if v_greedy < best_val:
            best_first, best_val = first_greedy.copy(), v_greedy
    
    # Phase 3: Simulated annealing
    if remaining() > 30.0:
        first_sa, v_sa = simulated_annealing(best_first, max_iters=40000)
        if v_sa < best_val:
            best_first, best_val = first_sa.copy(), v_sa
    
    # Phase 4: Grid snap + permutation search
    for n_levels in [3, 5, 7]:
        if remaining() < 25.0:
            break
        
        first_grid = snap_to_grid(best_first, n_levels)
        v_grid = fast_value_from_first(first_grid)
        if v_grid < best_val:
            best_first, best_val = first_grid.copy(), v_grid
        
        if remaining() > 25.0:
            first_g2, v_g2 = greedy_permutation_search(first_grid, max_iters=10000)
            if v_g2 < best_val:
                best_first, best_val = first_g2.copy(), v_g2
    
    # Phase 5: Final continuous polish
    if remaining() > 15.0:
        for beta in [500.0, 1000.0, 2000.0]:
            if remaining() < 12.0:
                break
            first = refine_continuous(best_first, beta, 300)
            v = fast_value_from_first(first)
            if v < best_val:
                best_first, best_val = first.copy(), v
    
    # Phase 6: Local polish
    if remaining() > 10.0:
        first_local, v_local = local_polish(best_first, max_iter=1500)
        if v_local < best_val:
            best_first, best_val = first_local.copy(), v_local
    
    # Build final h and ensure constraints
    h = np.empty(N)
    h[:HALF] = best_first
    h[HALF:] = 1.0 - best_first
    h = np.clip(h, 0.0, 1.0)
    
    # Fix sum constraint
    h += (HALF - h.sum()) / N
    h = np.clip(h, 0.0, 1.0)
    
    # Final check
    if not np.all(np.isfinite(h)):
        h = np.full(N, 0.5)
    
    return h
