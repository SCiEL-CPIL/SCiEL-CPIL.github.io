"""Program for Erdos minimum overlap task using anti-periodic model + progressive smooth optimization.

The objective max(overlap(h,1-h)) is minimized when h is anti-periodic
(h[i+N/2] = 1 - h[i]), which forces the overlap to 0 at the shifts
t = 0, N, 2N (the 'forbidden' shifts). We build a strong anti-periodic
seed and refine it with progressive smooth optimization, sharpening the
surrogate to target the true max correlation, followed by discrete refinement.
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
    """The exact evaluator value (lower is better)."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0


def value_smooth(h, beta):
    """Smooth surrogate for the max via log-sum-exp (lower is better)."""
    c = np.correlate(h, 1.0 - h, mode="full")
    cm = float(c.max())
    z = beta * (c - cm)
    # Clip for numerical stability
    z = np.clip(z, -100, 100)
    logsum = np.log(np.sum(np.exp(z))) + beta * cm
    return float(logsum) / N * 2.0


def make_seed():
    """Strong anti-periodic seed: triangular wave, zero-mean per anti-pair."""
    i = np.arange(HALF)
    d = np.abs(i - (HALF - 1) / 2.0) / ((HALF - 1) / 2.0)
    tri = 1.0 - d
    first = 0.5 + (0.5 - tri.mean()) * tri
    h = np.empty(N)
    h[:HALF] = first
    h[HALF:] = 1.0 - first
    h = np.clip(h, 0.0, 1.0)
    h += (HALF - h.sum()) / N
    h = np.clip(h, 0.0, 1.0)
    return h


def make_seed_v2():
    """Alternative seed: sinusoidal anti-periodic pattern."""
    i = np.arange(HALF)
    # Try multiple frequency patterns
    best_h = None
    best_v = float('inf')
    for freq in [1, 2, 3, 4, 5, 7]:
        phase = np.pi * freq * i / HALF
        first = 0.5 + 0.3 * np.sin(phase)
        h = np.empty(N)
        h[:HALF] = first
        h[HALF:] = 1.0 - first
        h = np.clip(h, 0.0, 1.0)
        h += (HALF - h.sum()) / N
        h = np.clip(h, 0.0, 1.0)
        v = value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
    return best_h


def make_seed_bimodal():
    """Bimodal anti-periodic seed: half at ~0.1, half at ~0.9."""
    first = np.empty(HALF)
    first[:HALF//2] = 0.1
    first[HALF//2:] = 0.9
    first += (HALF - first.sum()) / HALF
    first = np.clip(first, 0.0, 1.0)
    h = np.empty(N)
    h[:HALF] = first
    h[HALF:] = 1.0 - first
    h = np.clip(h, 0.0, 1.0)
    h += (HALF - h.sum()) / N
    h = np.clip(h, 0.0, 1.0)
    return h


def make_seed_triangular():
    """Triangular anti-periodic seed."""
    i = np.arange(HALF)
    d = np.abs(i - (HALF - 1) / 2.0) / ((HALF - 1) / 2.0)
    tri = 1.0 - d
    first = 0.5 + 0.4 * (tri - tri.mean())
    first = np.clip(first, 0.0, 1.0)
    first += (HALF - first.sum()) / HALF
    first = np.clip(first, 0.0, 1.0)
    h = np.empty(N)
    h[:HALF] = first
    h[HALF:] = 1.0 - first
    h = np.clip(h, 0.0, 1.0)
    h += (HALF - h.sum()) / N
    h = np.clip(h, 0.0, 1.0)
    return h


def refine_continuous(h, beta, maxiter):
    """Optimize the anti-periodic parameters to lower the smooth objective."""
    h0 = np.asarray(h, dtype=np.float64).copy()
    h0 = np.clip(h0, 1e-12, 1.0 - 1e-12)
    p0 = h0[:HALF].copy()
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
        return unpack(np.clip(res.x, 0.0, 1.0))
    except Exception:
        return h0


def snap_to_grid(h, n_levels):
    """Snap h to a symmetric grid of n_levels values, preserving anti-periodicity."""
    h = np.asarray(h, dtype=np.float64)
    first = h[:HALF].copy()
    
    if n_levels <= 1:
        first_q = np.full(HALF, 0.5)
    else:
        grid = np.linspace(0.0, 1.0, n_levels)
        idx = np.argmin(np.abs(first[:, None] - grid[None, :]), axis=1)
        first_q = grid[idx]
    
    # Adjust to satisfy sum constraint: sum(first_q) should be HALF
    target_sum = HALF
    current_sum = first_q.sum()
    diff = target_sum - current_sum
    
    if abs(diff) > 1e-10:
        first_q += diff / HALF
        first_q = np.clip(first_q, 0.0, 1.0)
        for _ in range(10):
            diff = target_sum - first_q.sum()
            if abs(diff) < 1e-12:
                break
            first_q += diff / HALF
            first_q = np.clip(first_q, 0.0, 1.0)
    
    h_out = np.empty(N)
    h_out[:HALF] = first_q
    h_out[HALF:] = 1.0 - first_q
    h_out = np.clip(h_out, 0.0, 1.0)
    
    total = h_out.sum()
    if abs(total - HALF) > 1e-9 * N:
        h_out += (HALF - total) / N
        h_out = np.clip(h_out, 0.0, 1.0)
    
    return h_out


def fast_value_from_first(first):
    """Compute value from first-half array, exploiting anti-periodicity."""
    h = np.empty(N)
    h[:HALF] = first
    h[HALF:] = 1.0 - first
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0


def greedy_permutation_search(h, max_iters=5000):
    """Greedy local search: swap pairs in the first half to reduce max correlation."""
    h = np.asarray(h, dtype=np.float64).copy()
    first = h[:HALF].copy()
    best_val = value(h)
    
    rng = np.random.default_rng(42)
    
    improved = True
    iters = 0
    while improved and iters < max_iters and remaining() > 20.0:
        improved = False
        iters += 1
        # Randomized order for diversity
        order = rng.permutation(HALF)
        for idx_i in order:
            if remaining() < 20.0:
                break
            for idx_j in order:
                if idx_j <= idx_i:
                    continue
                if remaining() < 20.0:
                    break
                first[idx_i], first[idx_j] = first[idx_j], first[idx_i]
                h_candidate = np.empty(N)
                h_candidate[:HALF] = first
                h_candidate[HALF:] = 1.0 - first
                v_candidate = value(h_candidate)
                
                if v_candidate < best_val - 1e-12:
                    best_val = v_candidate
                    improved = True
                else:
                    first[idx_i], first[idx_j] = first[idx_j], first[idx_i]
    
    h_out = np.empty(N)
    h_out[:HALF] = first
    h_out[HALF:] = 1.0 - first
    return h_out, best_val


def local_polish(h, max_iter=1000):
    """Fine-grained local optimization using small perturbations."""
    h = np.asarray(h, dtype=np.float64).copy()
    first = h[:HALF].copy()
    best_val = value(h)
    rng = np.random.default_rng(99)
    
    step_size = 0.05
    for it in range(max_iter):
        if remaining() < 20.0:
            break
        if it % 100 == 0:
            step_size *= 0.7
        
        idx = rng.integers(0, HALF)
        delta = rng.normal(0, step_size)
        new_val = first[idx] + delta
        
        if 0.0 <= new_val <= 1.0:
            first[idx] = new_val
            h_candidate = np.empty(N)
            h_candidate[:HALF] = first
            h_candidate[HALF:] = 1.0 - first
            v_candidate = value(h_candidate)
            
            if v_candidate < best_val:
                best_val = v_candidate
            else:
                first[idx] = new_val - delta
    
    h_out = np.empty(N)
    h_out[:HALF] = first
    h_out[HALF:] = 1.0 - first
    return h_out, best_val


def simulated_annealing_permutation(h, max_iters=20000):
    """Simulated annealing on permutations of the first half."""
    h = np.asarray(h, dtype=np.float64).copy()
    first = h[:HALF].copy()
    best_first = first.copy()
    best_val = value(h)
    current_val = best_val
    
    rng = np.random.default_rng(123)
    
    T0 = 0.1
    T_end = 1e-6
    n_steps = min(max_iters, int(remaining() * 100))
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
        h_candidate = np.empty(N)
        h_candidate[:HALF] = first
        h_candidate[HALF:] = 1.0 - first
        v_candidate = value(h_candidate)
        
        delta = v_candidate - current_val
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current_val = v_candidate
            if v_candidate < best_val:
                best_val = v_candidate
                best_first = first.copy()
        else:
            first[i], first[j] = first[j], first[i]
    
    h_out = np.empty(N)
    h_out[:HALF] = best_first
    h_out[HALF:] = 1.0 - best_first
    return h_out, best_val


def construct():
    t0 = time.time()

    # Phase 1: Try multiple seeds and continuous optimization
    seeds = [make_seed(), make_seed_v2(), make_seed_bimodal(), make_seed_triangular()]
    best = None
    best_val = float('inf')
    
    for seed in seeds:
        if remaining() < 50.0:
            break
        h = seed.copy()
        v = value(h)
        if v < best_val:
            best, best_val = h.copy(), v
        
        # Progressive smooth optimization with increasing beta
        for beta in [10.0, 20.0, 50.0, 100.0, 200.0, 400.0, 800.0]:
            if remaining() < 40.0:
                break
            h = refine_continuous(h, beta, 300)
            v = value(h)
            if v < best_val:
                best, best_val = h.copy(), v
    
    # Phase 2: Snap to discrete grid and do permutation search
    for n_levels in [3, 5, 7, 11, 15]:
        if remaining() < 30.0:
            break
        
        h_grid = snap_to_grid(best, n_levels)
        v_grid = value(h_grid)
        
        if v_grid < best_val:
            best, best_val = h_grid.copy(), v_grid
        
        # Greedy permutation search
        if remaining() > 25.0:
            h_greedy, v_greedy = greedy_permutation_search(h_grid, max_iters=3000)
            if v_greedy < best_val:
                best, best_val = h_greedy.copy(), v_greedy
        
        # Simulated annealing (if time permits)
        if remaining() > 25.0:
            h_sa, v_sa = simulated_annealing_permutation(
                h_greedy if 'h_greedy' in locals() else h_grid, max_iters=10000)
            if v_sa < best_val:
                best, best_val = h_sa.copy(), v_sa

    # Phase 3: Final continuous polish on the best discrete solution
    if remaining() > 15.0:
        p0 = np.clip(best[:HALF], 0.0, 1.0).copy()
        bounds = [(0.0, 1.0)] * HALF

        def unpack(p):
            p = np.clip(p, 0.0, 1.0)
            hh = np.empty(N)
            hh[:HALF] = p
            hh[HALF:] = 1.0 - p
            return hh

        # Try multiple betas for the final polish
        for beta in [200.0, 500.0, 1000.0, 2000.0]:
            if remaining() < 10.0:
                break
            try:
                res = minimize(lambda p: value_smooth(unpack(p), beta), p0,
                               method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": 500, "ftol": 1e-16,
                                        "gtol": 1e-14, "maxls": 50})
                h = unpack(np.clip(res.x, 0.0, 1.0))
                v = value(h)
                if v < best_val:
                    best, best_val = h.copy(), v
                    p0 = np.clip(res.x, 0.0, 1.0).copy()
            except Exception:
                pass

    # Phase 4: Local polish with small perturbations
    if remaining() > 10.0:
        h_local, v_local = local_polish(best, max_iter=500)
        if v_local < best_val:
            best, best_val = h_local.copy(), v_local

    # Ensure constraints are met
    best = np.clip(best, 0.0, 1.0)
    best += (HALF - best.sum()) / N
    best = np.clip(best, 0.0, 1.0)
    
    return best
