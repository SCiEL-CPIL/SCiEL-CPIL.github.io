import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0

def _correlation_value(h):
    """Compute the evaluator's value: max(corr) / N * 2."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _project_to_feasible(h):
    """Project h onto feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    for _ in range(200):
        diff = np.sum(h) - N / 2.0
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h > 1e-14
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
        else:
            mask = h < 1.0 - 1e-14
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    return h

def _soft_max_objective(h, T):
    """Smooth max approximation using log-sum-exp."""
    corr = np.correlate(h, 1 - h, mode="full")
    x = corr / T
    x_max = np.max(x)
    shifted = x - x_max
    sum_exp = np.sum(np.exp(shifted))
    lse = x_max + np.log(sum_exp)
    return T * lse / N * 2

def _soft_max_grad(h, T):
    """Gradient of the smooth max objective."""
    corr = np.correlate(h, 1 - h, mode="full")
    x = corr / T
    x_max = np.max(x)
    shifted = x - x_max
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    
    # d/dh of corr = d/dh of correlate(h, 1-h, mode="full")
    # corr[k] = sum_i h[i] * (1-h)[i-k+N-1] for appropriate ranges
    # Gradient w.r.t. h[i] = sum_k weights[k] * d(corr[k])/d(h[i])
    # d(corr[k])/d(h[i]) = (1-h)[i-k+N-1] - h[i] * 0 (since 1-h is fixed in this term)
    # Actually corr[k] = sum_j h[j] * (1-h)[j-(k-(N-1))]
    # Let's compute it properly
    # np.correlate(h, 1-h, mode="full")[k] = sum_j h[j] * (1-h)[j - k + (N-1)]
    # d/dh[i] = sum_k weights[k] * (1-h)[i - k + (N-1)]
    
    # More efficiently:
    # grad[i] = sum_k weights[k] * (1-h)[i - k + N - 1]
    # This is a convolution of weights with (1-h)
    
    # Let's use a direct approach
    grad = np.zeros(N)
    one_minus_h = 1.0 - h
    for i in range(N):
        # For each i, corr[k] involves h[i] when j=i, so k = i - j + N - 1 = N - 1
        # Wait, let me reconsider.
        # corr[k] = sum_j h[j] * v[j - k + N - 1] where v = 1-h
        # d(corr[k])/d(h[i]) = v[i - k + N - 1] if 0 <= i - k + N - 1 < N else 0
        # So grad[i] = sum_k weights[k] * v[i - k + N - 1]
        # Let m = i - k + N - 1, so k = i + N - 1 - m
        # grad[i] = sum_m v[m] * weights[i + N - 1 - m] where 0 <= i + N - 1 - m < 2N - 1
        # This is a convolution-like operation
        
        # Simpler: just compute directly
        for k in range(2 * N - 1):
            m = i - k + N - 1
            if 0 <= m < N:
                grad[i] += weights[k] * one_minus_h[m]
    
    return grad / N * 2

def _golden_section_search_1d(f, a, b, tol=1e-8, max_iter=100):
    """Golden section search for minimum of univariate function f on [a, b]."""
    phi = (np.sqrt(5) - 1) / 2  # ~0.618
    
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = f(c)
    fd = f(d)
    
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = f(d)
    
    return (a + b) / 2

def _coordinate_descent_golden(h, max_rounds=50, tol=1e-10):
    """Coordinate descent with golden section search along each coordinate."""
    h = h.copy()
    best_val = _correlation_value(h)
    
    for _ in range(max_rounds):
        improved = False
        for i in range(N):
            # Define 1D function: vary h[i], adjust others to maintain sum constraint
            # We'll vary h[i] and distribute the change to other elements
            def f_1d(x):
                h_new = h.copy()
                h_new[i] = x
                # Adjust sum
                diff = np.sum(h_new) - N / 2.0
                if abs(diff) > 1e-14:
                    # Distribute to other elements
                    mask = np.ones(N, dtype=bool)
                    mask[i] = False
                    if np.sum(mask) > 0:
                        h_new[mask] -= diff / np.sum(mask)
                        h_new = np.clip(h_new, 0.0, 1.0)
                return _correlation_value(h_new)
            
            # Find bounds for h[i]
            lo = max(0.0, h[i] - 0.3)
            hi = min(1.0, h[i] + 0.3)
            
            if hi - lo < 1e-10:
                continue
            
            x_opt = _golden_section_search_1d(f_1d, lo, hi)
            
            h_new = h.copy()
            h_new[i] = x_opt
            diff = np.sum(h_new) - N / 2.0
            if abs(diff) > 1e-14:
                mask = np.ones(N, dtype=bool)
                mask[i] = False
                if np.sum(mask) > 0:
                    h_new[mask] -= diff / np.sum(mask)
                    h_new = np.clip(h_new, 0.0, 1.0)
            
            val_new = _correlation_value(h_new)
            if val_new < best_val - 1e-12:
                h = h_new
                best_val = val_new
                improved = True
        
        if not improved:
            break
    
    return h, best_val

def _pairwise_golden_search(h, max_rounds=30):
    """Pairwise golden section search on pairs of coordinates."""
    h = h.copy()
    best_val = _correlation_value(h)
    
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    np.random.seed(42)
    np.random.shuffle(pairs)
    
    for i, j in pairs[:200]:  # Limit number of pairs
        def f_2d(xi, xj):
            h_new = h.copy()
            h_new[i] = xi
            h_new[j] = xj
            diff = np.sum(h_new) - N / 2.0
            if abs(diff) > 1e-14:
                mask = np.ones(N, dtype=bool)
                mask[i] = False
                mask[j] = False
                if np.sum(mask) > 0:
                    h_new[mask] -= diff / np.sum(mask)
                    h_new = np.clip(h_new, 0.0, 1.0)
            return _correlation_value(h_new)
        
        # 2D search via alternating 1D
        xi = h[i]
        xj = h[j]
        
        for _ in range(10):
            # Optimize xi
            lo_i = max(0.0, xi - 0.2)
            hi_i = min(1.0, xi + 0.2)
            xi_new = _golden_section_search_1d(lambda x: f_2d(x, xj), lo_i, hi_i)
            
            # Optimize xj
            lo_j = max(0.0, xj - 0.2)
            hi_j = min(1.0, xj + 0.2)
            xj_new = _golden_section_search_1d(lambda x: f_2d(xi_new, x), lo_j, hi_j)
            
            if abs(xi_new - xi) < 1e-8 and abs(xj_new - xj) < 1e-8:
                break
            xi, xj = xi_new, xj_new
        
        h_new = h.copy()
        h_new[i] = xi
        h_new[j] = xj
        diff = np.sum(h_new) - N / 2.0
        if abs(diff) > 1e-14:
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            mask[j] = False
            if np.sum(mask) > 0:
                h_new[mask] -= diff / np.sum(mask)
                h_new = np.clip(h_new, 0.0, 1.0)
        
        val_new = _correlation_value(h_new)
        if val_new < best_val - 1e-12:
            h = h_new
            best_val = val_new
    
    return h, best_val

def _local_polish(h, start_time, deadline):
    """Local polishing using multiple techniques."""
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    
    # Try L-BFGS-B with soft max
    for T in [5.0, 10.0, 20.0, 50.0]:
        if time.time() > deadline - 3:
            break
        
        def obj(x):
            x_proj = _project_to_feasible(x)
            return _soft_max_objective(x_proj, T)
        
        try:
            res = minimize(
                obj,
                best_h,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-9, 'disp': False}
            )
            h_opt = _project_to_feasible(res.x)
            val_opt = _correlation_value(h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Coordinate descent with golden section
    if time.time() < deadline - 5:
        h_cd, val_cd = _coordinate_descent_golden(best_h, max_rounds=30)
        if val_cd < best_val:
            best_val = val_cd
            best_h = h_cd.copy()
    
    # Pairwise golden search
    if time.time() < deadline - 3:
        h_pw, val_pw = _pairwise_golden_search(best_h, max_rounds=20)
        if val_pw < best_val:
            best_val = val_pw
            best_h = h_pw.copy()
    
    return best_h, best_val

def _incommensurate_pattern(f1, f2, phase1=0.0, phase2=0.0):
    """Generate pattern with incommensurate frequencies."""
    t = np.arange(N)
    h = 0.5 + 0.45 * np.cos(2 * np.pi * f1 * t / N + phase1) + 0.45 * np.cos(2 * np.pi * f2 * t / N + phase2)
    return _project_to_feasible(h)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    rng = np.random.default_rng(100)
    
    # Phase 1: Generate diverse initial candidates
    initial_candidates = []
    
    # Incommensurate frequency patterns (golden ratio related)
    phi = (1 + np.sqrt(5)) / 2
    freq_pairs = [
        (1, phi), (2, 2*phi), (3, 3*phi), (1, 1/phi), (2, 2/phi),
        (1, 1.5), (2, 3.5), (3, 5.5), (1, 2.5), (2, 4.5),
        (1.1, 2.3), (1.7, 3.1), (2.1, 4.3), (0.7, 1.9), (1.3, 2.9)
    ]
    for f1, f2 in freq_pairs:
        for p1 in [0.0, np.pi/2, np.pi, 3*np.pi/2]:
            for p2 in [0.0, np.pi/2, np.pi, 3*np.pi/2]:
                h_pat = _incommensurate_pattern(f1, f2, p1, p2)
                initial_candidates.append(h_pat)
    
    # Sine waves with different frequencies
    for f in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 25, 30]:
        h_sin = 0.5 + 0.48 * np.sin(2 * np.pi * f * np.arange(N) / N)
        initial_candidates.append(_project_to_feasible(h_sin))
    
    # Cosine waves
    for f in [1, 2, 3, 4, 5, 7, 10, 14, 20]:
        h_cos = 0.5 + 0.48 * np.cos(2 * np.pi * f * np.arange(N) / N)
        initial_candidates.append(_project_to_feasible(h_cos))
    
    # Block patterns
    for block_size in [2, 3, 4, 5, 7, 10, 14, 20, 25, 50]:
        h_block = np.zeros(N)
        for i in range(N):
            if (i // block_size) % 2 == 0:
                h_block[i] = 1.0
        initial_candidates.append(_project_to_feasible(h_block))
    
    # Random candidates
    for _ in range(30):
        h_rand = rng.uniform(0, 1, N)
        initial_candidates.append(_project_to_feasible(h_rand))
    
    # Evaluate and select top candidates
    candidate_vals = [_correlation_value(h) for h in initial_candidates]
    sorted_indices = np.argsort(candidate_vals)
    top_indices = sorted_indices[:15]
    
    # Phase 2: Polish each top candidate
    for idx in top_indices:
        if time.time() > deadline - 10:
            break
        
        h_candidate = initial_candidates[idx].copy()
        h_polished, val_polished = _local_polish(h_candidate, start_time, deadline)
        
        if val_polished < best_val:
            best_val = val_polished
            best_h = h_polished.copy()
    
    # Phase 3: Random restart polishing from best solution
    if time.time() < deadline - 15:
        n_restarts = 15
        for r in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            h_pert = best_h.copy()
            n_perturb = rng.integers(3, 8)
            indices = rng.choice(N, size=n_perturb, replace=False)
            for idx in indices:
                h_pert[idx] += rng.uniform(-0.1, 0.1)
            h_pert = _project_to_feasible(h_pert)
            
            h_polished, val_polished = _local_polish(h_pert, start_time, deadline)
            
            if val_polished < best_val:
                best_val = val_polished
                best_h = h_polished.copy()
    
    # Phase 4: Final fine-tuning
    if time.time() < deadline - 5:
        h_final, val_final = _coordinate_descent_golden(best_h, max_rounds=50)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final.copy()
    
    if time.time() < deadline - 3:
        h_final2, val_final2 = _pairwise_golden_search(best_h, max_rounds=25)
        if val_final2 < best_val:
            best_val = val_final2
            best_h = h_final2.copy()
    
    # Final projection and validation
    best_h = _project_to_feasible(best_h)
    diff = np.sum(best_h) - N / 2.0
    if abs(diff) > 1e-9:
        mask = (best_h > 1e-8) & (best_h < 1.0 - 1e-8)
        if np.any(mask):
            best_h[mask] -= diff / np.sum(mask)
        else:
            best_h -= diff / N
        best_h = np.clip(best_h, 0, 1)
    
    if not np.all(np.isfinite(best_h)):
        best_h = np.full(N, 0.5)
    
    return best_h
