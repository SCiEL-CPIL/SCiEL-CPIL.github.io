# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_P(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] for discrete measure with given support and weights.
    Uses O(n^2 log n) vectorized approach with sorted pairwise sums and cumulative weights.
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = len(s)
    if n == 0:
        return 0.0
    w_sum = w.sum()
    if w_sum <= 1e-15:
        return 0.0
    w = w / w_sum
    
    # Pairwise sums and weights for X2+X3
    s_pair = s[:, None] + s[None, :]  # (n, n)
    w_pair = w[:, None] * w[None, :]  # (n, n)
    
    # Flatten and sort by sum
    flat_s = s_pair.ravel()  # (n*n,)
    flat_w = w_pair.ravel()  # (n*n,)
    
    order = np.argsort(flat_s, kind='mergesort')
    sorted_s = flat_s[order]
    sorted_w = flat_w[order]
    cum_w = np.cumsum(sorted_w)
    
    # Targets: targets[k, l] = 2*s[l] - s[k]
    targets = 2.0 * s[None, :] - s[:, None]  # (n, n), targets[k, l]
    
    # For each target, find cumulative weight of pairs with sum < target
    flat_targets = targets.ravel()  # (n*n,)
    idx = np.searchsorted(sorted_s, flat_targets, side='left')  # (n*n,)
    
    P_X2X3 = np.zeros_like(flat_targets)
    mask = idx > 0
    P_X2X3[mask] = cum_w[idx[mask] - 1]
    
    P_X2X3 = P_X2X3.reshape(n, n)
    
    # P = sum_k sum_l w[k]*w[l] * P_X2X3[k, l]
    P = np.sum(w[:, None] * w[None, :] * P_X2X3)
    
    return P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1..X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = start_time + 610.0  # safety margin before 630s kill

    best_P = -1.0
    best_support = None
    best_weights = None

    def update_best(s, w):
        nonlocal best_P, best_support, best_weights
        if len(s) == 0:
            return
        w = np.asarray(w, dtype=np.float64)
        s = np.asarray(s, dtype=np.float64)
        w_sum = w.sum()
        if w_sum <= 1e-15:
            return
        w = w / w_sum
        # Ensure non-negative support
        if np.any(s < -1e-12):
            return
        # Ensure finite
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(w))):
            return
        P = _compute_P(s, w)
        if P > best_P:
            best_P = P
            best_support = s.copy()
            best_weights = w.copy()

    # --- Phase 1: Analytic 2-point baseline (fast) ---
    def two_point_P(p):
        """P[X1+X2+X3 < 2*X4] for 2-point {0, 1} with mass p on 0."""
        # X4=0: need sum < 0 (impossible)
        # X4=1: need sum < 2. Sum can be 0,1,2,3. P[sum<2] = p^3 + 3*p^2*(1-p)
        return (1.0 - p) * (p**3 + 3.0 * p**2 * (1.0 - p))
    
    a, b = 0.01, 0.99
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = two_point_P(c)
    fd = two_point_P(d)
    for _ in range(100):
        if fc < fd:
            a = c; c = d; fc = fd
            d = a + gr * (b - a); fd = two_point_P(d)
        else:
            b = d; d = c; fd = fc
            c = b - gr * (b - a); fc = two_point_P(c)
    p_opt = (a + b) / 2.0
    update_best(np.array([0.0, 1.0]), np.array([p_opt, 1.0 - p_opt]))

    # --- Phase 2: Simulated annealing over support size and geometry ---
    # Structural change: dynamically vary number of support points, explore
    # the piecewise-constant indicator landscape more broadly.
    
    def _sample_new_measure(current_s, current_w, temperature, n_min=2, n_max=10):
        """
        Generate a new candidate measure by perturbing or changing the current one.
        temperature controls the magnitude of perturbation.
        """
        n_cur = len(current_s)
        
        # Decide whether to change the number of points
        r = np.random.random()
        if r < 0.3:
            # Change number of points
            n_new = np.random.randint(n_min, n_max + 1)
            if n_new == n_cur:
                n_new = n_cur + 1
            # Generate new support points
            # Mix of structured and random
            if np.random.random() < 0.5:
                # Structured: geometric or arithmetic progression
                ratio = np.random.uniform(0.5, 3.0)
                s_new = np.array([0.0] + [ratio**i for i in range(1, n_new)])
                if np.random.random() < 0.5:
                    s_new = np.sort(s_new + np.random.uniform(-0.1, 0.1, n_new) * temperature)
                    s_new = np.maximum(s_new, 0.0)
                w_new = np.random.dirichlet(np.ones(n_new))
            else:
                # Random support in [0, 5]
                s_new = np.sort(np.random.uniform(0, 5.0, n_new))
                w_new = np.random.dirichlet(np.ones(n_new))
            return s_new, w_new
        else:
            # Perturb current measure
            s_new = current_s.copy().astype(np.float64)
            w_new = current_w.copy().astype(np.float64)
            
            # Perturb support
            s_new += np.random.normal(0, temperature * 0.5, n_cur)
            s_new = np.maximum(s_new, 0.0)
            s_new = np.sort(s_new)
            
            # Perturb weights via multiplicative noise
            w_new *= np.exp(np.random.normal(0, temperature * 0.3, n_cur))
            w_new = np.maximum(w_new, 1e-10)
            w_new = w_new / w_new.sum()
            
            return s_new, w_new

    # Annealing parameters
    T0 = 2.0
    T_min = 0.001
    cooling_rate = 0.995
    max_iter = 3000
    
    s_cur = best_support.copy() if best_support is not None else np.array([0.0, 1.0])
    w_cur = best_weights.copy() if best_weights is not None else np.array([0.5, 0.5])
    P_cur = best_P
    
    T = T0
    accepted = 0
    total = 0
    
    for it in range(max_iter):
        if time.time() > deadline - 30.0:
            break
        
        s_new, w_new = _sample_new_measure(s_cur, w_cur, T)
        
        # Quick validity check
        if len(s_new) != len(w_new) or len(s_new) < 1:
            continue
        if np.any(s_new < -1e-12) or np.any(w_new < -1e-12):
            continue
        if not (np.all(np.isfinite(s_new)) and np.all(np.isfinite(w_new))):
            continue
        
        P_new = _compute_P(s_new, w_new)
        total += 1
        
        if P_new > P_cur:
            # Accept
            s_cur = s_new
            w_cur = w_new
            P_cur = P_new
            accepted += 1
            if P_new > best_P:
                best_P = P_new
                best_support = s_new.copy()
                best_weights = w_new.copy()
        else:
            # Metropolis acceptance
            delta = P_new - P_cur
            if delta < 0:
                accept_prob = np.exp(delta / T) if T > 1e-10 else 0.0
                if np.random.random() < accept_prob:
                    s_cur = s_new
                    w_cur = w_new
                    P_cur = P_new
                    accepted += 1
        
        # Cool
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # --- Phase 3: Local refinement using scipy on the best found measure ---
    if time.time() < deadline - 20.0 and best_support is not None:
        n_best = len(best_support)
        
        # Optimize weights for fixed support
        def _log_w_to_w(x):
            x_shifted = x - x.max()
            e = np.exp(x_shifted)
            return e / e.sum()
        
        def _neg_P_log_w(x):
            w = _log_w_to_w(x)
            return -_compute_P(best_support, w)
        
        x0 = np.log(best_weights + 1e-300)
        try:
            result = minimize(_neg_P_log_w, x0, method='L-BFGS-B',
                             options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
            if result.fun < -best_P:
                w_opt = _log_w_to_w(result.x)
                update_best(best_support, w_opt)
        except Exception:
            pass
        
        # Optimize support and weights jointly (if n is small enough)
        if n_best <= 6 and time.time() < deadline - 15.0:
            def _neg_P_full(params):
                log_w = params[:n_best]
                s = params[n_best:]
                if np.any(s < -1e-12):
                    return 1e10
                w = _log_w_to_w(log_w)
                s_sorted = np.sort(s)
                # Re-sort weights accordingly
                order = np.argsort(s)
                w_sorted = w[order]
                return -_compute_P(s_sorted, w_sorted)
            
            # Initialize
            s_init = best_support.copy()
            w_init = best_weights.copy()
            log_w_init = np.log(w_init + 1e-300)
            params0 = np.concatenate([log_w_init, s_init])
            
            try:
                result2 = minimize(_neg_P_full, params0, method='Nelder-Mead',
                                  options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
                if result2.fun < -best_P:
                    log_w_opt = result2.x[:n_best]
                    s_opt = result2.x[n_best:]
                    w_opt = _log_w_to_w(log_w_opt)
                    s_sorted = np.sort(s_opt)
                    order = np.argsort(s_opt)
                    w_sorted = w_opt[order]
                    update_best(s_sorted, w_sorted)
            except Exception:
                pass

    # --- Phase 4: Try a few structured families with optimized weights ---
    if time.time() < deadline - 10.0:
        # Geometric supports
        for n in [3, 4, 5]:
            for ratio in [1.5, 2.0, 2.5, 3.0]:
                if time.time() > deadline - 5.0:
                    break
                s_geo = np.array([0.0] + [ratio**i for i in range(1, n)])
                w_geo = np.random.dirichlet(np.ones(n))
                
                # Optimize weights
                def _neg_P_log_w_geo(x):
                    w = _log_w_to_w(x)
                    return -_compute_P(s_geo, w)
                
                x0 = np.log(w_geo + 1e-300)
                try:
                    res = minimize(_neg_P_log_w_geo, x0, method='L-BFGS-B',
                                 options={'maxiter': 100, 'ftol': 1e-12})
                    if res.fun < -best_P:
                        w_opt = _log_w_to_w(res.x)
                        update_best(s_geo, w_opt)
                except Exception:
                    pass
        
        # Arithmetic supports
        for n in [3, 4, 5]:
            for spacing in [0.5, 1.0, 1.5, 2.0]:
                if time.time() > deadline - 5.0:
                    break
                s_arith = np.arange(n) * spacing
                w_arith = np.random.dirichlet(np.ones(n))
                
                def _neg_P_log_w_arith(x):
                    w = _log_w_to_w(x)
                    return -_compute_P(s_arith, w)
                
                x0 = np.log(w_arith + 1e-300)
                try:
                    res = minimize(_neg_P_log_w_arith, x0, method='L-BFGS-B',
                                 options={'maxiter': 100, 'ftol': 1e-12})
                    if res.fun < -best_P:
                        w_opt = _log_w_to_w(res.x)
                        update_best(s_arith, w_opt)
                except Exception:
                    pass

    # --- Final Output ---
    if best_support is None:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
    
    # Clean up: remove points with negligible weight
    mask = best_weights > 1e-10
    best_support = best_support[mask]
    best_weights = best_weights[mask]
    
    # Normalize
    best_weights = best_weights / best_weights.sum()
    
    # Validation
    assert len(best_support) == len(best_weights)
    assert len(best_support) >= 1
    assert np.all(np.isfinite(best_support))
    assert np.all(np.isfinite(best_weights))
    assert np.all(best_support >= -1e-9)
    assert np.all(best_weights >= -1e-9)
    assert np.sum(best_weights) > 1e-9
    
    return best_support.astype(np.float64), best_weights.astype(np.float64)

# EVOLVE-BLOCK-END
