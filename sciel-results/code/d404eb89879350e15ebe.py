# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, minimize_scalar

def _compute_P_fast(support, weights):
    """
    Fast exact computation of P[X1+X2+X3 < 2*X4] using sorting and binary search.
    Complexity: O(n^2 log n).
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    total = w.sum()
    if total <= 1e-9:
        return 0.0
    p = w / total
    n = len(s)
    if n == 0:
        return 0.0

    # Pairwise sums and probabilities
    pair_sums = s[:, None] + s[None, :]
    pair_probs = p[:, None] * p[None, :]

    flat_sums = pair_sums.flatten()
    flat_probs = pair_probs.flatten()

    sort_idx = np.argsort(flat_sums)
    sorted_sums = flat_sums[sort_idx]
    sorted_probs = flat_probs[sort_idx]
    cum_probs = np.cumsum(sorted_probs)

    # thresholds[k, l] = 2*s[l] - s[k]
    thresholds = 2.0 * s[None, :] - s[:, None]

    # count of elements strictly < threshold
    indices = np.searchsorted(sorted_sums, thresholds, side='left')

    sum_probs = np.zeros_like(indices, dtype=np.float64)
    mask_pos = indices > 0
    safe_indices = np.where(mask_pos, indices - 1, 0)
    safe_indices = np.clip(safe_indices, 0, len(cum_probs) - 1)
    sum_probs[mask_pos] = cum_probs[safe_indices[mask_pos]]

    P = np.sum(p[:, None] * p[None, :] * sum_probs)
    return float(P)

def _coordinate_descent_polish_weights(s, w, max_rounds=50):
    """
    Coordinate descent polish for weights on a fixed support.
    For each weight w_i, optimize it while keeping others fixed.
    """
    n = len(s)
    w = w.copy()
    for _ in range(max_rounds):
        improved = False
        for i in range(n):
            # Fix all weights except w_i, optimize w_i
            # w_i = t, others scaled to sum to 1-t
            # P(t) is a function of t
            def P_of_t(t):
                if t < -1e-15 or t > 1 + 1e-15:
                    return -1.0
                t = np.clip(t, 0.0, 1.0)
                if t >= 1.0 - 1e-15:
                    # All mass on w_i
                    new_w = np.zeros(n)
                    new_w[i] = 1.0
                    return -_compute_P_fast(s, new_w)
                # Scale others
                others_sum = 1.0 - t
                if others_sum < 1e-15:
                    return -1.0
                new_w = w.copy()
                new_w[i] = t
                other_idx = [j for j in range(n) if j != i]
                if len(other_idx) > 0:
                    new_w[other_idx] = w[other_idx] * (others_sum / w[other_idx].sum() if w[other_idx].sum() > 1e-15 else 1.0)
                return -_compute_P_fast(s, new_w)
            
            try:
                res = minimize_scalar(P_of_t, bounds=(0.0, 1.0), method='bounded',
                                      options={'xatol': 1e-12, 'maxiter': 200})
                t_opt = res.x
                if t_opt < 1e-15 or t_opt > 1.0 - 1e-15:
                    continue
                others_sum = 1.0 - t_opt
                if others_sum < 1e-15:
                    continue
                new_w = w.copy()
                new_w[i] = t_opt
                other_idx = [j for j in range(n) if j != i]
                if len(other_idx) > 0 and w[other_idx].sum() > 1e-15:
                    new_w[other_idx] = w[other_idx] * (others_sum / w[other_idx].sum())
                else:
                    new_w[other_idx] = others_sum / len(other_idx) if len(other_idx) > 0 else 0.0
                new_w = np.maximum(new_w, 0.0)
                new_w = new_w / new_w.sum()
                if -res.fun > _compute_P_fast(s, w) + 1e-14:
                    w = new_w
                    improved = True
            except Exception:
                pass
        if not improved:
            break
    return w

def _optimize_weights_for_support(s, z0=None, maxiter=2000):
    """Optimize weights (via softmax) for a fixed support."""
    n = len(s)
    if z0 is None:
        z0 = np.zeros(n)

    def objective_softmax(z):
        exp_z = np.exp(z - np.max(z))
        w = exp_z / exp_z.sum()
        return -_compute_P_fast(s, w)

    try:
        res = minimize(objective_softmax, z0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        z_opt = res.x
        exp_z = np.exp(z_opt - np.max(z_opt))
        w_opt = exp_z / exp_z.sum()
        P_opt = -res.fun
        
        # Coordinate descent polish
        w_cd = _coordinate_descent_polish_weights(s, w_opt, max_rounds=30)
        P_cd = _compute_P_fast(s, w_cd)
        if P_cd > P_opt:
            w_opt = w_cd
            P_opt = P_cd
            
        return s, w_opt, P_opt
    except Exception:
        return None, None, 0.0

def _polish_weights(s, w0, maxiter=5000):
    """High-precision polish of weights for a fixed support using very tight tolerances."""
    n = len(s)
    # Start from log-softmax parameterization
    w_safe = np.maximum(w0, 1e-15)
    z0 = np.log(w_safe)

    def objective_softmax(z):
        exp_z = np.exp(z - np.max(z))
        w = exp_z / exp_z.sum()
        return -_compute_P_fast(s, w)

    try:
        res = minimize(objective_softmax, z0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
        z_opt = res.x
        exp_z = np.exp(z_opt - np.max(z_opt))
        w_opt = exp_z / exp_z.sum()
        P_opt = -res.fun
        
        # Coordinate descent polish
        w_cd = _coordinate_descent_polish_weights(s, w_opt, max_rounds=50)
        P_cd = _compute_P_fast(s, w_cd)
        if P_cd > P_opt:
            w_opt = w_cd
            P_opt = P_cd
            
        return w_opt, P_opt
    except Exception:
        return w0, _compute_P_fast(s, w0)

def _joint_optimize(s_init, w_init, maxiter=2000):
    """Jointly optimize support and weights. Support = cumulative positive steps."""
    n = len(s_init)
    steps = np.diff(s_init)
    if np.any(steps <= 0):
        steps = np.maximum(np.abs(steps), 0.01)
    log_steps = np.log(steps)
    log_w = np.log(np.maximum(w_init, 1e-10))
    params = np.concatenate([log_steps, log_w])

    def objective(params):
        log_s = params[:n - 1]
        log_w = params[n - 1:]
        s = np.zeros(n)
        for i in range(1, n):
            s[i] = s[i - 1] + np.exp(log_s[i - 1])
        max_log_w = np.max(log_w)
        exp_w = np.exp(log_w - max_log_w)
        w = exp_w / np.sum(exp_w)
        return -_compute_P_fast(s, w)

    try:
        res = minimize(objective, params, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        log_s = res.x[:n - 1]
        log_w = res.x[n - 1:]
        s = np.zeros(n)
        for i in range(1, n):
            s[i] = s[i - 1] + np.exp(log_s[i - 1])
        max_log_w = np.max(log_w)
        exp_w = np.exp(log_w - max_log_w)
        w = exp_w / np.sum(exp_w)
        P = -res.fun
        return s, w, P
    except Exception:
        return None, None, 0.0

def _polish_joint(s_init, w_init, maxiter=5000):
    """High-precision joint polish with very tight tolerances."""
    n = len(s_init)
    steps = np.diff(s_init)
    if np.any(steps <= 0):
        steps = np.maximum(np.abs(steps), 0.001)
    log_steps = np.log(steps)
    log_w = np.log(np.maximum(w_init, 1e-15))
    params = np.concatenate([log_steps, log_w])

    def objective(params):
        log_s = params[:n - 1]
        log_w = params[n - 1:]
        s = np.zeros(n)
        for i in range(1, n):
            s[i] = s[i - 1] + np.exp(log_s[i - 1])
        max_log_w = np.max(log_w)
        exp_w = np.exp(log_w - max_log_w)
        w = exp_w / np.sum(exp_w)
        return -_compute_P_fast(s, w)

    try:
        res = minimize(objective, params, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
        log_s = res.x[:n - 1]
        log_w = res.x[n - 1:]
        s = np.zeros(n)
        for i in range(1, n):
            s[i] = s[i - 1] + np.exp(log_s[i - 1])
        max_log_w = np.max(log_w)
        exp_w = np.exp(log_w - max_log_w)
        w = exp_w / np.sum(exp_w)
        P = -res.fun
        return s, w, P
    except Exception:
        return s_init, w_init, _compute_P_fast(s_init, w_init)

def _differential_evolution_search(n, bounds_s, bounds_w, maxiter=200, popsize=10, seed=42):
    """
    Use differential evolution to jointly optimize support and weights.
    Support is constrained to be non-negative and sorted.
    Weights are constrained to be non-negative.
    """
    lb = np.concatenate([np.full(n, -6.9), np.full(n, -20.0)])
    ub = np.concatenate([np.full(n, 4.6), np.full(n, 20.0)])
    
    def objective(params):
        log_steps = params[:n]
        log_w = params[n:]
        
        steps = np.exp(log_steps)
        s = np.cumsum(steps)
        
        max_log_w = np.max(log_w)
        exp_w = np.exp(log_w - max_log_w)
        w = exp_w / np.sum(exp_w)
        
        if np.any(s < -1e-9) or np.any(w < -1e-9) or w.sum() <= 1e-9:
            return 1.0
        
        P = _compute_P_fast(s, w)
        return -P
    
    try:
        res = differential_evolution(objective, list(zip(lb, ub)), 
                                     maxiter=maxiter, popsize=popsize, 
                                     seed=seed, tol=1e-8, atol=1e-10,
                                     polish=True, init='sobol')
        
        log_steps = res.x[:n]
        log_w = res.x[n:]
        
        steps = np.exp(log_steps)
        s = np.cumsum(steps)
        
        max_log_w = np.max(log_w)
        exp_w = np.exp(log_w - max_log_w)
        w = exp_w / np.sum(exp_w)
        
        P = -res.fun
        return s, w, P
    except Exception:
        return None, None, 0.0

def _pareto_like_support(n, alpha=2.0, scale=1.0):
    """Generate Pareto-like support: heavy-tailed distribution."""
    # Pareto distribution: P(X > x) = (x_m / x)^alpha for x >= x_m
    # CDF: F(x) = 1 - (x_m/x)^alpha
    # Inverse CDF: x = x_m / (1-u)^(1/alpha)
    u = np.linspace(0.01, 0.99, n)
    s = scale / (1 - u) ** (1.0 / alpha)
    s = np.sort(s)
    return s

def _lognormal_like_support(n, mu=0.0, sigma=1.0):
    """Generate log-normal-like support."""
    u = np.linspace(0.01, 0.99, n)
    # Standard normal quantile approximation
    z = np.sqrt(2) * np.erfinv(2 * u - 1)
    s = np.exp(mu + sigma * z)
    s = np.sort(s)
    return s

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 600.0 - 30.0  # 30s safety margin

    best_P = 0.0
    best_support = np.array([0.0, 1.0])
    best_weights = np.array([0.5, 0.5])
    
    # 1. Analytical 2-point solution as a strong baseline
    q_opt = (15.0 - np.sqrt(33.0)) / 16.0
    s_2pt = np.array([0.0, 1.0])
    w_2pt = np.array([q_opt, 1.0 - q_opt])
    P_2pt = _compute_P_fast(s_2pt, w_2pt)
    if P_2pt > best_P:
        best_P = P_2pt
        best_support = s_2pt.copy()
        best_weights = w_2pt.copy()

    # 2. Differential Evolution for small n (3, 4, 5)
    for n in [3, 4, 5, 6]:
        if time.time() > deadline - 60.0:
            break
        
        t_start = time.time()
        s_de, w_de, P_de = _differential_evolution_search(n, None, None, maxiter=50, popsize=10, seed=42)
        
        if s_de is not None and P_de > best_P:
            best_P = P_de
            best_support = s_de.copy()
            best_weights = w_de.copy()
        
        if s_de is not None and P_de > best_P * 0.9:
            if time.time() < deadline - 30.0:
                s_ref, w_ref, P_ref = _joint_optimize(s_de, w_de, maxiter=1000)
                if s_ref is not None and P_ref > best_P:
                    best_P = P_ref
                    best_support = s_ref.copy()
                    best_weights = w_ref.copy()

    # 3. Explore diverse support structures with weight optimization
    supports_to_try = []
    
    # Geometric progressions
    for n in [3, 4, 5, 6, 8]:
        for base in [1.5, 2.0, 3.0, 5.0, 10.0]:
            s = np.array([base**i for i in range(n)], dtype=np.float64)
            supports_to_try.append(s)
    
    # Arithmetic progressions
    for n in [3, 4, 5, 6, 8]:
        for step in [0.5, 1.0, 2.0, 5.0]:
            s = np.arange(n, dtype=np.float64) * step
            supports_to_try.append(s)
    
    # Mixed: small cluster + large jumps
    for n in [5, 6, 8]:
        for k in range(2, min(n, 5)):
            s = np.zeros(n)
            s[:k] = np.arange(k, dtype=np.float64)
            for j in range(k, n):
                s[j] = 10.0 * (j - k + 1)
            supports_to_try.append(s)
    
    # Exponential-like: 0, 1, 2, 4, 8, ...
    for n in [4, 5, 6]:
        s = np.array([0.0] + [2.0**i for i in range(1, n)], dtype=np.float64)
        supports_to_try.append(s)
    
    # Log-space structures: s_i = exp(i * alpha)
    for n in [4, 5, 6, 8]:
        for alpha in [0.3, 0.5, 0.7, 1.0, 1.5]:
            s = np.exp(np.arange(n, dtype=np.float64) * alpha)
            supports_to_try.append(s)
    
    # Power-law: s_i = (i+1)^p
    for n in [4, 5, 6, 8]:
        for p in [0.5, 1.0, 1.5, 2.0, 3.0]:
            s = (np.arange(1, n+1, dtype=np.float64)) ** p
            supports_to_try.append(s)
    
    # NEW: Pareto-like supports (heavy-tailed)
    for n in [3, 4, 5, 6, 8]:
        for alpha in [1.5, 2.0, 2.5, 3.0, 4.0]:
            for scale in [1.0, 2.0, 5.0]:
                s = _pareto_like_support(n, alpha=alpha, scale=scale)
                supports_to_try.append(s)
    
    # NEW: Log-normal-like supports
    for n in [3, 4, 5, 6, 8]:
        for sigma in [0.5, 1.0, 1.5, 2.0]:
            for mu in [-1.0, 0.0, 1.0]:
                s = _lognormal_like_support(n, mu=mu, sigma=sigma)
                supports_to_try.append(s)
    
    # Try each support with weight optimization
    candidates = []
    for s in supports_to_try:
        if time.time() > deadline - 120.0:
            break
        
        n_s = len(s)
        initial_zs = [
            np.zeros(n_s),  # uniform
            np.concatenate([np.log([1.5]), np.zeros(n_s-1)]),  # biased toward 0
            np.log(np.linspace(1.0, 0.3, n_s)),  # decreasing
            np.concatenate([np.zeros(n_s-1), np.log([1.5])]),  # biased toward large
        ]
        
        for z0 in initial_zs:
            if time.time() > deadline - 100.0:
                break
            s_opt, w_opt, P_opt = _optimize_weights_for_support(s, z0, maxiter=1500)
            if s_opt is not None and P_opt > best_P:
                best_P = P_opt
                best_support = s_opt.copy()
                best_weights = w_opt.copy()
                candidates.append((P_opt, s_opt.copy(), w_opt.copy()))
            elif s_opt is not None and P_opt > best_P * 0.95:
                candidates.append((P_opt, s_opt.copy(), w_opt.copy()))
    
    # 4. Joint optimization on the best candidates from step 2
    if time.time() < deadline - 60.0:
        candidates.sort(key=lambda x: -x[0])
        top_candidates = candidates[:5] if len(candidates) >= 5 else candidates
        
        for P_c, s_c, w_c in top_candidates:
            if time.time() > deadline - 30.0:
                break
            
            s_j, w_j, P_j = _joint_optimize(s_c, w_c, maxiter=2000)
            if s_j is not None and P_j > best_P:
                best_P = P_j
                best_support = s_j.copy()
                best_weights = w_j.copy()
    
    # 5. Random support exploration in remaining time
    if time.time() < deadline - 30.0:
        np.random.seed(42)
        for _ in range(15):
            if time.time() > deadline - 10.0:
                break
            
            n = np.random.randint(3, 10)
            scale = np.random.uniform(1.0, 20.0)
            s_raw = np.sort(np.random.uniform(0, scale, n))
            for i in range(1, n):
                if s_raw[i] - s_raw[i-1] < 0.01:
                    s_raw[i] = s_raw[i-1] + 0.01
            
            s_opt, w_opt, P_opt = _optimize_weights_for_support(s_raw, np.zeros(n), maxiter=1000)
            if s_opt is not None and P_opt > best_P:
                best_P = P_opt
                best_support = s_opt.copy()
                best_weights = w_opt.copy()
    
    # 6. Final high-precision polish pass
    if time.time() < deadline - 10.0:
        w_pol, P_pol = _polish_weights(best_support, best_weights, maxiter=5000)
        if P_pol > best_P:
            best_P = P_pol
            best_weights = w_pol.copy()
        
        if time.time() < deadline - 5.0:
            s_pol, w_pol2, P_pol2 = _polish_joint(best_support, best_weights, maxiter=3000)
            if s_pol is not None and P_pol2 > best_P:
                best_P = P_pol2
                best_support = s_pol.copy()
                best_weights = w_pol2.copy()
    
    # Ensure validity
    if len(best_support) != len(best_weights):
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
        
    if np.any(best_support < -1e-9):
        best_support = np.abs(best_support)
        
    if np.any(best_weights < -1e-9):
        best_weights = np.abs(best_weights)
        
    total_w = best_weights.sum()
    if total_w <= 1e-9:
        best_weights = np.ones(len(best_support)) / len(best_support)
    else:
        best_weights = best_weights / total_w
        
    return best_support, best_weights

# EVOLVE-BLOCK-END
