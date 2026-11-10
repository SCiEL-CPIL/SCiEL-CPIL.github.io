# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _compute_P_fast(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] exactly in float64.
    Uses sorting and searchsorted for O(n^2 log n) efficiency.
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    n = len(s)
    if n == 0:
        return -np.inf
    if len(s) != len(w):
        return -np.inf
    if np.any(s < -1e-9) or np.any(w < -1e-9):
        return -np.inf
    if np.any(np.isnan(s)) or np.any(np.isinf(s)) or np.any(np.isnan(w)) or np.any(np.isinf(w)):
        return -np.inf
        
    total = w.sum()
    if total <= 1e-9:
        return -np.inf
        
    p = w / total
    
    # Pair sums and probabilities
    pair_sums = s[:, None] + s[None, :]
    pair_probs = p[:, None] * p[None, :]
    
    flat_sums = pair_sums.flatten()
    flat_probs = pair_probs.flatten()
    
    order = np.argsort(flat_sums, kind='mergesort')
    sorted_sums = flat_sums[order]
    sorted_probs = flat_probs[order]
    
    cum_probs = np.cumsum(sorted_probs)
    
    # Thresholds: 2*s[l] - s[k]
    # We need P[X1+X2 < 2*s[l] - s[k]]
    # thresholds[k, l] = 2*s[l] - s[k]
    thresholds = 2.0 * s[None, :] - s[:, None]
    flat_thresholds = thresholds.flatten()
    
    indices = np.searchsorted(sorted_sums, flat_thresholds, side='left')
    
    inner_sums = np.zeros_like(flat_thresholds)
    mask = indices > 0
    inner_sums[mask] = cum_probs[indices[mask] - 1]
    
    F_values = inner_sums.reshape(n, n)
    
    # Weight for X3 (k) and X4 (l)
    weight_matrix = p[:, None] * p[None, :]
    
    P = np.sum(weight_matrix * F_values)
    return P

def _objective_general(params, n):
    """
    Objective for general n-point distribution.
    params: [s_1, ..., s_{n-1}, w_0, ..., w_{n-1}]
    s_0 is fixed to 0.
    Returns -P (to be minimized).
    """
    s = np.zeros(n, dtype=np.float64)
    s[1:] = params[:n-1]
    
    w = params[n-1:]
    
    # Basic validity checks
    if np.any(s < -1e-9) or np.any(w < -1e-9):
        return 1e10
    if np.any(np.isnan(s)) or np.any(np.isinf(s)) or np.any(np.isnan(w)) or np.any(np.isinf(w)):
        return 1e10
        
    total_w = w.sum()
    if total_w <= 1e-9:
        return 1e10
        
    P = _compute_P_fast(s, w)
    if P == -np.inf:
        return 1e10
        
    return -P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 580.0 # Safety margin
    
    best_P = -np.inf
    best_support = None
    best_weights = None
    
    # 1. Analytic 2-point optimum
    # P(p) = (1-p)p^2(3-2p)
    # Optimal p approx 0.578
    p_opt_2 = (15 - np.sqrt(33)) / 16
    support_2 = np.array([0.0, 1.0], dtype=np.float64)
    weights_2 = np.array([p_opt_2, 1 - p_opt_2], dtype=np.float64)
    P_2 = _compute_P_fast(support_2, weights_2)
    
    if P_2 > best_P:
        best_P = P_2
        best_support = support_2.copy()
        best_weights = weights_2.copy()
        
    # 2. Optimize 3-point distribution
    def _objective_3point(params):
        a, b, p0, p1, p2 = params
        if a < 0 or b < 0 or a > b:
            return 1e10
        if p0 < 0 or p1 < 0 or p2 < 0:
            return 1e10
        total_p = p0 + p1 + p2
        if total_p <= 1e-9:
            return 1e10
        support = np.array([0.0, a, b], dtype=np.float64)
        weights = np.array([p0, p1, p2], dtype=np.float64)
        P = _compute_P_fast(support, weights)
        if P == -np.inf:
            return 1e10
        return -P

    guesses_3 = [
        [0.5, 1.0, p_opt_2 * 0.9, p_opt_2 * 0.1, 1 - p_opt_2],
        [0.2, 1.0, p_opt_2 * 0.8, p_opt_2 * 0.2, 1 - p_opt_2],
        [0.5, 2.0, p_opt_2 * 0.7, p_opt_2 * 0.3, 1 - p_opt_2],
        [1.0, 3.0, 0.5, 0.3, 0.2],
        [1.0, 5.0, 0.4, 0.4, 0.2],
        [1.0, 10.0, 0.3, 0.3, 0.4],
        [0.1, 1.0, 0.6, 0.2, 0.2],
        [1.0, 100.0, 0.5, 0.3, 0.2],
    ]
    
    for guess in guesses_3:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _objective_3point, 
                guess, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            P_val = -res.fun
            if P_val > best_P:
                a, b, p0, p1, p2 = res.x
                p0 = max(0, p0); p1 = max(0, p1); p2 = max(0, p2)
                a = max(0, a); b = max(0, b)
                if a > b: a, b = b, a
                support = np.array([0.0, a, b], dtype=np.float64)
                weights = np.array([p0, p1, p2], dtype=np.float64)
                P_check = _compute_P_fast(support, weights)
                if P_check != -np.inf and P_check > best_P:
                    best_P = P_check
                    best_support = support.copy()
                    best_weights = weights.copy()
        except Exception:
            pass

    # 3. Optimize 4-point distribution
    def _objective_4point(params):
        a, b, c, p0, p1, p2, p3 = params
        if a < 0 or b < 0 or c < 0 or a > b or b > c:
            return 1e10
        if p0 < 0 or p1 < 0 or p2 < 0 or p3 < 0:
            return 1e10
        total_p = p0 + p1 + p2 + p3
        if total_p <= 1e-9:
            return 1e10
        support = np.array([0.0, a, b, c], dtype=np.float64)
        weights = np.array([p0, p1, p2, p3], dtype=np.float64)
        P = _compute_P_fast(support, weights)
        if P == -np.inf:
            return 1e10
        return -P

    if best_support is not None and len(best_support) == 3:
        s0, s1, s2 = best_support
        w0, w1, w2 = best_weights
        guesses_4 = [
            [s1*0.5, s1, s2, w0*0.5, w0*0.5, w1, w2],
            [s1, s2*0.5, s2, w0, w1*0.5, w1*0.5, w2],
            [s1, s2, s2*2.0, w0*0.8, w1, w2*0.5, w2*0.5],
        ]
    else:
        guesses_4 = [
            [0.5, 1.0, 2.0, 0.4, 0.3, 0.2, 0.1],
            [0.5, 1.0, 3.0, 0.3, 0.3, 0.25, 0.15],
            [1.0, 2.0, 5.0, 0.4, 0.3, 0.2, 0.1],
            [1.0, 3.0, 10.0, 0.3, 0.3, 0.2, 0.2],
        ]
        
    for guess in guesses_4:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _objective_4point, 
                guess, 
                method='Nelder-Mead', 
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            P_val = -res.fun
            if P_val > best_P:
                a, b, c, p0, p1, p2, p3 = res.x
                p0 = max(0, p0); p1 = max(0, p1); p2 = max(0, p2); p3 = max(0, p3)
                a = max(0, a); b = max(0, b); c = max(0, c)
                support = np.array([0.0, a, b, c], dtype=np.float64)
                weights = np.array([p0, p1, p2, p3], dtype=np.float64)
                order = np.argsort(support)
                support = support[order]
                weights = weights[order]
                P_check = _compute_P_fast(support, weights)
                if P_check != -np.inf and P_check > best_P:
                    best_P = P_check
                    best_support = support.copy()
                    best_weights = weights.copy()
        except Exception:
            pass

    # 4. Global Optimization with Differential Evolution for larger n
    # Focus on heavy-tailed structures (Pareto/Log-normal-like)
    for n in [8, 16, 32, 64]:
        if time.time() > deadline - 100:
            break
            
        bounds_s = [(0.0, 100.0)] * (n - 1)
        bounds_w = [(0.0, 1.0)] * n
        bounds = bounds_s + bounds_w
        
        # Initial population: mix of geometric decay (heavy tail) and random
        x0_list = []
        rng = np.random.RandomState(42)
        
        # Geometric decay guess (exponential tail)
        s_init = np.zeros(n)
        for i in range(1, n):
            s_init[i] = i * 1.0
        w_init = np.exp(-np.arange(n) * 0.5)
        w_init = w_init / w_init.sum()
        x0_list.append(np.concatenate([s_init[1:], w_init]))
        
        # Power-law-like tail (heavier)
        w_init2 = 1.0 / (np.arange(n) + 1.0)**1.5
        w_init2 = w_init2 / w_init2.sum()
        x0_list.append(np.concatenate([s_init[1:], w_init2]))
        
        # Log-normal-like spread
        s_init_log = np.zeros(n)
        s_init_log[1:] = np.exp(rng.uniform(0, 3, n-1))
        s_init_log = np.sort(s_init_log)
        w_init3 = np.exp(-0.5 * (np.log(s_init_log[1:] + 1e-6) - np.mean(np.log(s_init_log[1:] + 1e-6)))**2)
        w_init3 = np.concatenate([[1.0], w_init3])
        w_init3 = w_init3 / w_init3.sum()
        x0_list.append(np.concatenate([s_init_log[1:], w_init3]))
        
        # Random guesses
        for _ in range(5):
            s_rand = np.zeros(n)
            s_rand[1:] = rng.uniform(0, 100, n-1)
            s_rand = np.sort(s_rand)
            w_rand = rng.dirichlet(np.ones(n))
            x0_list.append(np.concatenate([s_rand[1:], w_rand]))
            
        x0 = np.array(x0_list)
        
        try:
            res = differential_evolution(
                _objective_general, 
                bounds, 
                args=(n,),
                seed=42,
                maxiter=150,
                popsize=15,
                tol=1e-5,
                polish=False,
                init=x0
            )
            
            P_val = -res.fun
            if P_val > best_P:
                params = res.x
                s = np.zeros(n)
                s[1:] = params[:n-1]
                w = params[n-1:]
                w_sum = w.sum()
                if w_sum > 0:
                    w = w / w_sum
                order = np.argsort(s)
                s = s[order]
                w = w[order]
                
                P_check = _compute_P_fast(s, w)
                if P_check != -np.inf and P_check > best_P:
                    best_P = P_check
                    best_support = s.copy()
                    best_weights = w.copy()
        except Exception:
            pass

    # 5. Local refinement of the best found so far
    if best_support is not None and len(best_support) > 2 and time.time() < deadline - 20:
        n = len(best_support)
        s = best_support
        w = best_weights
        
        # Re-parameterize for minimize: [s_1, ..., s_{n-1}, w_0, ..., w_{n-1}]
        x0 = np.concatenate([s[1:], w])
        
        try:
            res = minimize(
                _objective_general, 
                x0, 
                args=(n,),
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            P_val = -res.fun
            if P_val > best_P:
                params = res.x
                s_new = np.zeros(n)
                s_new[1:] = params[:n-1]
                w_new = params[n-1:]
                w_sum = w_new.sum()
                if w_sum > 0:
                    w_new = w_new / w_sum
                order = np.argsort(s_new)
                s_new = s_new[order]
                w_new = w_new[order]
                
                P_check = _compute_P_fast(s_new, w_new)
                if P_check != -np.inf and P_check > best_P:
                    best_P = P_check
                    best_support = s_new.copy()
                    best_weights = w_new.copy()
        except Exception:
            pass

    # Fallback if no valid solution found
    if best_support is None:
        return np.array([0.0, 1.0], dtype=np.float64), np.array([p_opt_2, 1-p_opt_2], dtype=np.float64)

    # Final validity check
    if len(best_support) != len(best_weights):
        return np.array([0.0, 1.0], dtype=np.float64), np.array([p_opt_2, 1-p_opt_2], dtype=np.float64)
    if np.any(best_support < -1e-9) or np.any(best_weights < -1e-9) or np.sum(best_weights) <= 1e-9:
        return np.array([0.0, 1.0], dtype=np.float64), np.array([p_opt_2, 1-p_opt_2], dtype=np.float64)
    if np.any(np.isnan(best_support)) or np.any(np.isinf(best_support)) or np.any(np.isnan(best_weights)) or np.any(np.isinf(best_weights)):
        return np.array([0.0, 1.0], dtype=np.float64), np.array([p_opt_2, 1-p_opt_2], dtype=np.float64)

    return best_support, best_weights

# EVOLVE-BLOCK-END
