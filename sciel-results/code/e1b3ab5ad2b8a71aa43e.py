import numpy as np
import time
import sys
from scipy.optimize import differential_evolution

def evaluate_measure(s, w):
    """
    Fast evaluation of P[X1 + X2 + X3 < 2*X4].
    P = sum_{k,l} p_k * p_l * sum_{i,j: s_i+s_j < 2*s_l - s_k} p_i * p_j
    """
    n = len(s)
    if n == 0:
        return 0.0
    
    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        return 0.0
    p = w / w_sum
    p = np.maximum(p, 0.0)
    s = np.maximum(s, 0.0)
    
    # Re-normalize after clipping
    p_sum = np.sum(p)
    if p_sum <= 1e-9:
        return 0.0
    p = p / p_sum
    
    # Build pair sums and probs
    ss = s[:, None] + s[None, :]
    pp = p[:, None] * p[None, :]
    
    ss_flat = ss.flatten()
    pp_flat = pp.flatten()
    
    # Sort by sum
    sort_idx = np.argsort(ss_flat)
    ss_sorted = ss_flat[sort_idx]
    pp_sorted = pp_flat[sort_idx]
    
    # Cumulative sum of probs
    C = np.cumsum(pp_sorted)
    
    # Vectorized thresholds for all (k, l) pairs
    thresholds = 2.0 * s[None, :] - s[:, None]
    idx = np.searchsorted(ss_sorted, thresholds, side='left')
    # Get cumulative probabilities up to idx-1
    # Handle idx=0 case
    prob_inner = np.zeros_like(idx, dtype=np.float64)
    mask = idx > 0
    prob_inner[mask] = C[idx[mask] - 1]
    
    P = np.sum(p[:, None] * p[None, :] * prob_inner)
            
    return P

def get_support_from_params(params, n_points):
    """
    Generate support points from log-normal mixture parameters.
    params: [mu1, sigma1, w1, mu2, sigma2]
    w2 = 1 - w1
    """
    mu1, sigma1, w1, mu2, sigma2 = params
    
    # Ensure valid sigma
    sigma1 = max(0.1, sigma1)
    sigma2 = max(0.1, sigma2)
    
    # Ensure valid mixing weight
    w1 = np.clip(w1, 0.05, 0.95)
    w2 = 1.0 - w1
    
    # Generate quantiles
    u = np.linspace(0.01, 0.99, n_points)
    
    # Inverse CDF for log-normal mixture
    # We need to find x such that w1*Phi((ln(x)-mu1)/sigma1) + w2*Phi((ln(x)-mu2)/sigma2) = u
    # Use simple bisection or approximation
    # For speed, use a rough approximation by sampling from the mixture
    
    # Actually, for the optimizer, we can just use the quantiles of the mixture
    # by solving the equation numerically for each u
    
    # To be fast, we'll use a grid and interpolate
    # Or better, just use the properties of the distribution
    
    # Let's use a simpler approach: generate points from the mixture
    # and use them as support
    rng = np.random.default_rng(42)
    
    # Generate points from the mixture
    points = []
    for _ in range(n_points):
        if rng.random() < w1:
            x = np.exp(rng.normal(mu1, sigma1))
        else:
            x = np.exp(rng.normal(mu2, sigma2))
        points.append(x)
    
    points = np.array(points)
    points = np.sort(points)
    
    # Ensure strictly increasing
    if len(points) > 1:
        for i in range(1, len(points)):
            if points[i] <= points[i-1]:
                points[i] = points[i-1] * 1.001 + 1e-6
    
    return points

def objective(params, n_points):
    """
    Objective function for differential evolution.
    Returns negative score (since DE minimizes).
    """
    try:
        s = get_support_from_params(params, n_points)
        
        # Use uniform weights for now, or optimize them later
        w = np.ones(n_points) / n_points
        
        score = evaluate_measure(s, w)
        return -score
    except Exception:
        return 1.0  # Large penalty

def optimize_parametric_support(n_points, time_budget, seed=42):
    """
    Optimize the parametric support using differential evolution.
    """
    # Parameters: [mu1, sigma1, w1, mu2, sigma2]
    # mu1, mu2: log-mean
    # sigma1, sigma2: log-sd
    # w1: mixing weight for first component
    
    bounds = [
        (-2, 5),   # mu1
        (0.1, 3),  # sigma1
        (0.05, 0.95),  # w1
        (-2, 5),   # mu2
        (0.1, 3)   # sigma2
    ]
    
    start_time = time.time()
    
    # Use differential evolution
    result = differential_evolution(
        objective,
        bounds,
        args=(n_points,),
        seed=seed,
        maxiter=50,
        popsize=15,
        tol=1e-4,
        mutation=(0.3, 1.0),
        recombination=0.8,
        polish=True
    )
    
    if time.time() - start_time > time_budget:
        return None, None
    
    if result.success or result.fun < -0.5:
        best_params = result.x
        s = get_support_from_params(best_params, n_points)
        w = np.ones(n_points) / n_points
        score = evaluate_measure(s, w)
        return s, w, score
    
    return None, None, None

def optimize_weights_for_fixed_support(s, time_limit, seed=42):
    """
    Optimizes weights for a fixed support s using Random Restarted Hill Climbing.
    """
    n = len(s)
    if n == 0:
        return np.array([1.0]), 0.0
    
    # Precompute
    s = np.maximum(s, 0.0)
    ss = s[:, None] + s[None, :]
    ss_flat = ss.flatten()
    sort_idx = np.argsort(ss_flat)
    ss_sorted = ss_flat[sort_idx]
    
    best_w = np.ones(n) / n
    best_score = evaluate_measure(s, best_w)
    
    start_time = time.time()
    rng = np.random.default_rng(seed)
    
    max_restarts = 50
    max_steps_per_restart = 10000
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        w = rng.dirichlet(np.ones(n) * 0.5)
        current_score = evaluate_measure(s, w)
        
        if current_score > best_score:
            best_score = current_score
            best_w = w.copy()
        
        # Hill Climbing
        step_size = 0.1
        stuck_count = 0
        
        for step in range(max_steps_per_restart):
            if time.time() - start_time > time_limit:
                return best_w, best_score
            
            i1 = rng.integers(0, n)
            i2 = rng.integers(0, n)
            if i1 == i2:
                continue
                
            if w[i1] < 1e-7:
                continue
                
            max_move = w[i1] * 0.3
            move = max_move * (1 if rng.random() < 0.5 else -1)
            
            if move > 0 and w[i1] < move:
                move = w[i1] * 0.5
            if move < 0 and w[i2] < -move:
                move = w[i2] * 0.5
                
            if abs(move) < 1e-9:
                continue
                
            w_trial = w.copy()
            w_trial[i1] -= move
            w_trial[i2] += move
            
            if np.any(w_trial < -1e-9):
                continue
                
            score_trial = evaluate_measure(s, w_trial)
            
            if score_trial > current_score + 1e-12:
                w = w_trial
                current_score = score_trial
                if current_score > best_score:
                    best_score = current_score
                    best_w = w.copy()
                stuck_count = 0
                step_size *= 0.97
            else:
                stuck_count += 1
                step_size *= 1.03
                if step_size > 0.3:
                    step_size = 0.3
                if step_size < 0.005:
                    break
    
    return best_w, best_score

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    deadline = time.time() + 600.0
    safety_margin = 30.0
    
    best_support = None
    best_weights = None
    best_score = -1.0
    
    # Try different numbers of support points
    n_points_list = [8, 12, 16, 20]
    
    for n_points in n_points_list:
        if time.time() + 60.0 > deadline - safety_margin:
            break
        
        # Allocate time for this n_points
        time_budget = min(120.0, deadline - safety_margin - time.time() - 10.0)
        
        if time_budget < 10.0:
            break
        
        # Optimize parametric support
        s_param, w_param, score_param = optimize_parametric_support(n_points, time_budget, seed=n_points)
        
        if s_param is not None and score_param > best_score:
            best_score = score_param
            best_support = s_param.copy()
            best_weights = w_param.copy()
            
            # Now optimize weights for this support
            remaining = deadline - safety_margin - time.time() - 5.0
            if remaining > 10.0:
                w_opt, score_opt = optimize_weights_for_fixed_support(s_param, remaining, seed=n_points)
                if score_opt > best_score:
                    best_score = score_opt
                    best_support = s_param.copy()
                    best_weights = w_opt.copy()
    
    # Fallback: try some fixed supports
    if best_support is None or best_score < 0.5:
        fallback_supports = [
            np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]),
            np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]),
        ]
        
        for s in fallback_supports:
            if time.time() + 30.0 > deadline - safety_margin:
                break
            
            n = len(s)
            w = np.ones(n) / n
            score = evaluate_measure(s, w)
            
            if score > best_score:
                best_score = score
                best_support = s.copy()
                best_weights = w.copy()
                
                # Optimize weights
                remaining = deadline - safety_margin - time.time() - 5.0
                if remaining > 10.0:
                    w_opt, score_opt = optimize_weights_for_fixed_support(s, remaining, seed=123)
                    if score_opt > best_score:
                        best_score = score_opt
                        best_support = s.copy()
                        best_weights = w_opt.copy()
    
    # Final fallback
    if best_support is None:
        n = 10
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n) / n
    
    # Final validation and cleanup
    best_support = np.asarray(best_support, dtype=np.float64).flatten()
    best_weights = np.asarray(best_weights, dtype=np.float64).flatten()
    
    # Ensure non-negative
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    # Normalize weights
    w_sum = np.sum(best_weights)
    if w_sum > 1e-9:
        best_weights = best_weights / w_sum
    else:
        n = len(best_support)
        best_weights = np.ones(n) / n
    
    return best_support, best_weights

if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print(f"Score: {evaluate_measure(s, w)}")
    print(f"n = {len(s)}")
