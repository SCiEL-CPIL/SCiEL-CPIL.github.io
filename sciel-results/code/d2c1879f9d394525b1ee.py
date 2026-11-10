import numpy as np
import time
import sys
from scipy.optimize import minimize

def evaluate_measure(s, w):
    """
    Fast vectorized evaluation of P[X1 + X2 + X3 < 2*X4].
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
    # threshold_{k,l} = 2*s_l - s_k
    thresholds = 2.0 * s[None, :] - s[:, None]
    
    # Find indices where ss_sorted < threshold
    # searchsorted with side='left' gives the first index where ss_sorted >= threshold
    # We want the sum of pp_sorted for indices < idx
    idx = np.searchsorted(ss_sorted, thresholds, side='left')
    
    # Get cumulative probabilities up to idx-1
    # Handle idx=0 case
    prob_inner = np.zeros_like(idx, dtype=np.float64)
    mask = idx > 0
    prob_inner[mask] = C[idx[mask] - 1]
    
    P = np.sum(p[:, None] * p[None, :] * prob_inner)
    
    return P

def precompute_support(s):
    """
    Precompute structures for a fixed support to speed up repeated evaluations.
    """
    n = len(s)
    s = np.maximum(s, 0.0)
    
    ss = s[:, None] + s[None, :]
    ss_flat = ss.flatten()
    
    sort_idx = np.argsort(ss_flat)
    ss_sorted = ss_flat[sort_idx]
    
    return {
        'n': n,
        's': s,
        'ss_sorted': ss_sorted,
        'sort_idx': sort_idx
    }

def eval_with_precomp(precomp, p):
    """
    Fast evaluation using precomputed support structure.
    p must be normalized (sums to 1).
    """
    n = precomp['n']
    s = precomp['s']
    ss_sorted = precomp['ss_sorted']
    sort_idx = precomp['sort_idx']
    
    # Build pp in the sorted order
    pp = p[:, None] * p[None, :]
    pp_flat = pp.flatten()
    pp_sorted = pp_flat[sort_idx]
    
    C = np.cumsum(pp_sorted)
    
    # Vectorized thresholds
    thresholds = 2.0 * s[None, :] - s[:, None]
    idx = np.searchsorted(ss_sorted, thresholds, side='left')
    
    prob_inner = np.zeros_like(idx, dtype=np.float64)
    mask = idx > 0
    prob_inner[mask] = C[idx[mask] - 1]
    
    P = np.sum(p[:, None] * p[None, :] * prob_inner)
    
    return P

def objective_and_grad(x, precomp, time_budget):
    """
    Objective function and gradient for scipy.optimize.minimize.
    x = [log_w_0, ..., log_w_{n-1}, s_0, ..., s_{n-1}]
    Returns (loss, grad)
    """
    n = precomp['n']
    log_w = x[:n]
    s = x[n:]
    
    # Enforce s >= 0
    s_clipped = np.maximum(s, 0.0)
    
    # Softmax for weights
    log_w_max = np.max(log_w)
    exp_w = np.exp(log_w - log_w_max)
    w = exp_w / np.sum(exp_w)
    
    # Update precomp with new s
    ss = s_clipped[:, None] + s_clipped[None, :]
    ss_flat = ss.flatten()
    sort_idx = np.argsort(ss_flat)
    ss_sorted = ss_flat[sort_idx]
    
    pp = w[:, None] * w[None, :]
    pp_flat = pp.flatten()
    pp_sorted = pp_flat[sort_idx]
    C = np.cumsum(pp_sorted)
    
    thresholds = 2.0 * s_clipped[None, :] - s_clipped[:, None]
    idx = np.searchsorted(ss_sorted, thresholds, side='left')
    
    prob_inner = np.zeros_like(idx, dtype=np.float64)
    mask = idx > 0
    prob_inner[mask] = C[idx[mask] - 1]
    
    P = np.sum(w[:, None] * w[None, :] * prob_inner)
    
    loss = -P
    
    # Gradient computation (approximate via finite differences if needed, but let's try analytic or simple FD)
    # For speed and simplicity in this context, we might rely on scipy's numerical gradient if analytic is too complex
    # However, let's try to compute it.
    # dP/ds_k and dP/dw_k
    
    # Actually, implementing the full analytic gradient for this complex structure is error-prone and time-consuming.
    # Given the time limit, using `method='L-BFGS-B'` or `Nelder-Mead` with numerical gradients might be safer.
    # But numerical gradients for 2n dimensions are expensive.
    # Let's stick to a robust numerical gradient approach or a simpler method.
    
    # Let's use Nelder-Mead which doesn't require gradients and is robust.
    # Or, we can just return the loss and let scipy handle gradients if we use a method that supports it.
    # For this implementation, I will return just the loss and use Nelder-Mead or Powell.
    
    return loss

def optimize_joint(s_init, w_init, time_limit, seed=42):
    """
    Optimize support and weights jointly.
    """
    n = len(s_init)
    s_init = np.maximum(s_init, 0.0)
    w_init = np.maximum(w_init, 1e-9)
    w_init = w_init / np.sum(w_init)
    
    log_w_init = np.log(w_init)
    x0 = np.concatenate([log_w_init, s_init])
    
    # Use Nelder-Mead as it is derivative-free and robust
    # We need a wrapper that handles the time limit and returns the best found
    best_x = x0.copy()
    best_loss = objective_and_grad(x0, None, 0)[0]
    
    start_time = time.time()
    
    def wrapped_obj(x):
        nonlocal best_x, best_loss
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        
        n = len(s_init)
        log_w = x[:n]
        s = x[n:]
        s = np.maximum(s, 0.0)
        
        log_w_max = np.max(log_w)
        exp_w = np.exp(log_w - log_w_max)
        w = exp_w / np.sum(exp_w)
        
        # Evaluate
        precomp = precompute_support(s)
        P = eval_with_precomp(precomp, w)
        loss = -P
        
        if loss < best_loss:
            best_loss = loss
            best_x = x.copy()
            
        return loss

    try:
        res = minimize(wrapped_obj, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
    except TimeoutError:
        pass
    except Exception:
        pass

    # Extract best
    log_w_best = best_x[:n]
    s_best = best_x[n:]
    s_best = np.maximum(s_best, 0.0)
    
    log_w_max = np.max(log_w_best)
    exp_w = np.exp(log_w_best - log_w_max)
    w_best = exp_w / np.sum(exp_w)
    
    # Final evaluation
    precomp = precompute_support(s_best)
    P_best = eval_with_precomp(precomp, w_best)
    
    return s_best, w_best, P_best

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    deadline = time.time() + 600.0
    safety_margin = 30.0
    
    best_support = None
    best_weights = None
    best_score = -1.0
    
    # Generate a diverse set of initial supports
    # Focus on structures that allow for "heavy tail" X4 and "small sum" X1+X2+X3
    # e.g., points clustered near 0 and points spread out
    
    initializations = []
    
    # 1. Geometric progression with a point at 0
    for n in [4, 6, 8, 10, 12]:
        for r in [1.5, 2.0, 3.0, 5.0]:
            s = np.zeros(n)
            s[0] = 0.0
            s[1:] = r ** np.arange(1, n)
            initializations.append((s, np.ones(n)/n))
            
    # 2. Bimodal: cluster at 0, cluster at large values
    for n1 in [2, 3, 4]:
        for n2 in [2, 3, 4]:
            n = n1 + n2
            s1 = np.linspace(0, 0.5, n1)
            s2 = np.linspace(5, 20, n2)
            s = np.concatenate([s1, s2])
            w = np.ones(n)/n
            initializations.append((s, w))
            
    # 3. Linear with a gap
    for n in [5, 8, 10]:
        s = np.arange(n, dtype=np.float64)
        s[n//2:] += 10.0
        w = np.ones(n)/n
        initializations.append((s, w))
        
    # 4. Random sparse supports
    rng = np.random.default_rng(42)
    for _ in range(10):
        n = rng.integers(4, 12)
        s = rng.uniform(0, 20, n)
        s = np.sort(s)
        s[0] = 0.0 # Ensure 0 is in support for small sums
        w = rng.dirichlet(np.ones(n))
        initializations.append((s, w))
        
    # Sort initializations by initial score to prioritize good starting points
    scored_inits = []
    for s, w in initializations:
        s = np.maximum(s, 0.0)
        w = np.maximum(w, 1e-9)
        w = w / np.sum(w)
        score = evaluate_measure(s, w)
        scored_inits.append((score, s, w))
        
    scored_inits.sort(key=lambda x: -x[0])
    
    # Allocate time to top initializations
    # We will run joint optimization on the top few
    num_to_optimize = min(len(scored_inits), 15)
    
    # Time allocation: give more time to higher scoring initializations
    total_time_budget = deadline - safety_margin - time.time()
    
    # Exponential decay of time allocation
    time_weights = [0.7 ** i for i in range(num_to_optimize)]
    tw_sum = sum(time_weights)
    time_weights = [tw / tw_sum * total_time_budget for tw in time_weights]
    time_weights = [max(5.0, tw) for tw in time_weights] # Minimum 5s per optimization
    
    for i in range(num_to_optimize):
        if time.time() + 10.0 > deadline - safety_margin:
            break
            
        score_seed, s_seed, w_seed = scored_inits[i]
        
        if score_seed > best_score:
            best_score = score_seed
            best_support = s_seed.copy()
            best_weights = w_seed.copy()
            
        time_limit = time_weights[i]
        remaining = deadline - safety_margin - time.time()
        if remaining < 10.0:
            break
        time_limit = min(time_limit, remaining - 5.0)
        
        try:
            s_opt, w_opt, score_opt = optimize_joint(s_seed, w_seed, time_limit, seed=i)
            
            if score_opt > best_score:
                best_score = score_opt
                best_support = s_opt.copy()
                best_weights = w_opt.copy()
                
        except Exception as e:
            pass
    
    # Fallback
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
