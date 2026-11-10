import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _compute_P(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] for a discrete measure.
    support: 1D array of shape (n,)
    weights: 1D array of shape (n,)
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    # Ensure normalized
    total_w = w.sum()
    if total_w <= 1e-12:
        return 0.0
    w = w / total_w
    
    # 1. Pair sums S_ij = s_i + s_j
    S = s[:, None] + s[None, :]
    # 2. Thresholds T_kl = 2*s_l - s_k
    T = 2.0 * s[None, :] - s[:, None]
    
    # Weights matrix W = w w^T
    W = w[:, None] * w[None, :]
    
    # --- Compute F_kl ---
    # F_kl = sum_{i,j: S_ij < T_kl} W_ij
    S_flat = S.flatten()
    T_flat = T.flatten()
    W_flat = W.flatten()
    
    order_S = np.argsort(S_flat, kind='stable')
    S_sorted = S_flat[order_S]
    W_sorted_S = W_flat[order_S]
    cum_W_S = np.cumsum(W_sorted_S)
    
    idxs_F = np.searchsorted(S_sorted, T_flat, side='left')
    padded_cum_W_S = np.concatenate(([0.0], cum_W_S))
    F_flat = padded_cum_W_S[idxs_F]
    F = F_flat.reshape(n, n)
    
    # --- Compute P ---
    # P = sum_{k,l} w_k w_l F_kl
    P = np.sum(W * F)
    
    return float(P)

def _objective_de(x, n):
    """
    Objective function for Differential Evolution.
    x is flattened [s_0, ..., s_{n-1}, w_0, ..., w_{n-1}]
    Returns -P (to minimize)
    """
    s = x[:n]
    w = x[n:]
    
    # Clip weights to non-negative
    w = np.maximum(w, 0)
    if w.sum() <= 1e-12:
        return 1.0 # Penalty
    
    # Normalize weights for calculation
    w_norm = w / w.sum()
    
    # Clip support to [0, 1] just in case DE goes out of bounds slightly
    s = np.clip(s, 0, 1)
    
    P = _compute_P(s, w_norm)
    return -P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 590.0 # Safety margin
    
    best_support = None
    best_weights = None
    best_score = -1.0
    
    # Try different numbers of points
    # DE is slow, so we can't try too many n or too many iterations
    n_values = [5, 10, 15]
    
    for n in n_values:
        if time.time() > deadline - 30.0:
            break
            
        # Bounds for s: [0, 1], for w: [0, 1]
        # We will normalize w afterwards
        bounds = [(0, 1)] * n + [(0, 1)] * n
        
        # Run Differential Evolution
        # maxiter and popsize control the search
        # tol and atol control convergence
        try:
            res = differential_evolution(
                _objective_de,
                bounds,
                args=(n,),
                maxiter=100,
                popsize=15,
                tol=1e-5,
                seed=42,
                polish=False # We will polish manually
            )
            
            if time.time() > deadline - 10.0:
                break
                
            x_opt = res.x
            s_opt = x_opt[:n]
            w_opt = x_opt[n:]
            
            # Refine with L-BFGS-B
            # We need to handle the constraint sum(w)=1 and w>=0
            # L-BFGS-B supports bounds and simple constraints? No, only bounds.
            # We can parameterize w using a softmax or just project?
            # Simpler: Just use Powell on the original parameters with bounds.
            # Or, just accept the DE result if it's good enough, as DE with polish=False is already decent.
            # Let's try a quick local refinement with Powell.
            
            # Powell doesn't handle constraints well either, but bounds are okay.
            # The issue is the simplex constraint for w.
            # Let's stick to the DE result, maybe with a few Nelder-Mead steps.
            
            # Actually, let's just normalize and check.
            w_opt = np.maximum(w_opt, 0)
            if w_opt.sum() > 0:
                w_opt = w_opt / w_opt.sum()
            else:
                continue
                
            s_opt = np.sort(s_opt)
            # Re-sort weights to match sorted support? 
            # The DE didn't enforce order. We should sort s and correspondingly sort w?
            # No, s and w are paired. If we sort s, we must sort w by the index of s.
            idx = np.argsort(s_opt)
            s_opt = s_opt[idx]
            w_opt = w_opt[idx]
            
            score = _compute_P(s_opt, w_opt)
            
            if score > best_score:
                best_score = score
                best_support = s_opt.copy()
                best_weights = w_opt.copy()
                
        except Exception as e:
            print(f"DE failed for n={n}: {e}")
            continue
            
    # If DE didn't find anything good, or if we have time, try some random restarts with local optimization
    if time.time() < deadline - 60.0:
        # Try n=10 with random restarts and SLSQP (like before but more restarts)
        n = 10
        bounds = [(0, 1)] * n + [(0, 1)] * n
        constraints = [{'type': 'eq', 'fun': lambda x, n=n: x[n:].sum() - 1.0}]
        
        for r in range(20):
            if time.time() > deadline - 10.0:
                break
                
            s_init = np.sort(np.random.uniform(0, 1, n))
            w_init = np.random.dirichlet(np.ones(n))
            x0 = np.concatenate([s_init, w_init])
            
            def obj_slsqp(x, n=n):
                s = x[:n]
                w = x[n:]
                w = np.maximum(w, 0)
                if w.sum() <= 1e-12:
                    return 1.0
                w = w / w.sum()
                s = np.clip(s, 0, 1)
                return -_compute_P(s, w)
            
            try:
                res = minimize(
                    obj_slsqp, 
                    x0, 
                    args=(n,), 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=constraints,
                    options={'maxiter': 100, 'ftol': 1e-8}
                )
                
                if res.success:
                    s_opt = res.x[:n]
                    w_opt = res.x[n:]
                    w_opt = np.maximum(w_opt, 0)
                    if w_opt.sum() > 0:
                        w_opt = w_opt / w_opt.sum()
                    else:
                        continue
                        
                    idx = np.argsort(s_opt)
                    s_opt = s_opt[idx]
                    w_opt = w_opt[idx]
                    
                    score = _compute_P(s_opt, w_opt)
                    if score > best_score:
                        best_score = score
                        best_support = s_opt.copy()
                        best_weights = w_opt.copy()
                        
            except Exception:
                pass

    # Fallback
    if best_support is None:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
        
    # Prune small weights
    mask = best_weights > 1e-6
    best_support = best_support[mask]
    best_weights = best_weights[mask]
    
    if len(best_support) == 0:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
    else:
        best_weights = best_weights / best_weights.sum()
        
    return best_support.astype(np.float64), best_weights.astype(np.float64)
