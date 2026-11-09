# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] for given support and weights."""
    n = len(support)
    w = weights / weights.sum()
    s = support
    
    # Build pairwise sums s_i + s_j
    # P = sum_{k,l} w_k * w_l * (sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i * w_j)
    
    # For efficiency, compute for each (k, l) pair
    # 2*s_l - s_k is the threshold
    # We need sum of w_i*w_j over pairs where s_i + s_j < threshold
    
    # Precompute all pairwise sums and their weights
    # pairwise_sums[i,j] = s_i + s_j
    # pairwise_weights[i,j] = w_i * w_j
    
    pairwise_sums = s[:, None] + s[None, :]  # n x n
    pairwise_weights = w[:, None] * w[None, :]  # n x n
    
    # For each (k, l), threshold = 2*s_l - s_k
    # Sum pairwise_weights where pairwise_sums < threshold
    
    # Vectorized: for each (k,l), compute sum of pairwise_weights[sums < 2*s_l - s_k]
    # This is O(n^4) which is fine for small n
    
    total_P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            if threshold <= 0:
                # No pairs can have sum < 0 (since support >= 0)
                continue
            # Sum pairwise weights where pairwise_sums < threshold
            mask = pairwise_sums < threshold
            inner = np.sum(pairwise_weights[mask])
            total_P += w[k] * w[l] * inner
    
    return total_P

def _objective(params, n):
    """Objective function for optimization (negated P)."""
    weights = params[:n]
    support = params[n:]
    
    # Enforce constraints: weights > 0, support >= 0
    # Use soft constraints or transform
    w = np.exp(weights)  # ensures positive
    s = np.abs(support)  # ensures non-negative
    
    P = _compute_P(s, w)
    return -P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4].
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin
    
    # Start with a reasonable initial guess
    # Try a few different configurations
    
    best_P = 0.0
    best_support = None
    best_weights = None
    
    # Configuration 1: Two-point distribution {0, M}
    # Optimize over M and p
    for M in [1.0, 2.0, 5.0, 10.0, 50.0, 100.0]:
        for p in np.linspace(0.1, 0.9, 17):
            support = np.array([0.0, M])
            weights = np.array([p, 1.0 - p])
            P = _compute_P(support, weights)
            if P > best_P:
                best_P = P
                best_support = support.copy()
                best_weights = weights.copy()
    
    # Configuration 2: Three-point distribution {0, a, b}
    for a in [0.5, 1.0, 2.0, 5.0]:
        for b in [2.0, 5.0, 10.0, 20.0, 50.0]:
            if b <= a:
                continue
            for w0 in [0.1, 0.2, 0.3, 0.4, 0.5]:
                for w1 in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    w2 = 1.0 - w0 - w1
                    if w2 <= 0:
                        continue
                    support = np.array([0.0, a, b])
                    weights = np.array([w0, w1, w2])
                    P = _compute_P(support, weights)
                    if P > best_P:
                        best_P = P
                        best_support = support.copy()
                        best_weights = weights.copy()
    
    # Configuration 3: Use scipy optimization for 3-point distribution
    if time.time() < deadline:
        n = 3
        # Parameterize: log(weights) and support points
        # Initial guess from best found so far
        if best_support is not None and len(best_support) == 3:
            x0 = np.log(best_weights)
            x0 = np.concatenate([x0, best_support])
        else:
            x0 = np.concatenate([np.log([0.3, 0.3, 0.4]), [0.0, 1.0, 5.0]])
        
        def obj3(params):
            w_log = params[:3]
            s = params[3:]
            w = np.exp(w_log)
            s = np.abs(s)
            return -_compute_P(s, w)
        
        result = minimize(obj3, x0, method='Nelder-Mead', 
                         options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if result.fun < -best_P:
            w_log = result.x[:3]
            s = np.abs(result.x[3:])
            w = np.exp(w_log)
            P = -result.fun
            if P > best_P:
                best_P = P
                best_support = s.copy()
                best_weights = w.copy()
    
    # Configuration 4: 4-point distribution optimization
    if time.time() < deadline:
        n = 4
        # Try a structured initial guess
        for init_s in [[0.0, 0.5, 2.0, 10.0], [0.0, 1.0, 3.0, 10.0], [0.0, 1.0, 5.0, 20.0]]:
            x0 = np.concatenate([np.log(np.ones(n)/n), init_s])
            
            def obj4(params):
                w_log = params[:4]
                s = params[4:]
                w = np.exp(w_log)
                s = np.abs(s)
                return -_compute_P(s, w)
            
            result = minimize(obj4, x0, method='Nelder-Mead',
                             options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if result.fun < -best_P:
                w_log = result.x[:4]
                s = np.abs(result.x[4:])
                w = np.exp(w_log)
                P = -result.fun
                if P > best_P:
                    best_P = P
                    best_support = s.copy()
                    best_weights = w.copy()
    
    # Configuration 5: 5-point distribution
    if time.time() < deadline:
        n = 5
        x0 = np.concatenate([np.log(np.ones(n)/n), [0.0, 0.5, 1.5, 5.0, 20.0]])
        
        def obj5(params):
            w_log = params[:5]
            s = params[5:]
            w = np.exp(w_log)
            s = np.abs(s)
            return -_compute_P(s, w)
        
        result = minimize(obj5, x0, method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if result.fun < -best_P:
            w_log = result.x[:5]
            s = np.abs(result.x[5:])
            w = np.exp(w_log)
            P = -result.fun
            if P > best_P:
                best_P = P
                best_support = s.copy()
                best_weights = w.copy()
    
    # Fallback: if nothing found, use the two-point optimal
    if best_support is None:
        q = (1.0/4.0)**(1.0/3.0)
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([1.0 - q, q])
    
    # Normalize weights
    best_weights = best_weights / best_weights.sum()
    
    # Ensure validity
    assert len(best_support) == len(best_weights)
    assert len(best_support) >= 1
    assert not np.any(np.isnan(best_support))
    assert not np.any(np.isnan(best_weights))
    assert not np.any(np.isinf(best_support))
    assert not np.any(np.isinf(best_weights))
    assert np.all(best_support >= -1e-9)
    assert np.all(best_weights >= -1e-9)
    assert best_weights.sum() > 1e-9
    
    return best_support.astype(np.float64), best_weights.astype(np.float64)

# EVOLVE-BLOCK-END
