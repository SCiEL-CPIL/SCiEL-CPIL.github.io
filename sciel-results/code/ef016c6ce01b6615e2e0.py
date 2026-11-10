import numpy as np
import time

def _compute_P(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] for a discrete measure.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    total_w = w.sum()
    if total_w <= 1e-12:
        return 0.0
    w = w / total_w
    
    S = s[:, None] + s[None, :]
    T = 2.0 * s[None, :] - s[:, None]
    W = w[:, None] * w[None, :]
    
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
    
    P = np.sum(W * F)
    return float(P)

def _project_to_simplex(v):
    """Project vector v onto the probability simplex."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    k = np.max(ind[cond])
    theta = cssv[k - 1] / k
    return np.maximum(v - theta, 0)

def _compute_P_and_F(support, weights):
    """Compute P and the F matrix."""
    n = len(support)
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    total_w = w.sum()
    if total_w <= 1e-12:
        return 0.0, np.zeros((n, n))
    w = w / total_w
    
    S = s[:, None] + s[None, :]
    T = 2.0 * s[None, :] - s[:, None]
    W = w[:, None] * w[None, :]
    
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
    
    P = np.sum(W * F)
    return float(P), F

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 600.0
    
    best_support = None
    best_weights = None
    best_score = -1.0
    
    # Try different grid sizes
    for n_grid in [50, 100, 200]:
        if time.time() > deadline - 60.0:
            break
        
        # Create a grid of support points
        s_grid = np.linspace(0, 1, n_grid)
        
        # Initialize weights
        w = np.ones(n_grid) / n_grid
        
        # Projected gradient ascent on weights
        lr = 0.1
        for step in range(500):
            if time.time() > deadline - 30.0:
                break
            
            P, F = _compute_P_and_F(s_grid, w)
            
            # Gradient: dP/dw_m = sum_l w_l * F[m,l] + sum_k w_k * F[k,m]
            grad = np.sum(w * F, axis=1) + np.sum(w * F.T, axis=1)
            
            # Update weights
            w_new = w + lr * grad
            w = _project_to_simplex(w_new)
            
            # Check convergence
            if step % 50 == 0:
                score = _compute_P(s_grid, w)
                if score > best_score:
                    best_score = score
                    best_support = s_grid.copy()
                    best_weights = w.copy()
        
        # Final evaluation
        score = _compute_P(s_grid, w)
        if score > best_score:
            best_score = score
            best_support = s_grid.copy()
            best_weights = w.copy()
    
    # Prune small weights
    if best_support is not None:
        mask = best_weights > 1e-6
        best_support = best_support[mask]
        best_weights = best_weights[mask]
        
        if len(best_support) == 0:
            best_support = np.array([0.0, 1.0])
            best_weights = np.array([0.5, 0.5])
        else:
            best_weights = best_weights / best_weights.sum()
    else:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
    
    return best_support.astype(np.float64), best_weights.astype(np.float64)
