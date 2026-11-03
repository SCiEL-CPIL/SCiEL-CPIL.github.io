"""Erdos minimum overlap: choose h in [0,1]^N, sum = N/2, minimize max_k sum_i min(h_i, 1-h_{i-k}) * (2/N).

We use a continuous optimization approach with SLSQP and gradient-based methods.
The key insight is that the overlap function O(k) = sum_i min(h_i, 1-h_{i-k}) * (2/N)
is piecewise linear in h, so we can use a smooth approximation (soft-min) for the gradient,
and then polish with exact evaluation.
"""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 100
_DEADLINE = 150.0 - 15.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _compute_O(h, k):
    """Compute overlap at shift k: sum_i min(h_i, 1-h_{i-k}) * (2/N)"""
    h_shifted = np.roll(h, k)
    mins = np.minimum(h, 1.0 - h_shifted)
    return np.sum(mins) * (2.0 / N)

def _max_overlap(h):
    """Compute maximum overlap over all shifts."""
    max_ov = 0.0
    for k in range(N):
        ov = _compute_O(h, k)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _soft_max_overlap(h, beta=50.0):
    """Soft approximation of max overlap for gradient computation."""
    overlaps = np.array([_compute_O(h, k) for k in range(N)])
    # Soft max using log-sum-exp trick
    m = np.max(overlaps)
    if m < 1e-10:
        return 0.0
    return m + np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta

def _soft_max_grad(h, beta=50.0):
    """Gradient of soft max overlap."""
    overlaps = np.array([_compute_O(h, k) for k in range(N)])
    m = np.max(overlaps)
    exps = np.exp(beta * (overlaps - m))
    weights = exps / np.sum(exps)
    
    grad = np.zeros(N)
    for k in range(N):
        h_shifted = np.roll(h, k)
        # d/dh_i of min(h_i, 1-h_{i-k})
        # If h_i < 1-h_{i-k}: derivative w.r.t h_i is 1
        # If h_i > 1-h_{i-k}: derivative w.r.t h_i is 0
        # At equality: 0.5
        mask_lt = h < (1.0 - h_shifted)
        mask_gt = h > (1.0 - h_shifted)
        dmin = np.where(mask_lt, 1.0, np.where(mask_gt, 0.0, 0.5))
        
        # Also need derivative w.r.t h_{i-k} which affects other indices
        # d/dh_j of min(h_i, 1-h_{i-k}) where j = i-k
        # = -1 if h_i > 1-h_{i-k}, else 0
        # So for each i, the contribution to h_{i-k} is -d(1-h_{i-k})/dh_{i-k} * indicator
        # Let's just compute the full gradient numerically or analytically
        
        # Actually, let's compute gradient of O(k) w.r.t h:
        # O(k) = sum_i min(h_i, 1-h_{(i-k) mod N}) * (2/N)
        # dO(k)/dh_j = (2/N) * [sum_{i: i-k=j} d min(h_i, 1-h_{i-k})/dh_j]
        #            = (2/N) * [d min(h_{j+k}, 1-h_j)/dh_j]  (only one i contributes for each j)
        # d min(h_{j+k}, 1-h_j)/dh_j = -1 if h_{j+k} > 1-h_j, else 0
        # Wait, that's not right either. Let me think more carefully.
        
        # O(k) = (2/N) * sum_i min(h_i, 1-h_{(i-k) mod N})
        # For a fixed j, h_j appears in:
        #   - term i=j: min(h_j, 1-h_{(j-k) mod N})
        #   - term i=j+k: min(h_{j+k}, 1-h_j)  [since (i-k) mod N = j => i = j+k]
        # 
        # d/dh_j of min(h_j, 1-h_{j-k}):
        #   if h_j < 1-h_{j-k}: 1
        #   if h_j > 1-h_{j-k}: 0
        #   else: 0.5
        #
        # d/dh_j of min(h_{j+k}, 1-h_j):
        #   if h_{j+k} < 1-h_j: 0  (since 1-h_j is the second arg, and we're differentiating w.r.t h_j)
        #     wait: min(a, b) where b = 1-h_j, db/dh_j = -1
        #     if a < b: d min = 0
        #     if a > b: d min = db/dh_j = -1
        #     else: 0.5 * (-1) = -0.5
        #
        # So total dO(k)/dh_j = (2/N) * [I(h_j < 1-h_{j-k}) * 1 + I(h_{j+k} > 1-h_j) * (-1)]
        #                     = (2/N) * [I(h_j < 1-h_{j-k}) - I(h_{j+k} > 1-h_j)]
        
        j_idx = np.arange(N)
        jp = (j_idx + k) % N
        jm = (j_idx - k) % N
        
        # Term 1: I(h_j < 1-h_{j-k})
        t1 = (h[j_idx] < (1.0 - h[jm])).astype(float)
        # Term 2: I(h_{j+k} > 1-h_j)
        t2 = (h[jp] > (1.0 - h[j_idx])).astype(float)
        
        dOk = (2.0 / N) * (t1 - t2)
        grad += weights[k] * dOk
    
    return grad

def _project(h):
    """Project h onto the constraint set: 0 <= h_i <= 1, sum(h) = N/2."""
    h = np.clip(h, 0.0, 1.0)
    # Adjust to satisfy sum constraint
    diff = N/2.0 - np.sum(h)
    if abs(diff) > 1e-10:
        # Distribute the difference
        if diff > 0:
            # Need to increase some elements
            mask = h < 1.0
            if np.sum(mask) > 0:
                h[mask] += diff / np.sum(mask)
        else:
            # Need to decrease some elements
            mask = h > 0.0
            if np.sum(mask) > 0:
                h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    # Final adjustment to ensure exact sum
    diff = N/2.0 - np.sum(h)
    if abs(diff) > 1e-12:
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _objective(h):
    """Objective function: soft max overlap."""
    return _soft_max_overlap(h, beta=30.0)

def _gradient(h):
    """Gradient of objective."""
    return _soft_max_grad(h, beta=30.0)

def _optimize_gradient(h0, max_iter=500, lr=0.01):
    """Gradient descent with projection."""
    h = h0.copy()
    best_h = h.copy()
    best_val = _max_overlap(h)
    
    for it in range(max_iter):
        if _time_left() < 10.0:
            break
        
        grad = _gradient(h)
        h_new = h - lr * grad
        h_new = _project(h_new)
        
        val = _max_overlap(h_new)
        if val < best_val:
            best_val = val
            best_h = h_new.copy()
        
        h = h_new
        # Adaptive learning rate
        if it % 100 == 0:
            lr *= 0.95
    
    return best_h, best_val

def _optimize_slsqp(h0, max_iter=200):
    """SLSQP optimization."""
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0, 'jac': lambda h: np.ones(N)},
    ]
    bounds = [(0.0, 1.0)] * N
    
    try:
        res = minimize(
            _objective, h0, method='SLSQP',
            jac=_gradient, bounds=bounds, constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-8}
        )
        h_opt = _project(res.x)
        val = _max_overlap(h_opt)
        return h_opt, val
    except Exception:
        return h0, _max_overlap(h0)

def _polish(h, max_iters=1000):
    """Local search polish: try small perturbations."""
    best_h = h.copy()
    best_val = _max_overlap(h)
    rng = np.random.default_rng(12345)
    
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        
        # Try a small random perturbation
        h_new = best_h.copy()
        idx = rng.integers(0, N)
        delta = rng.uniform(-0.1, 0.1)
        h_new[idx] += delta
        h_new = _project(h_new)
        
        val = _max_overlap(h_new)
        if val < best_val - 1e-8:
            best_val = val
            best_h = h_new.copy()
    
    return best_h, best_val

def construct():
    """Main entry point: construct the optimal h."""
    best_h = None
    best_val = float('inf')
    
    # Multiple restarts with different initializations
    rng = np.random.default_rng(42)
    
    # Restart 1: Golden ratio construction
    alpha = (np.sqrt(5) - 1) / 2.0
    x = (np.arange(N) * alpha) % 1.0
    indices = np.argsort(x)
    h0 = np.zeros(N)
    h0[indices[:N//2]] = 1.0
    h0 = _project(h0)
    
    h_opt, val = _optimize_gradient(h0, max_iter=300, lr=0.005)
    if val < best_val:
        best_val = val
        best_h = h_opt.copy()
    
    # Restart 2: Random binary
    if _time_left() > 30.0:
        h0 = np.zeros(N)
        h0[rng.choice(N, N//2, replace=False)] = 1.0
        h0 = _project(h0)
        
        h_opt, val = _optimize_gradient(h0, max_iter=300, lr=0.005)
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()
    
    # Restart 3: SLSQP from golden ratio
    if _time_left() > 30.0:
        alpha = (np.sqrt(5) - 1) / 2.0
        x = (np.arange(N) * alpha) % 1.0
        indices = np.argsort(x)
        h0 = np.zeros(N)
        h0[indices[:N//2]] = 1.0
        h0 = _project(h0)
        
        h_opt, val = _optimize_slsqp(h0, max_iter=100)
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()
    
    # Final polish
    if _time_left() > 10.0 and best_h is not None:
        best_h, best_val = _polish(best_h, max_iters=500)
    
    if best_h is None:
        # Fallback: uniform
        best_h = np.full(N, 0.5)
    
    return best_h
