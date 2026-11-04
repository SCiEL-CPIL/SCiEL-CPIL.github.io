import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
DH = 1.0 / 20.0
OFFSET = 10

_DEADLINE = 150.0
_SAFETY = 10.0
_start = time.time()

def _elapsed():
    return time.time() - _start

def _time_left():
    return _DEADLINE - _SAFETY - _elapsed()

def _autoconv(h):
    """Discrete autoconvolution sampled on the same grid (length N)."""
    h = np.asarray(h, dtype=float)
    full = np.convolve(h, h, mode='full')
    out = full[OFFSET:OFFSET + N]
    return DH * out

def _ratio(h):
    """Return the objective value R = ||h||^2_2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return 0.0
    ac = _autoconv(h)
    acmax = float(np.max(ac))
    if acmax <= 0.0:
        return 0.0
    return l2sq / (l1 * acmax)

def _find_sidon_set(n, k_target, rng):
    """
    Try to find a Sidon set of size k_target in {0, ..., n-1}.
    Returns the set of indices or None.
    """
    # Greedy approach with random restarts/orderings
    best_set = None
    for _ in range(20):
        order = np.arange(n)
        rng.shuffle(order)
        current_set = []
        sums = set()
        for idx in order:
            # Check if adding idx causes collision
            # New sums: idx + y for y in current_set, and idx + idx
            valid = True
            new_sums = []
            for y in current_set:
                s = idx + y
                if s in sums:
                    valid = False
                    break
                new_sums.append(s)
            s_diag = idx + idx
            if s_diag in sums:
                valid = False
            else:
                new_sums.append(s_diag)
            
            if valid:
                current_set.append(idx)
                sums.update(new_sums)
                if len(current_set) >= k_target:
                    if best_set is None or len(current_set) > len(best_set):
                        best_set = current_set[:]
                    break # Found one of target size, stop this trial
        
        if best_set is not None and len(best_set) >= k_target:
            break
            
    return best_set

def _optimize_heights(support, time_budget):
    """
    Optimize the heights on the given support to maximize the ratio.
    Uses log-sum-exp for smooth max.
    Parameterization: h_i = exp(y_i) / sum(exp(y_j)) for i in support.
    This ensures h_i > 0 and sum(h) = 1.
    Objective: minimize -log(||h||_2^2) + log(LSE(autoconv(h)))
    """
    K = len(support)
    if K == 0:
        return None, 0.0
    
    def objective(y):
        # y is vector of log-heights
        # Normalize to sum 1
        # Use log-sum-exp for stability
        max_y = np.max(y)
        exp_y = np.exp(y - max_y)
        sum_exp_y = np.sum(exp_y)
        h_support = exp_y / sum_exp_y
        
        # Construct full h
        h = np.zeros(N)
        h[support] = h_support
        
        l2sq = float(np.dot(h, h))
        if l2sq <= 0:
            return 1e12, np.zeros_like(y)
        
        ac = _autoconv(h)
        
        # Log-sum-exp of ac to approximate log(max(ac))
        # We want log(max(ac))
        # max(ac) = exp(LSE(ac)) where LSE is log-sum-exp
        # But ac values are small. 
        # Let's just use a high-temperature LSE to approximate max.
        # Or simply: if we know the active index, we can use it.
        # For smoothness, use LSE with temperature T.
        T = 0.01
        ac_clipped = np.maximum(ac, 1e-12)
        log_ac = np.log(ac_clipped)
        max_log_ac = np.max(log_ac)
        lse = max_log_ac + np.log(np.sum(np.exp((log_ac - max_log_ac) / T)) * T)
        
        val = -np.log(l2sq) + lse
        return val, np.zeros_like(y) # Gradient computed analytically below? 
        # L-BFGS-B can use finite differences if jac not provided, but it's slow.
        # Let's provide a gradient if possible, or just use BFGS which estimates it.
        # For 50 vars, finite difference is 50 evals per step. 
        # 500 steps * 50 = 25000 evals. 
        # Each eval is fast (convolution). 
        # Should be fine.

    def objective_with_grad(y):
        max_y = np.max(y)
        exp_y = np.exp(y - max_y)
        sum_exp_y = np.sum(exp_y)
        h_support = exp_y / sum_exp_y
        
        h = np.zeros(N)
        h[support] = h_support
        
        l2sq = float(np.dot(h, h))
        if l2sq <= 0:
            return 1e12, np.zeros_like(y)
        
        ac = _autoconv(h)
        
        # Gradient of log(max(ac)) using active set approximation
        idx_max = int(np.argmax(ac))
        ac_max = ac[idx_max]
        if ac_max <= 0:
            return 1e12, np.zeros_like(y)
            
        # grad log(l2sq) = -2*h / l2sq
        # grad log(ac_max) = grad(ac_max) / ac_max
        # ac_max = sum_{j+k=idx_max+OFFSET} h_j h_k
        # grad(ac_max) wrt h_i = 2 * h_{idx_max+OFFSET-i} (if valid)
        
        M = OFFSET + idx_max
        grad_ac = np.zeros(N)
        i_start = max(0, M - (N - 1))
        i_end = min(N, M + 1)
        if i_start < i_end:
            i_range = np.arange(i_start, i_end)
            j_range = M - i_range
            grad_ac[i_range] = 2.0 * h[j_range]
            
        grad_l2 = -2.0 * h / l2sq
        grad_ac_norm = grad_ac / ac_max
        
        grad_h = grad_l2 + grad_ac_norm
        
        # Map grad_h to grad_y
        # h_i = exp(y_i) / S, where S = sum(exp(y))
        # dh_i/dy_j = h_i (delta_ij - h_j)
        # grad_y_j = sum_i grad_h_i * dh_i/dy_j
        # grad_y = h * (grad_h - sum(h * grad_h))
        
        grad_y = h_support * (grad_h[support] - np.dot(h_support, grad_h[support]))
        
        val = -np.log(l2sq) + np.log(ac_max)
        return val, grad_y

    y0 = np.zeros(K)
    
    try:
        res = minimize(
            objective_with_grad, y0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        y_opt = res.x
        # Convert back to h
        max_y = np.max(y_opt)
        exp_y = np.exp(y_opt - max_y)
        h_support = exp_y / np.sum(exp_y)
        h = np.zeros(N)
        h[support] = h_support
        
        # Normalize to unit L2 norm for final ratio calculation
        norm = np.linalg.norm(h)
        if norm > 0:
            h = h / norm
            v = _ratio(h)
        else:
            v = 0.0
        return h, v
    except Exception:
        return None, 0.0

def construct():
    global _start
    _start = time.time()
    
    best_h = np.zeros(N)
    best_v = 0.0
    
    rng = np.random.default_rng(42)
    
    # Try Sidon sets of increasing size
    # Max size is around sqrt(50) ~ 7
    for k in [8, 7, 6, 5]:
        if _time_left() < 10.0:
            break
            
        support = _find_sidon_set(N, k, rng)
        if support is None:
            continue
            
        # Optimize heights for this support
        h_opt, v_opt = _optimize_heights(support, time_budget=30.0)
        if h_opt is not None and v_opt > best_v:
            best_v = v_opt
            best_h = h_opt.copy()
            
    # Also try a few random supports of size 5-7 to see if non-Sidon helps
    for _ in range(5):
        if _time_left() < 5.0:
            break
        k = rng.integers(5, 8)
        support = np.sort(rng.choice(N, size=k, replace=False))
        h_opt, v_opt = _optimize_heights(support, time_budget=10.0)
        if h_opt is not None and v_opt > best_v:
            best_v = v_opt
            best_h = h_opt.copy()

    # Fallback: single spike
    if best_v <= 1.0:
        h_single = np.zeros(N)
        h_single[0] = 1.0
        best_h = h_single
        best_v = 1.0

    return [float(x) for x in best_h]

if __name__ == '__main__':
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _ratio(h))
    print("time =", time.time() - t0)
