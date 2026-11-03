"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses Projected Gradient Descent with Barzilai-Borwein step sizes and 
adaptive smooth-max parameter. 
Projection onto the simplex-with-boxes is done via efficient water-filling.
"""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 115  # Safety margin


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _smooth_max_val_grad(h, beta):
    """
    Smooth maximum of the linear overlap values and its gradient.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    # Avoid overflow
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights
    
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    for k in range(n):
        if w[k] == 0.0:
            continue
        factor = -(2.0 / n) * w[k]
        if k > 0:
            grad[k:] += factor * h[:n-k]
        if k < n:
            grad[:n-k] += factor * h[k:]
            
    return val, grad


def _project(h, target_sum):
    """
    Project h onto {x | sum(x) = target_sum, 0 <= x_i <= 1}.
    Uses bisection on the shift parameter tau.
    x_i = clip(h_i + tau, 0, 1)
    Find tau such that sum(x_i) = target_sum.
    """
    h = np.asarray(h, dtype=np.float64)
    # Lower and upper bounds for tau
    # If tau is very small, sum is close to sum(clip(h, 0, 1)) which might be < target
    # If tau is very large, sum is close to N
    # We want sum(clip(h+tau, 0, 1)) = target_sum
    
    def sum_proj(tau):
        return np.sum(np.clip(h + tau, 0.0, 1.0))
    
    # Initial bounds
    lo = -max(np.max(h), 1.0) - 1.0
    hi = 1.0 - min(np.min(h), 0.0) + 1.0
    
    # Ensure bracketing
    if sum_proj(lo) > target_sum:
        lo -= 10.0
    if sum_proj(hi) < target_sum:
        hi += 10.0
        
    # Bisection
    for _ in range(50):
        mid = (lo + hi) / 2.0
        s = sum_proj(mid)
        if s < target_sum:
            lo = mid
        else:
            hi = mid
            
    tau = (lo + hi) / 2.0
    h_proj = np.clip(h + tau, 0.0, 1.0)
    
    # Final correction for floating point precision
    diff = target_sum - np.sum(h_proj)
    if abs(diff) > 1e-12:
        # Adjust the element closest to boundary? No, adjust the one with most room.
        # Simply distribute among non-saturated elements
        mask = (h_proj > 1e-9) & (h_proj < 1.0 - 1e-9)
        if np.any(mask):
            h_proj[mask] += diff / np.sum(mask)
            h_proj = np.clip(h_proj, 0.0, 1.0)
            
    return h_proj


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    target_sum = N / 2.0

    def _update_best(h):
        nonlocal best_h, best_score
        h = _project(h, target_sum)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    def _optimize(h0, max_iter=500, beta_start=10.0, beta_end=500.0):
        nonlocal best_h, best_score
        h = _project(h0.copy(), target_sum)
        h_prev = h.copy()
        g_prev = None
        beta = beta_start
        
        for i in range(max_iter):
            if time.time() > _DEADLINE - 10:
                break
                
            # Animate beta
            beta = beta_start * (beta_end / beta_start) ** (i / max_iter)
            
            val, grad = _smooth_max_val_grad(h, beta)
            
            # Check time
            if time.time() > _DEADLINE - 5:
                break
                
            # Barzilai-Borwein Step Size
            if g_prev is not None:
                s_vec = h - h_prev
                y_vec = grad - g_prev
                s_norm_sq = np.dot(s_vec, s_vec)
                y_norm_sq = np.dot(y_vec, y_vec)
                sy = np.dot(s_vec, y_vec)
                
                if s_norm_sq > 1e-12 and y_norm_sq > 1e-12:
                    # BB+ or BB-
                    alpha_bb1 = s_norm_sq / sy if sy > 1e-12 else 1.0
                    alpha_bb2 = sy / y_norm_sq if y_norm_sq > 1e-12 else 1.0
                    alpha = 0.5 * (alpha_bb1 + alpha_bb2)
                    if alpha < 1e-6:
                        alpha = 1e-6
                    if alpha > 10.0:
                        alpha = 10.0
                else:
                    alpha = 1.0
            else:
                alpha = 1.0 / (np.linalg.norm(grad) + 1e-6)
                alpha = min(alpha, 0.1)

            # Update
            h_new = h - alpha * grad
            h_new = _project(h_new, target_sum)
            
            # Backtracking line search if necessary (simple version)
            # Since we project, we might move away from objective improvement.
            # Check if score improved
            score_new = _score(h_new)
            if score_new > _score(h) * 1.01 + 1e-6:
                # Try smaller step
                alpha = alpha * 0.5
                h_new = h - alpha * grad
                h_new = _project(h_new, target_sum)
                
            h_prev = h
            h = h_new
            g_prev = grad
            
            _update_best(h)
            
            # Convergence check
            if np.linalg.norm(h - h_prev) < 1e-8:
                break

    # 1. Cosine Start
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    _optimize(h_cos)
    
    # 2. Random Starts
    if time.time() - t0 < _DEADLINE - 20:
        for _ in range(5):
            if time.time() - t0 > _DEADLINE - 10:
                break
            h_rand = np.random.uniform(0, 1, N)
            _optimize(h_rand)
            
    # 3. Step Function Starts
    if time.time() - t0 < _DEADLINE - 10:
        h_step = np.zeros(N)
        h_step[:N//2] = 1.0
        _optimize(h_step)
        
        h_step2 = np.zeros(N)
        h_step2[::2] = 1.0
        h_step2 = _project(h_step2, target_sum)
        _optimize(h_step2)

    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _project(best_h, target_sum)
    return best_h
