import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 100
TARGET_SUM = N / 2.0

def _score(h):
    """Compute the maximum overlap of h with shifted (1-h)."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _fix_sum(h):
    """Project h to have sum exactly N/2 while staying in [0,1]."""
    h = h.copy()
    for _ in range(100):
        diff = TARGET_SUM - np.sum(h)
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            # Need to increase sum
            slack_up = 1.0 - h
            order = np.argsort(-slack_up)
            for i in order:
                if abs(diff) < 1e-14:
                    break
                amt = min(slack_up[i], diff)
                h[i] += amt
                diff -= amt
        else:
            # Need to decrease sum
            slack_down = h
            order = np.argsort(-slack_down)
            for i in order:
                if abs(diff) < 1e-14:
                    break
                amt = min(slack_down[i], -diff)
                h[i] -= amt
                diff += amt
    # Final exact correction
    diff = TARGET_SUM - np.sum(h)
    if abs(diff) > 1e-15:
        idx = np.argmin(h) if diff > 0 else np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h

def _smooth_max(x, alpha=200.0):
    """Smooth approximation of max(x)."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _overlap_and_grad(h, alpha):
    """Compute overlap vector and its gradient w.r.t. h."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    
    # Gradient of overlap_i w.r.t. h
    # overlap_i = h_i * (1 - h_{i+1}) for i < n-1
    # overlap_{n-1} = h_{n-1}
    
    grad = np.zeros(n)
    for i in range(n):
        if i < n - 1:
            grad[i] += (1.0 - h[i+1])
            grad[i+1] += (-h[i])
        else:
            grad[i] += 1.0
    
    # Smooth max gradient
    x_max = np.max(overlap)
    e = np.exp(alpha * (overlap - x_max))
    p = e / np.sum(e)
    
    grad_smooth = p * grad
    return overlap, grad_smooth

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s kill
    
    h_best = np.full(N, 0.5)
    score_best = _score(h_best)
    
    # Strategy 1: Piecewise-linear interpolation with Nelder-Mead
    # Represent h as piecewise-linear interpolation of K knots
    # Knot positions are fixed (evenly spaced), only values are optimized
    
    K = 20  # Number of knots
    knot_positions = np.linspace(0, N-1, K)
    
    def knots_to_h(knot_values):
        """Interpolate knot values to full h vector."""
        h = np.interp(np.arange(N), knot_positions, knot_values)
        return h
    
    def knots_objective(knot_values):
        h = knots_to_h(knot_values)
        h = np.clip(h, 0.0, 1.0)
        h = _fix_sum(h)
        return _score(h)
    
    # Global search over knot values
    bounds_knots = [(0.0, 1.0)] * K
    
    if time.time() < deadline - 40:
        try:
            # Try multiple random initializations
            best_knots = None
            best_knots_score = float('inf')
            
            for trial in range(5):
                if time.time() > deadline - 40:
                    break
                
                if trial == 0:
                    # Start with alternating high-low pattern
                    knot_init = np.array([0.8 if i % 2 == 0 else 0.2 for i in range(K)])
                else:
                    knot_init = np.random.uniform(0.0, 1.0, K)
                
                result = minimize(
                    knots_objective,
                    knot_init,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                
                h_candidate = knots_to_h(result.x)
                h_candidate = np.clip(h_candidate, 0.0, 1.0)
                h_candidate = _fix_sum(h_candidate)
                sc = _score(h_candidate)
                
                if sc < best_knots_score:
                    best_knots_score = sc
                    best_knots = result.x.copy()
                
                if sc < score_best:
                    score_best = sc
                    h_best = h_candidate.copy()
        except Exception:
            pass
    
    # Strategy 2: Local optimization with projected gradient descent
    if time.time() < deadline - 20:
        h = h_best.copy()
        
        for alpha in [100.0, 200.0, 400.0]:
            if time.time() > deadline - 20:
                break
            
            lr = 0.05
            stuck = 0
            for step in range(1000):
                if time.time() > deadline - 20:
                    break
                
                h_clipped = np.clip(h, 0.0, 1.0)
                h_clipped = _fix_sum(h_clipped)
                
                overlap, grad_smooth = _overlap_and_grad(h_clipped, alpha)
                
                # Projected gradient step
                h_new = h_clipped - lr * grad_smooth
                h_new = np.clip(h_new, 0.0, 1.0)
                h_new = _fix_sum(h_new)
                
                sc_new = _score(h_new)
                if sc_new < score_best - 1e-12:
                    score_best = sc_new
                    h_best = h_new.copy()
                    h = h_new.copy()
                    stuck = 0
                    lr *= 0.995
                else:
                    stuck += 1
                    lr *= 0.9
                    if stuck > 100 or lr < 1e-8:
                        break
    
    # Strategy 3: Fourier-based refinement with more modes
    x = np.arange(N)
    
    def fourier_h(params, K):
        h = np.full(N, 0.5)
        for k in range(1, K + 1):
            h += params[2*(k-1)] * np.cos(2 * np.pi * k * x / N)
            h += params[2*(k-1)+1] * np.sin(2 * np.pi * k * x / N)
        return h
    
    def fourier_objective(params, K):
        h = fourier_h(params, K)
        h = np.clip(h, 0.0, 1.0)
        h = _fix_sum(h)
        return _score(h)
    
    for K in [5, 8, 12]:
        if time.time() > deadline - 10:
            break
        
        n_params = 2 * K
        # Initialize from current best
        h_init = h_best
        coeffs = np.zeros(n_params)
        for k in range(1, K + 1):
            cos_k = np.cos(2 * np.pi * k * x / N)
            sin_k = np.sin(2 * np.pi * k * x / N)
            coeffs[2*(k-1)] = np.dot(h_init - 0.5, cos_k) * 2 / N
            coeffs[2*(k-1)+1] = np.dot(h_init - 0.5, sin_k) * 2 / N
        
        try:
            result = minimize(
                lambda p: fourier_objective(p, K),
                coeffs,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            h_candidate = fourier_h(result.x, K)
            h_candidate = np.clip(h_candidate, 0.0, 1.0)
            h_candidate = _fix_sum(h_candidate)
            sc = _score(h_candidate)
            if sc < score_best:
                score_best = sc
                h_best = h_candidate.copy()
        except Exception:
            pass
    
    # Final verification
    h_best = np.clip(h_best, 0.0, 1.0)
    h_best = _fix_sum(h_best)
    
    return h_best
