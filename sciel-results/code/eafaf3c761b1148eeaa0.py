"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_DEADLINE = 120.0  # seconds, with safety margin from 150s kill
_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _c3_value(h):
    """Compute the actual C3 upper bound: max|conv| / (sum|h|)^2"""
    conv = np.convolve(h, h, mode='full')
    max_abs_conv = np.max(np.abs(conv))
    sum_abs_h = np.sum(np.abs(h))
    if sum_abs_h < 1e-12:
        return np.inf
    return max_abs_conv / (sum_abs_h ** 2)

def _soft_max_objective(x, temperature=5.0):
    """Smooth objective using soft-max for the max term."""
    h = np.tanh(x)
    conv = np.convolve(h, h, mode='full')
    abs_conv = np.abs(conv)
    # Soft max approximation
    T = temperature
    # Numerical stability: subtract max
    max_val = np.max(abs_conv)
    log_sum_exp = max_val + T * np.log(np.sum(np.exp((abs_conv - max_val) / T)))
    sum_abs_h = np.sum(np.abs(h))
    if sum_abs_h < 1e-12:
        return 1e10
    value = log_sum_exp / (sum_abs_h ** 2)
    return value

def _discrete_refine(h, max_iter=200):
    """Discrete local search: flip signs to improve the objective."""
    h = h.copy()
    current_val = _c3_value(h)
    for _ in range(max_iter):
        if _elapsed() > _DEADLINE - 5:
            break
        best_improvement = 0
        best_idx = -1
        # Try flipping each position
        for i in range(N):
            h[i] = -h[i]
            val = _c3_value(h)
            h[i] = -h[i]  # restore
            if val < current_val - 1e-12:
                improvement = current_val - val
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_idx = i
        if best_idx >= 0:
            h[best_idx] = -h[best_idx]
            current_val -= best_improvement
        else:
            break
    return h, current_val

def _c3_upper_bound(h_init):
    """
    Optimize the signed step function to minimize max|conv|/(sum|h|)^2.
    Uses continuous optimization followed by discrete refinement.
    """
    global _start_time
    _start_time = time.time()
    
    best_h = h_init.copy()
    best_val = _c3_value(best_h)
    
    # Try multiple initializations
    initializations = []
    
    # 1. All positive (seed)
    initializations.append(np.ones(N))
    
    # 2. Alternating signs
    alt = np.ones(N)
    alt[1::2] = -1
    initializations.append(alt)
    
    # 3. Block pattern: positive then negative
    block = np.ones(N)
    block[N//2:] = -1
    initializations.append(block)
    
    # 4. Random with seed for reproducibility
    rng = np.random.RandomState(42)
    for _ in range(3):
        initializations.append(rng.choice([-1, 1], size=N))
    
    # 5. Smoothed random
    for _ in range(2):
        smooth = rng.randn(N)
        smooth = np.tanh(smooth * 0.5)
        initializations.append(smooth)
    
    for h0 in initializations:
        if _elapsed() > _DEADLINE - 10:
            break
        
        # Continuous optimization
        x0 = np.arctanh(np.clip(h0, -0.999, 0.999))
        
        def objective(x):
            return _soft_max_objective(x, temperature=3.0)
        
        try:
            res = minimize(
                objective, x0, method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h_cont = np.tanh(res.x)
        except Exception:
            h_cont = h0
        
        # Evaluate continuous result
        val_cont = _c3_value(h_cont)
        
        # Discrete refinement
        h_refined, val_refined = _discrete_refine(h_cont, max_iter=100)
        
        # Also try refining the original
        h_refined0, val_refined0 = _discrete_refine(h0, max_iter=50)
        
        # Keep the best
        candidates = [(h_cont, val_cont), (h_refined, val_refined), (h_refined0, val_refined0)]
        for hc, vc in candidates:
            if vc < best_val:
                best_val = vc
                best_h = hc.copy()
    
    # Final polish: more iterations of discrete refinement
    best_h, best_val = _discrete_refine(best_h, max_iter=300)
    
    return best_h

def construct():
    h_seed = np.ones(N)
    return _c3_upper_bound(h_seed)
# EVOLVE-BLOCK-END
