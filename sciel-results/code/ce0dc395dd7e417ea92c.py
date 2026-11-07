import numpy as np
import time
import scipy.optimize as so

N = 50

def compute_value_exact(h):
    """
    Computes the exact objective value for the given heights h.
    Used for final evaluation and verification.
    """
    h = np.asarray(h, dtype=np.float64)
    # Ensure non-negative and finite
    if np.any(h < -1e-12) or np.any(~np.isfinite(h)):
        return 0.0
    
    # Clip tiny negatives from numerical noise
    h = np.maximum(h, 0.0)
    
    g = np.convolve(h, h)
    m = len(g)
    
    # y = [0, g[0], ..., g[m-1], 0]
    # l2 = (w/3) * sum_{i=0}^m (y[i]^2 + y[i]y[i+1] + y[i+1]^2)
    # This simplifies to (w/3) * (2 * sum(g^2) + sum(g[:-1]*g[1:]))
    # Note: The prompt says w is spacing on [-0.5, 0.5] with m+2 points.
    # m = 2*N - 1 = 99. Points = 101. Spacing w = 1.0 / 100.0.
    w = 1.0 / (m + 1)
    
    g_sq_sum = np.dot(g, g)
    cross_sum = np.dot(g[:-1], g[1:]) if m > 1 else 0.0
    l2 = (w / 3.0) * (2.0 * g_sq_sum + cross_sum)
    
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    
    if norm_1 * norm_inf == 0:
        return 0.0
        
    return l2 / (norm_1 * norm_inf)

def compute_value_soft(h, beta=100.0):
    """
    Computes a smoothed objective value using soft-max for the L-infinity norm.
    This provides a differentiable surrogate for gradient-based optimization.
    """
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12) or np.any(~np.isfinite(h)):
        return 1e10 # Large penalty for invalid inputs
    
    h = np.maximum(h, 0.0)
    
    g = np.convolve(h, h)
    m = len(g)
    w = 1.0 / (m + 1)
    
    g_sq_sum = np.dot(g, g)
    cross_sum = np.dot(g[:-1], g[1:]) if m > 1 else 0.0
    l2 = (w / 3.0) * (2.0 * g_sq_sum + cross_sum)
    
    norm_1 = np.sum(g) / (m + 1)
    
    # Soft-max approximation for max(g)
    # Use log-sum-exp trick for stability
    g_max = np.max(g)
    if g_max > 0:
        # exp(beta * (g - g_max))
        exp_terms = np.exp(beta * (g - g_max))
        log_sum_exp = g_max + np.log(np.sum(exp_terms))
        soft_max = log_sum_exp / beta
    else:
        soft_max = 0.0
        
    if norm_1 * soft_max < 1e-30:
        return 1e10
        
    return -l2 / (norm_1 * soft_max)

def optimize_with_soft_max(h0, deadline, betas=None):
    """
    Runs L-BFGS-B optimization on the soft-max surrogate with a schedule of betas.
    """
    if betas is None:
        betas = [10.0, 50.0, 200.0, 1000.0, 5000.0]
        
    best_h = np.array(h0, dtype=np.float64)
    best_val = compute_value_exact(best_h)
    
    for beta in betas:
        if time.time() > deadline - 5.0:
            break
            
        def neg_soft_obj(h):
            return compute_value_soft(h, beta=beta)
            
        try:
            res = so.minimize(
                neg_soft_obj,
                best_h,
                method='L-BFGS-B',
                bounds=[(1e-10, 10.0)] * N, # Reasonable bounds for h
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30}
            )
            
            h_cand = np.maximum(res.x, 0.0)
            val_cand = compute_value_exact(h_cand)
            
            if val_cand > best_val:
                best_val = val_cand
                best_h = h_cand.copy()
        except Exception:
            pass
            
    return best_h, best_val

def coordinate_polish(h, deadline, max_rounds=50):
    """
    Simple coordinate descent for final polishing.
    """
    h = np.maximum(h.copy(), 0.0)
    best_val = compute_value_exact(h)
    
    # Adaptive step size
    step = 0.01
    
    for _ in range(max_rounds):
        if time.time() > deadline - 2.0:
            break
            
        improved = False
        for i in range(N):
            if time.time() > deadline - 2.0:
                break
                
            # Try small perturbations
            for delta in [step, -step, 0.1*step, -0.1*step]:
                trial = h.copy()
                trial[i] = max(0.0, h[i] + delta)
                v = compute_value_exact(trial)
                
                if v > best_val:
                    h = trial
                    best_val = v
                    improved = True
                    break
        
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
        else:
            # Slightly shrink step if improving to settle
            step *= 0.9 
            
    return h, best_val

def construct():
    deadline = time.time() + 145.0 # 5s safety margin
    
    # Initial guesses
    x = np.linspace(0, 1, N)
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N))
    
    # 2. Gaussian bump
    inits.append(np.exp(-((x - 0.5)**2) / 0.05))
    
    # 3. Triangular
    inits.append(np.maximum(0, 1 - 2*abs(x - 0.5)))
    
    # 4. Two peaks
    inits.append(np.exp(-((x - 0.25)**2) / 0.02) + np.exp(-((x - 0.75)**2) / 0.02))
    
    # 5. Ramp
    inits.append(x)
    
    # 6. Cosine
    inits.append(0.5 * (1 + np.cos(2 * np.pi * x)))
    
    # 7. Random smooth
    rng = np.random.RandomState(42)
    r = rng.randn(N)
    # Simple smoothing
    kernel = np.ones(5) / 5.0
    r = np.convolve(r, kernel, mode='same')
    r = np.maximum(r, 0)
    if np.max(r) > 0:
        r = r / np.max(r)
    inits.append(r)
    
    # 8. Inverse sine (flat-ish)
    inits.append(1.0 - 0.5*np.sin(np.pi * x))
    
    best_h = np.ones(N)
    best_val = compute_value_exact(best_h)
    
    # Time budget allocation
    # We have ~145 seconds.
    # Each optimization run (with soft max schedule) takes ~5-10s.
    # We can afford ~10-15 runs.
    
    # Sort inits by initial value to prioritize good starts? 
    # Or just iterate. Let's iterate.
    
    for i, h0 in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
            
        # Normalize initial guess to have max 1 for consistency
        h0 = np.asarray(h0, dtype=np.float64)
        h0 = np.maximum(h0, 0)
        if np.max(h0) > 0:
            h0 = h0 / np.max(h0)
            
        # Run soft-max optimization
        h_opt, val_opt = optimize_with_soft_max(h0, deadline - 10.0)
        
        if val_opt > best_val:
            best_val = val_opt
            best_h = h_opt.copy()
            
    # Final polish on the best solution found
    if time.time() < deadline - 5.0:
        best_h, best_val = coordinate_polish(best_h, deadline - 2.0, max_rounds=100)
        
    # One last check with very tight L-BFGS-B on the exact objective?
    # The soft-max with high beta is very close to exact.
    # Let's do a final L-BFGS-B on the exact objective if time permits.
    if time.time() < deadline - 3.0:
        def neg_exact_obj(h):
            return -compute_value_exact(h)
            
        try:
            res = so.minimize(
                neg_exact_obj,
                best_h,
                method='L-BFGS-B',
                bounds=[(0, 10.0)] * N,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            h_cand = np.maximum(res.x, 0.0)
            val_cand = compute_value_exact(h_cand)
            if val_cand > best_val:
                best_val = val_cand
                best_h = h_cand.copy()
        except Exception:
            pass
            
    return list(best_h)

# Test locally if needed
if __name__ == "__main__":
    start = time.time()
    h = construct()
    end = time.time()
    val = compute_value_exact(h)
    print(f"Time: {end-start:.2f}s")
    print(f"Value: {val}")
    print(f"Seed Value: {compute_value_exact(np.ones(50))}")
    print(f"Score: {val / compute_value_exact(np.ones(50))}")
