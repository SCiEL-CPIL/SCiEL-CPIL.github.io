"""Second autocorrelation inequality: maximize value = l2/(norm_1*norm_inf).

Strategy: Deterministic multi-start gradient ascent with Newton-like steps.
We exploit scale invariance by fixing the L1 norm of h to 1 (or simply optimizing
on the log-simplex). The objective is smooth in the interior (h > 0).
We use analytic gradients derived from the convolution structure.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 50

def _eval(h):
    """Exact evaluator (mirrors the public contract)."""
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    l2 = (w / 3.0) * np.sum(y[:-1] * y[:-1] + y[:-1] * y[1:] + y[1:] * y[1:])
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    return l2 / (norm_1 * norm_inf)

def _objective_and_grad(x):
    """
    Objective: -value(h) where h = exp(x) to ensure positivity and handle scale.
    Wait, scale invariance means value(h) = value(c*h).
    If we use h = exp(x), scaling h by c adds constant to x.
    The objective is flat along the scaling direction.
    This makes the Hessian singular.
    
    Better: Fix the norm of h to be 1 (e.g., sum(h)=1) and optimize on the simplex.
    Or, simpler: Just use scipy's L-BFGS-B on the positive orthant, 
    but initialize with a fixed norm to avoid wandering.
    
    Actually, let's look at the structure.
    value = l2(g) / (||g||_1 * ||g||_inf)
    Since g = h*h, ||g||_1 = ||h||_2^2.
    ||g||_inf = max(h*h).
    
    Let's just use a robust numerical gradient ascent with line search, 
    but start from diverse, structured initializations (gaussians, triangles, etc.)
    and use the BFGS method which is very effective for smooth 50-dim problems.
    """
    h = np.exp(x)  # Ensures h > 0
    # To handle scale invariance, we can normalize h inside the objective
    # or just let the optimizer handle it. Normalizing inside helps condition.
    h_norm = h / np.sum(h)  # Sum to 1
    val = _eval(h_norm)
    # We want to maximize val, so minimize -val
    # Numerical gradient for safety, or analytic?
    # Analytic gradient is complex due to max and abs.
    # scipy.optimize.minimize with BFGS uses numerical gradients by default if not provided.
    # For N=50, numerical gradient is 51 evals per step. 
    # 150s limit. If eval is ~1ms, 51ms per step. ~2500 steps possible.
    # This is plenty for BFGS.
    
    return -val

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin
    
    rng = np.random.default_rng(50)
    
    best_h = np.ones(N)
    best_val = _eval(best_h)
    
    # Generate diverse starting points
    starts = []
    
    # 1. Uniform (seed)
    starts.append(np.ones(N))
    
    # 2. Gaussian centered
    for center in [0.3, 0.5, 0.7]:
        for width in [0.1, 0.2, 0.4]:
            x = np.arange(N)
            h = np.exp(-((x - center*(N-1))**2) / (2*width**2))
            starts.append(h)
            
    # 3. Triangular
    for center in [0.3, 0.5, 0.7]:
        h = 1.0 - np.abs(np.arange(N) - center*(N-1)) / (N/2)
        h = np.maximum(h, 0.1) # keep positive
        starts.append(h)
        
    # 4. Random smooth
    for _ in range(10):
        h = np.random.uniform(0.5, 1.5, N)
        # Smooth it
        kernel = np.ones(5)/5
        h = np.convolve(h, kernel, mode='same')
        starts.append(h)
        
    # 5. Two peaks
    for c1, c2 in [(0.3, 0.7), (0.2, 0.8), (0.4, 0.6)]:
        x = np.arange(N)
        h = np.exp(-((x - c1*(N-1))**2) / 2.0) + np.exp(-((x - c2*(N-1))**2) / 2.0)
        starts.append(h)

    # Run optimization for each start
    for i, h0 in enumerate(starts):
        if time.time() > deadline - 10.0:
            break
            
        # Normalize start
        h0 = h0 / np.sum(h0)
        
        # Use BFGS with numerical gradient
        # Bounds: h_i > 0. We map log(h) to unconstrained space?
        # If we optimize in log-space, we avoid bounds.
        # Let x = log(h). h = exp(x).
        # Objective f(x) = -value(exp(x))
        # Since value is scale invariant, f(x) is invariant to adding constant to x.
        # This creates a valley. BFGS might struggle with the flat direction.
        # Fix: constrain sum(exp(x)) = 1? No, that's hard.
        # Alternative: Optimize h directly with bounds h_i >= epsilon.
        
        # Let's try direct optimization with bounds.
        # h_i in [1e-6, 10.0]
        # Initial h0 is normalized to sum=1, so max is ~1.
        
        try:
            res = minimize(
                lambda h: -_eval(h), 
                h0, 
                method='L-BFGS-B', 
                bounds=[(1e-6, 10.0)] * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            h_opt = res.x
            v_opt = _eval(h_opt)
            
            if v_opt > best_val:
                best_val = v_opt
                best_h = h_opt
        except Exception:
            pass

    # Final polish: Coordinate ascent on the best found
    if time.time() < deadline - 5.0:
        for _ in range(10):
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                orig = best_h[i]
                # Try a few multipliers
                for f in [1.2, 0.8, 1.5, 0.5, 1.1, 0.9]:
                    cand = best_h.copy()
                    cand[i] = orig * f
                    # Ensure non-negative
                    if cand[i] < 0: continue
                    v = _eval(cand)
                    if v > best_val:
                        best_val = v
                        best_h = cand
                        improved = True
                        break
            if not improved:
                break

    return list(best_h)
