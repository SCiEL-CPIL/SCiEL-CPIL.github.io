import numpy as np
import time
import scipy.optimize

N = 50
START_TIME = time.time()
TIME_LIMIT = 140.0  # Leave margin before 150s kill

def compute_ratio_log(h):
    """Compute log ratio: log(||h||_2^2 / (||h||_1 * ||h*h||_inf))
    h is a vector of positive values (exp of log-space variables).
    """
    if np.any(h <= 0):
        return -np.inf
    
    dx = 1.0 / N
    l1 = dx * np.sum(h)
    if l1 <= 0:
        return -np.inf
    
    l2_sq = dx * np.sum(h**2)
    
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    conv_inf = np.max(conv) * dx
    
    if conv_inf <= 0:
        return -np.inf
    
    # Ratio = l2_sq / (l1 * conv_inf)
    # Log Ratio = log(l2_sq) - log(l1) - log(conv_inf)
    # log(l2_sq) = log(dx) + log(sum(h^2))
    # log(l1) = log(dx) + log(sum(h))
    # log(conv_inf) = log(dx) + log(max(conv))
    
    # Note: dx cancels out in the ratio?
    # R = (dx * sum(h^2)) / (dx * sum(h) * dx * max(conv)) = sum(h^2) / (dx * sum(h) * max(conv))
    # Let's just compute the values directly to be safe, but use logs for stability if possible.
    # Actually, for L-BFGS-B, we need the gradient. 
    # Let's define the objective as -log(R) to minimize.
    
    val = l2_sq / (l1 * conv_inf)
    if val <= 0:
        return -np.inf
    return np.log(val)

def objective_log(x):
    """Minimize negative log ratio. x are log heights."""
    # To avoid overflow/underflow in exp(x), we can shift x.
    # But usually x will be around -10 to 0.
    h = np.exp(x)
    
    # If h is too large or small, it might cause issues, but exp handles a wide range.
    # We want to maximize log(R). So minimize -log(R).
    r_log = compute_ratio_log(h)
    if np.isinf(r_log):
        return 1e6
    return -r_log

def objective_grad(x):
    """Compute gradient of objective_log."""
    # Use numerical gradient or analytic?
    # Analytic is better for speed.
    # Let f(x) = -log(R(exp(x)))
    # R(h) = ||h||_2^2 / (||h||_1 ||h*h||_inf)
    # This is complex due to max operator in ||h*h||_inf.
    # Numerical gradient is safer and N=50 is small.
    eps = 1e-5
    grad = np.zeros_like(x)
    f0 = objective_log(x)
    for i in range(N):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (objective_log(xp) - objective_log(xm)) / (2 * eps)
    return grad

def optimize_shape_log(shape0, max_time=10.0):
    """Optimize in log space."""
    shape0 = np.maximum(shape0, 1e-10)
    x0 = np.log(shape0)
    
    best_x = x0.copy()
    best_val = objective_log(x0)
    
    start = time.time()
    
    try:
        # Use L-BFGS-B
        res = scipy.optimize.minimize(
            objective_log, 
            x0, 
            method='L-BFGS-B',
            jac=None, # Use numerical gradient for simplicity and robustness, N=50 is small
            options={'maxiter': 200, 'maxfun': 500, 'disp': False}
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
        elif not np.isinf(res.fun):
            # If it didn't improve but is valid, maybe it's better numerically?
            # We stick with the best log ratio.
            pass
            
    except Exception:
        pass
    
    # Check time
    if time.time() - start > max_time:
        return best_x, -best_val
    
    # Final polish: maybe a few steps of Nelder-Mead or just return
    # The ratio is scale invariant?
    # R(c*h) = (c^2 ||h||_2^2) / (c ||h||_1 * c ||h*h||_inf) = R(h)
    # So the scale doesn't matter for the ratio. 
    # The problem asks to maximize the ratio. The scale is irrelevant to the score.
    # So we can return any scale. Let's return the optimized shape.
    
    h_opt = np.exp(best_x)
    return h_opt, -best_val

def construct():
    np.random.seed(42)
    best_h = None
    best_score = -np.inf
    
    # Initial shapes (positive)
    # 1. Uniform
    h1 = np.ones(N)
    # 2. Gaussian
    x = np.linspace(-3, 3, N)
    h2 = np.exp(-x**2 / 2)
    # 3. Sparse single peak
    h3 = np.zeros(N)
    h3[N//2] = 1.0
    # 4. Sparse two peaks
    h4 = np.zeros(N)
    h4[10] = 1.0
    h4[40] = 1.0
    # 5. Random sparse
    h5 = np.zeros(N)
    idx = np.random.choice(N, 5, replace=False)
    h5[idx] = 1.0
    # 6. Triangle
    h6 = np.abs(1 - 2*np.linspace(0, 1, N))
    
    shapes = [h1, h2, h3, h4, h5, h6]
    
    for i, h0 in enumerate(shapes):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        # Ensure non-negative
        h0 = np.maximum(h0, 1e-6)
        
        # Normalize to unit L1 for consistency, though scale invariant
        s = np.sum(h0)
        if s > 0:
            h0 = h0 / s
            
        # Optimize
        h_opt, score = optimize_shape_log(h0, max_time=15.0)
        
        if score > best_score:
            best_score = score
            best_h = h_opt
            
        # Print progress (optional, but good for debugging if we could see it)
        # print(f"Shape {i}: Score {score:.6f}")
        
    if best_h is None:
        # Fallback
        return [1.0] * N
        
    # The ratio is scale invariant, so we can just return best_h.
    # However, the evaluator might check for specific properties? 
    # "nonnegative step heights". 
    # Returning the optimized shape directly.
    result = np.maximum(0, best_h)
    return [float(x) for x in result]
