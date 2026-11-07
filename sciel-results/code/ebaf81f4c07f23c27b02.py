import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_value(h):
    """
    Computes the objective value for a given height vector h.
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        h = np.clip(h, 0, None)
    h_sum = np.sum(h)
    if h_sum == 0:
        return 0.0
    
    g = np.convolve(h, h, mode='full')
    m = len(g)
    
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    w = 1.0 / (m + 1)
    
    y_curr = y[:m+1]
    y_next = y[1:m+2]
    terms = y_curr**2 + y_curr * y_next + y_next**2
    S = np.sum(terms)
    l2 = (w / 3.0) * S
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf == 0:
        return 0.0
        
    value = l2 / (norm_1 * norm_inf)
    return value

def _tv_denoise(x, lam, max_iter=100):
    """
    1D Total Variation Denoising via taut string algorithm (or simple iterative solver).
    Here we use a simple iterative solver for the convex problem:
    min 0.5 * ||z - x||^2 + lam * sum(|z[i] - z[i-1]|)
    """
    z = x.copy()
    n = len(z)
    # Simple iterative projection
    for _ in range(max_iter):
        # Soft thresholding on differences
        d = np.diff(z)
        # The optimality condition for TV denoising involves the subgradient
        # For simplicity, we use a gradient descent step with subgradient
        # grad = z - x - lam * (sign(diff(z)) - sign(diff(z)[:-1])) roughly
        # Actually, let's use the standard iterative scheme for TV denoising
        # z_{k+1} = z_k - alpha * (z_k - x - lam * div(sign(grad(z_k))))
        # This is tricky. Let's use a simpler approach: 
        # Since we want piecewise constant, we can just try to smooth sharp edges.
        # But the directive specifically mentions TV denoising.
        # A robust way is to solve the dual problem or use the taut string.
        # Given time constraints, a simple heuristic "plateau enhancement" might be safer/faster.
        # However, let's try a basic iterative solver.
        
        # Compute gradient (differences)
        grad = np.zeros(n)
        if n > 1:
            grad[1:-1] = (z[2:] - z[:-2]) / 2.0
            grad[0] = z[1] - z[0]
            grad[-1] = z[-1] - z[-2]
        
        # Subgradient of TV norm
        # sign(0) = 0
        sg = np.sign(grad)
        # Divergence of subgradient
        div_sg = np.zeros(n)
        if n > 1:
            div_sg[0] = -sg[0]
            div_sg[-1] = sg[-1]
            div_sg[1:-1] = sg[1:] - sg[:-1]
        
        # Update
        alpha = 1.0 / (1.0 + lam) # Simple step size
        z = z - alpha * (z - x - lam * div_sg)
        
        # Enforce non-negativity
        z = np.clip(z, 0, None)
        
    return z

def _knots_to_h(params):
    """
    Convert knot parameters to a height vector of length N.
    Uses 7 Gaussian tents.
    """
    K = 7
    centers = params[0:K]
    log_widths = params[K:2*K]
    log_amps = params[2*K:3*K]
    
    widths = np.exp(log_widths)
    amps = np.exp(log_amps)
    
    x = np.arange(N)
    h = np.zeros(N)
    
    for i in range(K):
        c = centers[i]
        w = widths[i]
        a = amps[i]
        
        # Gaussian tent
        h += a * np.exp(-((x - c) / w)**2)
        
    return h

def _objective_gauss(params):
    """
    Objective function for Nelder-Mead minimization (returns negative value).
    """
    try:
        h = _knots_to_h(params)
        val = _compute_value(h)
        return -val
    except Exception:
        return 1e10

def _coordinate_descent(h, num_steps=500, time_limit=10.0):
    """
    Perform coordinate descent on the height vector h.
    """
    start_time = time.time()
    h = np.array(h, dtype=float)
    
    for step in range(num_steps):
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        for i in range(N):
            # Try increasing and decreasing h[i]
            original = h[i]
            
            # Simple local search around current value
            # We can try a few candidate values
            candidates = [original]
            # Scale based on current value or mean
            scale = max(1e-4, abs(original) * 0.1, np.mean(h) * 0.1)
            for delta in [-scale, -scale*0.5, scale*0.5, scale]:
                val = original + delta
                if val >= 0:
                    candidates.append(val)
            
            best_val = original
            best_obj = _compute_value(h)
            
            for c_val in candidates:
                h[i] = c_val
                obj = _compute_value(h)
                if obj > best_obj:
                    best_obj = obj
                    best_val = c_val
            
            h[i] = best_val
            if h[i] != original:
                improved = True
        
        if not improved:
            break
            
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    K = 7
    num_params = 3 * K
    
    # Initial candidates
    candidates = []
    
    # 1. Symmetric multi-lobe: 7 tents evenly spaced
    c0 = np.linspace(5, 45, K)
    w0 = np.full(K, 3.0)
    a0 = np.full(K, 1.0)
    p0 = np.concatenate([c0, np.log(w0), np.log(a0)])
    candidates.append(p0)
    
    # 2. Central peak + side lobes
    c1 = np.array([25, 15, 35, 5, 45, 10, 40])
    w1 = np.array([5.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0])
    a1 = np.array([1.0, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2])
    p1 = np.concatenate([c1, np.log(w1), np.log(a1)])
    candidates.append(p1)
    
    # 3. Random starts
    np.random.seed(50)
    for _ in range(30):
        c = np.random.uniform(0, N, K)
        w = np.random.uniform(1.0, 10.0, K)
        a = np.random.uniform(0.1, 5.0, K)
        p = np.concatenate([c, np.log(w), np.log(a)])
        candidates.append(p)
        
    # Evaluate all candidates
    scored = []
    for p in candidates:
        val = _compute_value(_knots_to_h(p))
        scored.append((val, p))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    best_val = scored[0][0]
    best_params = scored[0][1].copy()
    best_h = _knots_to_h(best_params)
    
    # Polish top candidates with Nelder-Mead
    top_candidates = [p for _, p in scored[:5]]
    
    for p_init in top_candidates:
        if time.time() > deadline - 30:
            break
            
        try:
            res = minimize(
                _objective_gauss,
                p_init,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            val_opt = -res.fun
            if val_opt > best_val:
                best_val = val_opt
                best_params = res.x.copy()
                best_h = _knots_to_h(best_params)
        except Exception:
            pass
            
    # Apply TV denoising / Plateau enhancement
    # We try different lambda values for TV denoising
    if time.time() < deadline - 20:
        h_tv_candidates = []
        for lam in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
            h_tv = _tv_denoise(best_h, lam, max_iter=50)
            h_tv_candidates.append(h_tv)
            
        # Evaluate TV denoised versions
        for h_tv in h_tv_candidates:
            val_tv = _compute_value(h_tv)
            if val_tv > best_val:
                best_val = val_tv
                best_h = h_tv
        
        # If TV denoising improved things, we might want to refine the TV denoised version
        # But since TV denoising is a projection, the result is a specific vector.
        # We can try coordinate descent on the TV denoised vector.
        
    # Final Coordinate Descent Polish
    if time.time() < deadline - 5:
        h_cd = _coordinate_descent(best_h, num_steps=200, time_limit=deadline - 10 - time.time())
        val_cd = _compute_value(h_cd)
        if val_cd > best_val:
            best_val = val_cd
            best_h = h_cd

    # Final check
    if best_h is None:
        return list(np.ones(N) / N)
        
    # Ensure non-negative and finite
    best_h = np.clip(best_h, 0, None)
    if np.sum(best_h) == 0:
        best_h = np.ones(N) / N
        
    return list(best_h)

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
