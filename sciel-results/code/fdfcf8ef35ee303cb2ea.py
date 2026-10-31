import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for h >= 0.
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    
    l2_sq = np.sum(h * h)
    
    # Efficient autoconvolution using FFT
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=n_fft)
    hh = np.fft.irfft(H * H, n=n_fft)
    hh_max = np.max(hh)
    
    if hh_max < 1e-300:
        return 0.0
        
    return l2_sq / (l1 * hh_max)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_h = np.ones(N) / N
    best_score = compute_ratio(best_h)
    
    # Objective function for minimization: -ratio
    # We work with unnormalized weights w, then normalize inside the evaluation
    # to handle the scale invariance naturally, but L-BFGS-B needs a fixed dimension.
    # Actually, it's better to optimize the normalized weights directly?
    # No, standard practice: optimize w >= 0. The ratio is scale invariant.
    # We can fix sum(w)=1 by optimizing N-1 variables? 
    # Or just optimize N variables and normalize. The gradient of normalized ratio is tricky.
    # Let's optimize w in R^N with bounds [0, inf). 
    # f(w) = - compute_ratio(w)
    # Since compute_ratio is scale invariant, the landscape is flat along rays.
    # L-BFGS-B might struggle with the flat direction.
    # Better: Fix ||w||_1 = 1. 
    # We can parameterize using N-1 variables x_0...x_{N-2} where w_i = x_i for i<N-1, w_{N-1} = 1 - sum(x).
    # This requires sum(x) <= 1. Bounds: x_i >= 0, sum(x_i) <= 1.
    # This is a simplex constraint.
    # Alternatively, use softmax? No, we want sparse solutions potentially.
    
    # Let's stick to optimizing w in R^N with bounds [0, 1] and normalizing.
    # The gradient will point towards increasing ratio.
    # To avoid the scale invariance issue, we can add a small regularization?
    # Or just let it find a local max on the unit sphere/simplex.
    # Actually, if we normalize inside the objective, the gradient is well-defined 
    # as long as we project back to the simplex?
    
    # Let's use a simpler approach: 
    # Optimize w in R^N. 
    # Objective: - ratio(w)
    # If we start with sum(w)=1, and the optimizer moves w, the ratio stays same if it scales w.
    # But L-BFGS-B uses gradients. The gradient of ratio(w) w.r.t w is 0 in the scaling direction?
    # Let's check: R(cw) = R(w). So dR/dc = 0.
    # This means the Hessian is singular. L-BFGS-B might be slow or unstable.
    
    # Better Strategy: 
    # Optimize the weights on the simplex explicitly.
    # We represent the state as N-1 variables.
    # But implementing the projection/gradient for simplex is complex.
    
    # Alternative: 
    # Use `minimize` on the full N-dimensional vector, but penalize deviation from sum=1?
    # Or just normalize w inside the objective and hope the optimizer stays near sum=1?
    # If w becomes large, numerical issues. If w becomes small, 0/0.
    # Let's constrain w to [0, 1]. 
    # Then sum(w) is in [0, N]. 
    # If we start with sum(w)=1, and the gradient pushes w, the normalization 
    # effectively projects the step back to the simplex? No, the gradient is computed 
    # on the raw w.
    
    # Let's try a different optimization variable: 
    # Log-weights? No, zeros are important.
    
    # Let's try: 
    # 1. Generate diverse starting points on the simplex (sum=1).
    # 2. Use L-BFGS-B on the N-1 dimensional simplex coordinates.
    
    def simplex_objective(x, N):
        # x has length N-1
        # w has length N
        # w_i = x_i for i < N-1
        # w_{N-1} = 1 - sum(x)
        # If sum(x) > 1, w_{N-1} < 0. 
        # We must ensure sum(x) <= 1.
        # L-BFGS-B bounds: x_i >= 0. But sum(x) <= 1 is a linear constraint, not handled by simple bounds.
        # We can use SLSQP for constraints, but it's slower.
        
        # Workaround: Use softmax parameterization? 
        # w_i = exp(y_i) / sum(exp(y_j)).
        # This always gives positive weights on simplex.
        # But it cannot reach zeros (sparse solutions). 
        # The optimal solution might be sparse.
        
        # Let's stick to the N-dimensional optimization with normalization inside.
        # It often works in practice if initialized well.
        
        pass

    # Let's implement the N-dimensional optimization with normalization.
    # To handle the scale invariance, we can fix one variable? 
    # No, let's just use the raw vector and normalize.
    # We will use `minimize` with method='L-BFGS-B'.
    # Bounds: [0, 1] for each weight.
    
    def neg_ratio(w):
        w = np.asarray(w)
        w = np.maximum(w, 0)
        s = np.sum(w)
        if s < 1e-12:
            return 1e6
        w_norm = w / s
        r = compute_ratio(w_norm)
        return -r

    def neg_ratio_grad(w):
        # Numerical gradient is expensive but N=50 is small.
        # Analytical gradient:
        # R = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
        # Let h = w / ||w||_1
        # This is complex due to the max in autoconvolution.
        # Let's use finite differences? 
        # N=50, 100 evaluations per gradient step? Too slow for 140s if many steps.
        # Let's assume the optimizer can use numerical gradients? 
        # L-BFGS-B supports finite difference gradients.
        # With N=50, one gradient eval takes ~50 * cost(f).
        # cost(f) is FFT ~ 50 log 50 ~ 300 ops.
        # 50 * 300 = 15000 ops per gradient.
        # 100 steps * 15000 = 1.5M ops. Very fast.
        return None # Let scipy handle it with numerical differentiation

    # Starting points
    starts = []
    
    # 1. Uniform
    starts.append(np.ones(N) / N)
    
    # 2. Sparse Random (size 5, 10, 20)
    for k in [5, 10, 20]:
        for _ in range(10):
            idx = rng.choice(N, size=k, replace=False)
            w = np.zeros(N)
            w[idx] = 1.0
            w /= np.sum(w)
            starts.append(w)
            
    # 3. Sidon-like / Structured
    # Try to place peaks far apart to reduce autoconvolution peak.
    # e.g. indices 0, 10, 20, 30, 40
    idx = np.array([0, 10, 20, 30, 40])
    w = np.zeros(N)
    w[idx] = 1.0
    w /= np.sum(w)
    starts.append(w)
    
    # 4. Random Gaussian on simplex
    for _ in range(20):
        w = np.abs(rng.normal(0, 1, N))
        w /= np.sum(w)
        starts.append(w)
        
    # 5. "Blocky" structures
    for block_size in [2, 5]:
        for start_idx in range(0, N, block_size * 2):
            w = np.zeros(N)
            w[start_idx:start_idx+block_size] = 1.0
            if np.sum(w) > 0:
                w /= np.sum(w)
                starts.append(w)

    # Deduplicate starts roughly
    unique_starts = []
    for s in starts:
        is_dup = False
        for u in unique_starts:
            if np.allclose(s, u, atol=1e-6):
                is_dup = True
                break
        if not is_dup:
            unique_starts.append(s)
    
    # Evaluate initial scores
    for w in unique_starts:
        score = compute_ratio(w)
        if score > best_score:
            best_score = score
            best_h = w.copy()
            
    # Optimize each start
    bounds = [(0, 1.0) for _ in range(N)]
    
    for w0 in unique_starts:
        if time.time() > deadline:
            break
            
        # Optimize
        try:
            res = minimize(neg_ratio, w0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50})
            
            w_opt = res.x
            w_opt = np.maximum(w_opt, 0)
            s = np.sum(w_opt)
            if s > 1e-12:
                w_opt /= s
                score = compute_ratio(w_opt)
                if score > best_score:
                    best_score = score
                    best_h = w_opt.copy()
        except Exception:
            pass

    # Final local polish with finer steps if time permits
    # Use the best_h found so far
    if time.time() < deadline - 5:
        # Do a few more random restarts from the best point with small noise
        for _ in range(20):
            if time.time() > deadline:
                break
            noise = rng.normal(0, 0.05, N)
            w_noisy = best_h + noise
            w_noisy = np.maximum(w_noisy, 0)
            s = np.sum(w_noisy)
            if s < 1e-12: continue
            w_noisy /= s
            
            try:
                res = minimize(neg_ratio, w_noisy, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 200, 'ftol': 1e-14})
                w_opt = res.x
                w_opt = np.maximum(w_opt, 0)
                s = np.sum(w_opt)
                if s > 1e-12:
                    w_opt /= s
                    score = compute_ratio(w_opt)
                    if score > best_score:
                        best_score = score
                        best_h = w_opt.copy()
            except Exception:
                pass

    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
