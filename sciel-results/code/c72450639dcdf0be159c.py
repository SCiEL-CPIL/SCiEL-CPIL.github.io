import time
import numpy as np
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes the ratio D/M where:
    D = H(x - y)
    M = max(H(x), H(x + y), H(x + 2y))
    
    Strategy:
    1. Identify that the problem is a non-convex ratio of concave functions.
    2. Use a multi-start local optimization approach with SLSQP.
    3. Generate high-quality starting points:
       - Uniform
       - "Concentrated X" solutions (optimize y distribution for each fixed x)
       - "Concentrated Y" solutions
       - Random Dirichlet samples
    4. Polish the best results with SLSQP to high precision.
    """
    n = 10
    deadline = time.time() + 590.0  # Safety margin
    
    # Grid coordinates
    ys, xs = np.mgrid[0:n, 0:n]
    
    # Flatten for binning
    d_val = (xs - ys).flatten()
    m0_val = xs.flatten()
    m1_val = (xs + ys).flatten()
    m2_val = (xs + 2 * ys).flatten()
    
    # Unique values and inverse mapping
    d_vals, d_inv = np.unique(d_val, return_inverse=True)
    m0_vals, m0_inv = np.unique(m0_val, return_inverse=True)
    m1_vals, m1_inv = np.unique(m1_val, return_inverse=True)
    m2_vals, m2_inv = np.unique(m2_val, return_inverse=True)
    
    num_d = len(d_vals)
    num_m0 = len(m0_vals)
    num_m1 = len(m1_vals)
    num_m2 = len(m2_vals)

    def compute_entropies_from_p(p_flat):
        p_d = np.bincount(d_inv, weights=p_flat, minlength=num_d)
        p_m0 = np.bincount(m0_inv, weights=p_flat, minlength=num_m0)
        p_m1 = np.bincount(m1_inv, weights=p_flat, minlength=num_m1)
        p_m2 = np.bincount(m2_inv, weights=p_flat, minlength=num_m2)
        
        def shannon(p_dist):
            p_nonzero = p_dist[p_dist > 1e-15]
            if len(p_nonzero) == 0:
                return 0.0
            return -np.sum(p_nonzero * np.log2(p_nonzero))
        
        H_d = shannon(p_d)
        H_m0 = shannon(p_m0)
        H_m1 = shannon(p_m1)
        H_m2 = shannon(p_m2)
        
        M = max(H_m0, H_m1, H_m2)
        return H_d, M

    def get_score(p_flat):
        H_d, M = compute_entropies_from_p(p_flat)
        if M < 1e-9:
            return -1e9
        return H_d / M

    def objective(p_flat):
        H_d, M = compute_entropies_from_p(p_flat)
        if M < 1e-9:
            return 1e9
        return -H_d / M

    # Generate Starting Points
    starts = []
    
    # 1. Uniform
    starts.append(np.ones(100) / 100.0)
    
    # 2. "Concentrated X" solutions
    # For each x0, fix p[x0, :] to be a distribution over y, and 0 elsewhere.
    # Optimize the y-distribution to maximize H(y) / max(H(y), H(2y)).
    def solve_concentrated_x(x0):
        # Variables: p_y (10 values)
        # Objective: - H(y) / max(H(y), H(2y))
        
        def obj_y(p_y):
            p_y = np.clip(p_y, 0, None)
            s = np.sum(p_y)
            if s < 1e-9: return 1e9
            p_y = p_y / s
            
            # Compute H(y)
            p_nonzero = p_y[p_y > 1e-15]
            H_y = -np.sum(p_nonzero * np.log2(p_nonzero)) if len(p_nonzero) > 0 else 0.0
            
            # Compute H(2y)
            # 2y values: 0, 2, ..., 18
            # Bin p_y into these
            # y=0->0, y=1->2, ..., y=9->18
            # All distinct, so H(2y) = H(y)
            # Wait, if y is uniform, 2y is uniform on 10 points.
            # If y is non-uniform, 2y is non-uniform on 10 points.
            # So H(2y) = H(y) always for this mapping?
            # Yes, because the mapping y -> 2y is a bijection on the support {0..9}.
            # So M = max(0, H(y), H(y)) = H(y).
            # Ratio = H(y) / H(y) = 1.
            # So any distribution on a single x-line gives ratio 1.
            # This is a plateau.
            # We need to move off the line to improve.
            return -1.0 # Constant
            
        # Since it's constant, just pick a good y distribution to start the off-line search.
        # Uniform y is a good start.
        p_flat = np.zeros(100)
        p_flat[x0 * 10 : (x0+1) * 10] = 1.0 / 10.0
        return p_flat

    for x0 in range(10):
        starts.append(solve_concentrated_x(x0))
        
    # 3. "Concentrated Y" solutions
    # Fix y=y0.
    # x varies.
    # D = H(x-y0) = H(x).
    # M = max(H(x), H(x+y0), H(x+2y0)).
    # H(x+y0) = H(x). H(x+2y0) = H(x).
    # M = H(x).
    # Ratio = 1.
    # Also a plateau.
    for y0 in range(10):
        p_flat = np.zeros(100)
        p_flat[y0 * 10 : (y0+1) * 10] = 1.0 / 10.0
        starts.append(p_flat)
        
    # 4. Random Dirichlet
    rng = np.random.default_rng(42)
    for _ in range(20):
        alpha = rng.uniform(0.1, 2.0, 100)
        p_flat = rng.dirichlet(alpha)
        starts.append(p_flat)
        
    # 5. "Diagonal" patterns
    # p[x, x] = 1/10
    p_diag = np.zeros(100)
    for i in range(10):
        p_diag[i*10+i] = 1.0/10.0
    starts.append(p_diag)
    
    # 6. "Anti-diagonal" patterns
    p_adiag = np.zeros(100)
    for i in range(10):
        p_adiag[i*10+(9-i)] = 1.0/10.0
    starts.append(p_adiag)

    best_score = -1e9
    p_best = starts[0]
    
    # Optimize from each start
    for i, start_p in enumerate(starts):
        if time.time() > deadline - 10:
            break
            
        # Convert to log-space for better conditioning? 
        # SLSQP works in linear space with bounds.
        # Bounds: [0, 1]
        # Constraints: sum(p) = 1
        
        constraints = [{'type': 'eq', 'fun': lambda p: np.sum(p) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(100)]
        
        try:
            res = minimize(
                objective, 
                start_p, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            p_res = res.x
            p_res = np.clip(p_res, 0, None)
            p_res /= np.sum(p_res)
            
            score = get_score(p_res)
            
            if score > best_score:
                best_score = score
                p_best = p_res
                
        except Exception as e:
            pass

    # Final validation and normalization
    P = p_best.reshape((n, n))
    P = np.clip(P, 0.0, None)
    s = np.sum(P)
    if s > 1e-9:
        P = P / s
    else:
        P = np.ones((n, n)) / (n * n)
        
    return P
