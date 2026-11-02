import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _score(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = float(np.sum(h))
    if l1 <= 1e-12:
        return 0.0
    l2_sq = float(np.dot(h, h))
    
    # Autoconvolution h * h
    c = np.convolve(h, h, mode='full')
    c_inf = float(np.max(np.abs(c)))
    
    if c_inf < 1e-12:
        return 1.0
        
    return l2_sq / (l1 * c_inf)

def _neg_score(weights):
    """
    Objective function for minimization (we want to maximize the score).
    """
    return -_score(weights)

def construct():
    t0 = time.time()
    deadline = t0 + 110.0
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    def update_best(h_candidate):
        nonlocal best_h, best_score
        if np.any(h_candidate < 0):
            h_candidate = np.maximum(h_candidate, 0)
        s = _score(h_candidate)
        if s > best_score:
            best_score = s
            best_h = h_candidate.copy()
            return True
        return False

    # 1. Initial candidates
    # Spike
    h_spike = np.zeros(N)
    h_spike[0] = 1.0
    update_best(h_spike)
    
    # Uniform
    h_uni = np.ones(N) / N
    update_best(h_uni)
    
    # Gaussian
    x = np.linspace(-1, 1, N)
    h_gauss = np.exp(-x**2 * 5)
    update_best(h_gauss)
    
    # 2. Spectral Water-Filling / Iterative Re-weighting Strategy
    # The prompt suggests: "Model the problem as a spectral optimization... use iterative water-filling to allocate heights inversely proportional to the local spectral density"
    
    # Let's implement a coordinate descent / multiplicative update method that is robust.
    # We want to maximize J(h) = ||h||_2^2 / (||h||_1 ||h*h||_inf)
    # Since it's scale invariant, we can fix ||h||_1 = 1 and maximize ||h||_2^2 / ||h*h||_inf.
    # Or fix ||h||_2 = 1 and maximize 1 / (||h||_1 ||h*h||_inf).
    
    # Let's use a general optimization approach with scipy, starting from good seeds.
    # We will use Nelder-Mead or COBYLA (since we have inequality constraints h>=0, but we can handle it via projection or bounds).
    # Nelder-Mead doesn't handle bounds directly well, so we'll use a transformation or COBYLA.
    # Actually, let's use scipy.optimize.minimize with method='COBYLA' or just project after NM.
    # Projecting after NM is usually faster.
    
    def _neg_score_projected(weights):
        # Weights are unconstrained, we project to h>=0 and sum=1 (or just h>=0)
        # Since scale invariant, we can normalize inside score, but it's better to optimize the shape.
        # Let's just pass the vector to _score which handles non-negative check.
        # To help NM, let's ensure non-negativity by using softplus or similar? 
        # No, simple projection is fine if we start positive.
        return -_score(weights)
    
    # Strategy: Start from the best initial, then run a local optimization.
    # We'll try a few restarts with different initializations.
    
    rng = np.random.RandomState(123)
    
    candidates = [best_h.copy(), h_uni, h_gauss, h_spike]
    
    # Add some random smooth candidates
    for _ in range(3):
        h_rand = rng.rand(N)
        h_rand = np.convolve(h_rand, np.ones(3)/3, mode='same') # Smooth
        h_rand /= np.sum(h_rand)
        candidates.append(h_rand)
    
    # Add "anti-correlation" like structures?
    # Alternating signs are not allowed (non-negative).
    # But we can try sparse patterns.
    for mask_size in [5, 10, 20]:
        indices = rng.choice(N, mask_size, replace=False)
        h_sparse = np.zeros(N)
        h_sparse[indices] = 1.0 / mask_size
        candidates.append(h_sparse)
        
    # Local Optimization
    # We will optimize the vector h directly.
    # To handle non-negativity, we can optimize in log-space if all positive, 
    # or use COBYLA which handles inequality constraints.
    # COBYLA is slower but handles constraints.
    # Let's try Nelder-Mead on the raw values, projecting to [0, inf) at each step is tricky with NM.
    # Instead, let's use scipy.optimize.minimize with method='L-BFGS-B' or 'COBYLA'.
    # L-BFGS-B requires gradients. We can compute them.
    # Or just use 'Nelder-Mead' and hope it stays positive, or penalize negatives heavily.
    
    def _objective_with_penalty(weights):
        h = np.asarray(weights, dtype=float)
        if np.min(h) < 0:
            penalty = 1e6 * np.sum(np.minimum(h, 0)**2)
            return -_score(np.maximum(h, 0)) + penalty
        return -_score(h)
        
    # Actually, a simpler approach for "Spectral Water Filling" logic:
    # The peak of the autocorrelation is at lag 0.
    # Off-diagonal peaks are what we want to minimize relative to the energy.
    # If we can make the autocorrelation more "flat" (white noise like), the off-diagonal peaks drop.
    # But we want to maximize ||h||_2^2. Concentrating energy increases L2.
    # The trade-off is between concentration (high L2) and side-lobes (high Autocorr).
    
    # Let's use a gradient-based approach with projected gradient descent.
    # We maximize f(h) = ||h||_2^2 / ||h*h||_inf subject to h>=0, ||h||_1=1.
    # Let's just run COBYLA on the raw h vector with constraints.
    
    for start_h in candidates:
        if time.time() > deadline - 5:
            break
            
        # Normalize start_h
        s1 = np.sum(start_h)
        if s1 > 0:
            start_h = start_h / s1
        
        try:
            # Use COBYLA for constrained optimization
            # Constraints: h >= 0
            constraints = [{'type': 'ineq', 'fun': lambda h, i=i: h[i]} for i in range(N)]
            
            res = minimize(
                _neg_score, 
                start_h, 
                method='COBYLA',
                constraints=constraints,
                options={'maxiter': 200, 'tol': 1e-4}
            )
            
            if res.success:
                h_opt = res.x
                # Ensure non-negative
                h_opt = np.maximum(h_opt, 0)
                if np.sum(h_opt) > 0:
                    h_opt = h_opt / np.sum(h_opt) # Normalize
                    update_best(h_opt)
                    
        except Exception:
            pass

    # Additional refinement: Coordinate Descent / Hill Climbing on the best found
    # Since COBYLA might get stuck, let's try a simple local search on the best result.
    current_best = best_h.copy()
    if np.sum(current_best) > 0:
        current_best = current_best / np.sum(current_best)
        
    step_size = 0.01
    for _ in range(500): # Max steps
        if time.time() > deadline - 2:
            break
            
        improved = False
        # Try modifying one coordinate
        i = rng.randint(N)
        for delta in [-step_size, step_size]:
            h_new = current_best.copy()
            h_new[i] = max(0, h_new[i] + delta * np.sum(current_best)) # Relative step
            # Renormalize to sum 1 to stay on the manifold
            s_new = np.sum(h_new)
            if s_new > 0:
                h_new = h_new / s_new
                s_new_score = _score(h_new)
                if s_new_score > best_score:
                    best_score = s_new_score
                    best_h = h_new.copy()
                    current_best = h_new.copy()
                    improved = True
                    break
        if not improved:
            # Shrink step size slightly
            step_size *= 0.9
            if step_size < 1e-5:
                break

    return list(np.maximum(best_h, 0.0))
