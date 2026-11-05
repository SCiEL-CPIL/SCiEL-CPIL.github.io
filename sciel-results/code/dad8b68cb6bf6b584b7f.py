import numpy as np
import scipy.optimize as opt
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    best_h = np.ones(N)
    best_val = np.inf
    
    # Objective function to minimize
    # We want to minimize max(c).
    # Use log-sum-exp for smoothness.
    beta = 50.0  # sharpness of max approximation
    
    def objective(h):
        # h is assumed to be normalized to sum 1 by the caller or handled here?
        # It's better to handle scaling outside or inside.
        # Let's enforce sum(h)=1 inside the objective?
        # No, minimize on the manifold is better.
        # But scipy minimize doesn't handle manifolds well.
        # We can use the trick: h = h_base + delta, where delta sums to 0.
        # Or just penalize the sum.
        
        # Calculate autoconvolution
        c = np.convolve(h, h, mode='full')
        
        # Soft max
        # c can be large. Scale it?
        # If h is small, c is small.
        # Let's assume h is normalized to sum 1.
        # Then c is O(1/N).
        
        # log-sum-exp trick
        c_max = np.max(c)
        if c_max == 0:
            return 0.0
        # log(sum(exp(beta * c)))
        # To avoid overflow, subtract max
        log_sum_exp = c_max + np.log(np.sum(np.exp(beta * (c - c_max))))
        
        # Penalty for sum != 1
        s = np.sum(h)
        penalty = 1e6 * (s - 1.0)**2
        
        return log_sum_exp / beta + penalty

    # We need to optimize h.
    # Start with several random initializations.
    num_restarts = 100
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Mix of positive and negative to allow cancellation
        h0 = np.random.randn(N)
        # Ensure sum is not 0
        if np.abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        h0 = h0 / np.sum(h0)
        
        # Run L-BFGS-B
        # Bounds: none (unconstrained)
        try:
            res = opt.minimize(
                objective, 
                h0, 
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            h_opt = res.x
            s = np.sum(h_opt)
            if np.abs(s) < 1e-6:
                continue
            h_norm = h_opt / s
            
            # Calculate true value
            c = np.convolve(h_norm, h_norm, mode='full')
            max_c = np.max(c)
            val = 2 * N * max_c / (s**2) # s is 1 now, so just 2*N*max_c
            
            if val < best_val:
                best_val = val
                best_h = h_norm.copy()
                
        except Exception:
            continue

    # Final polish: if we have a good h, do a few more iterations with higher beta?
    # Or just return best_h
    
    return best_h
