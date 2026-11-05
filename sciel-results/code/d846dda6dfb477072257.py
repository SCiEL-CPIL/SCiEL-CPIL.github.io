import numpy as np
from scipy.optimize import minimize
import time

N = 400

def objective(h):
    # h is flattened
    s = np.sum(h)
    if s == 0:
        return 1e10
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    # To avoid division by zero issues in optimization if s becomes small
    val = 2.0 * N * max_conv / (s * s)
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    best_h = np.ones(N)
    best_val = objective(best_h)
    
    # Try a few random initializations
    # Asymmetric initializations are key
    for i in range(5):
        if time.time() > deadline:
            break
            
        # Linear ramp or random
        if i == 0:
            h0 = np.linspace(1, 2, N)
        elif i == 1:
            h0 = np.random.randn(N)
        else:
            h0 = np.random.uniform(0, 1, N)
        
        # Normalize to sum 1 to fix scale
        h0 = h0 / np.sum(h0)
        
        # Use Powell or Nelder-Mead? 
        # N=400 is too big for Nelder-Mead.
        # L-BFGS-B with numerical grad?
        # Or just use a few steps of gradient descent on a smoothed max.
        
        # Let's try minimizing LogSumExp of conv
        # logsumexp(x) ~ max(x)
        def smooth_obj(h_flat):
            h = h_flat.reshape(N)
            s = np.sum(h)
            if s == 0:
                return 1e10
            conv = np.convolve(h, h)
            # Scale factor for max approximation
            # We want to minimize max(conv) / s^2
            # Equivalent to minimizing log(max(conv)) - 2 log(s)
            # Use logsumexp
            # To avoid overflow, subtract max
            c_max = np.max(conv)
            if c_max <= 0:
                # If all conv are negative, max is negative.
                # But conv[0]=h[0]^2 >= 0. So max is >= 0.
                pass
            lse = c_max + np.log(np.sum(np.exp(conv - c_max)))
            # We want to minimize lse - 2 log(s)
            # But we fixed s=1 in initial step? No, we should allow s to vary or fix it.
            # If we fix s=1, we minimize max(conv).
            # Let's stick to fixing sum=1 by projecting? 
            # Simplest: Minimize f(h) directly with L-BFGS-B.
            
            return lse - 2 * np.log(s) if s > 0 else 1e10

        # L-BFGS-B might struggle with the log(s) if s crosses 0.
        # Let's just use the raw objective with a numerical gradient check? 
        # Or use Powell which is derivative free but slow.
        
        # Let's try a simpler heuristic: 
        # The best found so far in similar tasks often involves making the vector 
        # "anti-symmetric" around the center to cancel out the central peak of convolution.
        
        # Let's try an explicit anti-symmetric construction:
        # h[i] = 1 for i < N/2, -1 for i >= N/2?
        # Sum is 0. Add a small constant?
        
        # Let's just run a quick optimization on the raw objective using scipy.
        # We will use 'L-BFGS-B' with numerical gradient.
        # To make it faster, we can limit iterations.
        
        try:
            res = minimize(
                objective, 
                h0, 
                method='L-BFGS-B', 
                options={'maxiter': 50, 'ftol': 1e-9}
            )
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_h = res.x
        except Exception:
            pass

    return best_h
