"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct step heights h on [0, 2] (represented as N values in [0,1])
    with sum exactly N/2, minimizing the maximum overlap of h with shifted 1-h.
    
    The 'overlap' metric is interpreted as the L2 norm (or similar) of h * (1-h_shifted)
    for all shifts. Given the discrete nature, we consider circular shifts.
    
    Let h be a vector of length N.
    Let g = 1 - h.
    For each shift k, compute overlap_k = sum_i h[i] * g[(i+k) % N].
    We want to minimize max_k overlap_k.
    
    Note: sum_i h[i] * g[(i+k)%N] is the cross-correlation of h and g at lag k.
    Let's denote C_hg(k) = sum_i h[i] * (1-h[(i+k)%N]).
    
    We want to minimize max_k C_hg(k) subject to 0 <= h[i] <= 1 and sum(h) = N/2.
    
    This is a non-convex problem due to the max and the quadratic terms in h.
    However, we can use a smooth approximation or directly optimize using gradient-based methods.
    
    Let's use a soft-max approximation for the max:
    f(h) = logsumexp(C_hg(k) / temperature) * temperature
    
    And use scipy.optimize.minimize with SLSQP or L-BFGS-B.
    """
    start_time = time.time()
    deadline = 150.0  # seconds
    safety_margin = 5.0
    
    # Initial guess: constant 0.5
    h0 = np.full(N, 0.5)
    
    # Constraints: sum(h) = N/2
    # Bounds: 0 <= h[i] <= 1
    
    def compute_correlations(h):
        # Compute circular cross-correlation of h and 1-h
        g = 1.0 - h
        # Use FFT for efficiency
        N_fft = 2 * N
        H = np.fft.rfft(h, N_fft)
        G = np.fft.rfft(g, N_fft)
        corr = np.fft.irfft(H * np.conj(G), N_fft)[:N]
        return corr
    
    def objective_and_grad(h_flat):
        h = h_flat.reshape(-1)
        corr = compute_correlations(h)
        # Soft max approximation
        temp = 0.1
        # To avoid numerical issues, shift by max
        max_corr = np.max(corr)
        log_sum_exp = np.log(np.sum(np.exp((corr - max_corr) / temp))) + max_corr / temp
        val = log_sum_exp * temp
        
        # Gradient computation
        # d/dh logsumexp(corr/temp) * temp = sum_k (exp(corr_k/temp) / sum_j exp(corr_j/temp)) * d(corr_k)/dh
        weights = np.exp((corr - max_corr) / temp)
        weights /= np.sum(weights)
        
        # d(corr_k)/dh_i
        # corr_k = sum_j h[j] * (1 - h[(j+k)%N])
        # d(corr_k)/dh_i = (1 - h[(i+k)%N]) - h[(i-k)%N]  (from the two terms where h_i appears)
        # Wait, let's be precise.
        # corr_k = sum_j h[j] * g[(j+k)%N] where g = 1-h
        # d(corr_k)/dh_i:
        # Term 1: h[i] * g[(i+k)%N] -> derivative is g[(i+k)%N] = 1 - h[(i+k)%N]
        # Term 2: h[i-k] * g[i] -> derivative is -h[i-k] (since g[i] = 1-h[i], d/dh_i g[i] = -1)
        # So d(corr_k)/dh_i = (1 - h[(i+k)%N]) - h[(i-k)%N]
        
        # We need sum_k weights[k] * d(corr_k)/dh_i
        # Let's compute this efficiently.
        # grad_i = sum_k weights[k] * (1 - h[(i+k)%N] - h[(i-k)%N])
        # = sum_k weights[k] - sum_k weights[k] * h[(i+k)%N] - sum_k weights[k] * h[(i-k)%N]
        # = 1 - sum_k weights[k] * h[(i+k)%N] - sum_k weights[k] * h[(i-k)%N]
        
        # Let w = weights
        # term1_i = sum_k w[k] * h[(i+k)%N]
        # term2_i = sum_k w[k] * h[(i-k)%N]
        
        # These are convolutions of w and h.
        # term1 = circular_conv(w, h)
        # term2 = circular_conv(w_rev, h) where w_rev[k] = w[-k] = w[N-k]
        
        w = weights
        w_rev = w[::-1].copy()
        w_rev[0] = w[0]  # w_rev[0] should be w[0]
        
        # Circular convolution
        W = np.fft.rfft(w, N)
        W_rev = np.fft.rfft(w_rev, N)
        H = np.fft.rfft(h, N)
        
        term1 = np.fft.irfft(W * H, N)
        term2 = np.fft.irfft(W_rev * H, N)
        
        grad = 1.0 - term1 - term2
        grad = grad * temp  # Chain rule for temp * logsumexp
        
        return val, grad
    
    # Use L-BFGS-B with constraints
    # Since L-BFGS-B doesn't support equality constraints directly, we can use SLSQP or penalty method.
    # Let's use SLSQP for small N.
    
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2, 'jac': lambda h: np.ones(N)}]
    bounds = [(0, 1)] * N
    
    best_h = h0.copy()
    best_val = np.inf
    
    # Initial evaluation
    val, _ = objective_and_grad(h0)
    best_val = val
    
    # Try a few different temperatures and initializations
    temperatures = [0.5, 0.2, 0.1, 0.05]
    
    for temp in temperatures:
        if time.time() > deadline - safety_margin:
            break
            
        # Modify the objective to use this temperature
        def obj_func(h_flat, temp=temp):
            h = h_flat.reshape(-1)
            corr = compute_correlations(h)
            max_corr = np.max(corr)
            log_sum_exp = np.log(np.sum(np.exp((corr - max_corr) / temp))) + max_corr / temp
            return log_sum_exp * temp
        
        # We need a gradient that matches the temperature
        # For simplicity, let's use numerical gradient if SLSQP struggles, or stick to one temp.
        # Actually, let's just use the first temperature for gradient-based and check time.
        
        # Let's try a simple gradient descent with projection for the sum constraint.
        # Or use scipy minimize with SLSQP.
        
        res = minimize(
            obj_func, 
            h0, 
            method='SLSQP', 
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-8}
        )
        
        if res.success or res.fun < best_val:
            h_opt = res.x
            # Ensure sum constraint is met exactly (SLSQP might be slightly off)
            diff = N/2 - np.sum(h_opt)
            h_opt += diff / N
            # Clip to [0,1]
            h_opt = np.clip(h_opt, 0, 1)
            # Re-normalize if clipping changed the sum
            s = np.sum(h_opt)
            if s > 0:
                h_opt *= (N/2) / s
            
            val, _ = objective_and_grad(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
    
    # Final polish: ensure sum is exactly N/2 and values in [0,1]
    # Simple projection: adjust to meet sum, clip, then adjust again
    h_final = best_h.copy()
    for _ in range(10):
        h_final = np.clip(h_final, 0, 1)
        diff = N/2 - np.sum(h_final)
        # Distribute diff proportionally to slack
        slack = np.where(h_final < 1, 1 - h_final, 0) + np.where(h_final > 0, h_final, 0)
        # Simple: add diff/N
        h_final += diff / N
        h_final = np.clip(h_final, 0, 1)
    
    # Final exact adjustment
    diff = N/2 - np.sum(h_final)
    # Add to the elements with most slack
    idx = np.argsort(h_final)
    if diff > 0:
        # Add to smallest
        for i in idx:
            add = min(diff, 1 - h_final[i])
            h_final[i] += add
            diff -= add
            if diff < 1e-12:
                break
    elif diff < 0:
        # Remove from largest
        for i in idx[::-1]:
            rem = min(-diff, h_final[i])
            h_final[i] -= rem
            diff += rem
            if diff > -1e-12:
                break
    
    return h_final
# EVOLVE-BLOCK-END
