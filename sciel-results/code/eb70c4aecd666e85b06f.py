import numpy as np
from scipy.optimize import minimize
import time

N = 400

def construct():
    start_time = time.time()
    deadline = 120.0
    
    # Initial guess: uniform
    h0 = np.ones(N) / N
    
    # We will optimize h[0:N-1] and set h[N-1] = 1 - sum(h[0:N-1])
    # This ensures sum(h) = 1
    
    def objective(params):
        h = np.zeros(N)
        h[:N-1] = params
        h[N-1] = 1.0 - np.sum(params)
        
        # Compute autoconvolution
        c = np.convolve(h, h)
        
        # Smoothed max
        # Use beta passed from closure or global? 
        # We'll use a global variable for beta in the loop
        nonlocal beta
        max_c = np.max(c)
        # Shift for stability
        shifted = beta * (c - max_c)
        # Clip to avoid overflow in exp
        shifted = np.clip(shifted, -100, 100)
        log_sum_exp = max_c + (1.0 / beta) * np.log(np.sum(np.exp(shifted)))
        
        return log_sum_exp

    def gradient(params):
        h = np.zeros(N)
        h[:N-1] = params
        h[N-1] = 1.0 - np.sum(params)
        
        c = np.convolve(h, h)
        max_c = np.max(c)
        shifted = beta * (c - max_c)
        shifted = np.clip(shifted, -100, 100)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        
        # Gradient of log_sum_exp w.r.t c
        # d(log_sum_exp)/dc_k = (1/beta) * exp(beta(c_k - max_c)) / sum_exp
        # But max_c also depends on h. This is tricky.
        # For large beta, the gradient is dominated by the max term.
        # Approximation: gradient is just the gradient of the max term.
        # Let k* be argmax(c).
        # c_{k*} = sum_i h_i h_{i+k*}
        # d c_{k*} / d h_j = h_{j+k*} + h_{j-k*} (with bounds)
        
        # Let's use numerical gradient or just SLSQP without gradient?
        # SLSQP can compute finite differences.
        # With N=400, finite differences might be slow.
        # Let's try to provide an analytical gradient for the max term.
        
        # Identify the index of the max
        k_max = np.argmax(c)
        
        # Gradient of c_{k_max} w.r.t h
        # c_k = sum_i h_i h_{i+k}
        # d c_k / d h_j = h_{j+k} + h_{j-k} (if indices valid)
        
        grad_c = np.zeros(N)
        # Term 1: h_i h_{i+k} -> d/dh_j = h_{j+k}
        # Term 2: h_i h_{i-k} (symmetry) -> d/dh_j = h_{j-k}
        
        # Indices for h_{j+k_max}
        # j+k_max < N
        idx1 = np.arange(N - k_max)
        grad_c[idx1] += h[idx1 + k_max]
        
        # Indices for h_{j-k_max}
        # j-k_max >= 0
        idx2 = np.arange(k_max, N)
        grad_c[idx2] += h[idx2 - k_max]
        
        # This is the gradient of the MAX term.
        # The smoothed max gradient is a weighted average, but for large beta, it's close to the max gradient.
        # Let's use this as the gradient.
        
        # Map back to params
        # h[N-1] = 1 - sum(params)
        # d h[N-1] / d params = -1
        # d h[j] / d params[j] = 1 for j < N-1
        
        grad_params = np.zeros(N-1)
        for j in range(N-1):
            grad_params[j] = grad_c[j] - grad_c[N-1]
            
        return grad_params

    beta = 10.0
    best_h = h0.copy()
    best_val = 2.0 # Seed value
    
    # Iterative refinement of beta
    for beta_val in [10, 50, 100, 200, 500, 1000]:
        if time.time() > deadline - 30:
            break
        beta = beta_val
        try:
            res = minimize(objective, best_h[:N-1], method='L-BFGS-B', jac=gradient, options={'maxiter': 200, 'ftol': 1e-10})
            if res.success:
                h_opt = np.zeros(N)
                h_opt[:N-1] = res.x
                h_opt[N-1] = 1.0 - np.sum(res.x)
                
                # Calculate actual value
                c = np.convolve(h_opt, h_opt)
                max_c = np.max(c)
                val = 2 * N * max_c # Since sum is 1
                
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
        except Exception:
            pass

    return best_h
