"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 20s safety margin before 150s kill
    
    # Baseline: Flat function
    best_f = np.ones(N)
    best_val = np.inf

    def compute_value(f):
        """Compute normalized max autoconvolution."""
        f = np.asarray(f, dtype=np.float64)
        s = np.sum(f)
        if s < 1e-12:
            return np.inf
        # Autoconvolution
        conv = np.convolve(f, f, mode='full')
        max_conv = np.max(conv)
        # Normalized maximum: max(f*f) / (sum(f))^2 * N
        return max_conv / (s ** 2) * N

    best_val = compute_value(best_f)

    def softmax(x):
        # Numerically stable softmax
        x_shifted = x - np.max(x)
        e_x = np.exp(x_shifted)
        return e_x / np.sum(e_x)

    def objective_and_grad(x):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return best_val, np.zeros_like(x)
            
        f = softmax(x)
        
        # Compute convolution
        conv = np.convolve(f, f, mode='full')
        
        # Log-sum-exp approximation for max
        beta = 50.0
        # To avoid overflow in exp(beta * conv), shift by max
        max_conv = np.max(conv)
        exp_conv = np.exp(beta * (conv - max_conv))
        log_sum_exp = np.log(np.sum(exp_conv)) + beta * max_conv
        obj = log_sum_exp / beta
        
        # Gradient of obj w.r.t conv
        weights = exp_conv / np.sum(exp_conv)
        d_obj_d_conv = weights / beta
        
        # Gradient w.r.t f using FFT
        # d_obj_d_f = correlate(d_obj_d_conv, f) + convolve(d_obj_d_conv, f)
        L = 4 * N
        F = np.fft.fft(f, L)
        U = np.fft.fft(d_obj_d_conv, L)
        
        # Term 1: Correlation sum_k u_k f_{k-i} -> IFFT(U * conj(F))
        # Term 2: Convolution sum_k u_k f_{i-k} -> IFFT(U * F)
        d_obj_d_f_full = np.fft.ifft(U * (np.conj(F) + F)).real
        
        # The valid part is indices 0 to 2N-2
        # We need d_obj_d_f for i=0 to N-1
        d_obj_d_f = d_obj_d_f_full[:N]
        
        # Gradient w.r.t x
        # d_obj_d_x_j = sum_i d_obj_d_f_i * (f_i * (delta_ij - f_j))
        # = f_j * (sum_i d_obj_d_f_i * f_i - d_obj_d_f_j)
        
        sum_d_f_f = np.sum(d_obj_d_f * f)
        grad_x = f * (sum_d_f_f - d_obj_d_f)
        
        return obj, grad_x

    # Optimization
    n_restarts = 10
    best_x = np.zeros(N)
    
    for r in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
            
        if r == 0:
            x0 = np.zeros(N)
        else:
            x0 = np.random.randn(N) * 0.5
            
        try:
            result = minimize(
                objective_and_grad, x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success:
                f_opt = softmax(result.x)
                val = compute_value(f_opt)
                if val < best_val:
                    best_val = val
                    best_f = f_opt.copy()
                    
        except Exception:
            continue
            
    result_heights = np.maximum(best_f, 0)
    return [float(h) for h in result_heights]
# EVOLVE-BLOCK-END
