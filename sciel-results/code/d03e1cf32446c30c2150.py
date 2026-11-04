"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))

def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _soft_max_abs(conv, beta=100.0):
    """Smooth approximation of max |conv| using log-sum-exp."""
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    if max_val == 0:
        return 0.0
    shifted = abs_conv - max_val
    return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta

def _objective_and_grad(x, beta):
    """
    Compute the soft-max objective and its gradient w.r.t. x.
    J(x) = soft_max(|conv(x)|)
    grad = 2 * conv(x, K) where K_k = w_k * sign(conv_k)
    """
    fx = np.fft.rfft(x, _FFT_SIZE)
    Y = fx * np.conj(fx)
    conv = np.fft.irfft(Y, _FFT_SIZE)[:2 * N - 1]
    
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    if max_val < 1e-12:
        return 0.0, np.zeros(N)
    
    shifted = abs_conv - max_val
    exp_shifted = np.exp(beta * shifted)
    w = exp_shifted / np.sum(exp_shifted)
    
    G = w * np.sign(conv)
    
    # Gradient computation:
    # grad_j = sum_k G_k * (x_{j-k} + x_{j+k})
    # This is equivalent to 2 * (x * G)_circular (with appropriate padding)
    # Using FFT:
    # grad_freq = 2 * FFT(x) * conj(FFT(G_padded))
    G_padded = np.zeros(_FFT_SIZE)
    G_padded[:2 * N - 1] = G
    G_freq = np.fft.rfft(G_padded, _FFT_SIZE)
    
    grad_freq = 2 * fx * np.conj(G_freq)
    grad = np.fft.irfft(grad_freq, _FFT_SIZE)[:N]
    
    return float(_soft_max_abs(conv, beta)), grad

def _construct():
    deadline = time.time() + 145.0
    
    # Start with a random signal
    rng = np.random.default_rng(2024)
    x_init = rng.normal(size=N)
    
    # Normalize to unit sum for stability
    S = np.sum(x_init)
    if abs(S) > 1e-6:
        x_init = x_init * (N / S)
    
    best_x = x_init.copy()
    best_val = _metric(best_x)
    
    # Multi-restart L-BFGS with adaptive beta
    betas = [50.0, 100.0, 200.0]
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
            
        # Define objective and gradient for L-BFGS
        def obj_grad(x_vec):
            val, grad = _objective_and_grad(x_vec, beta)
            return val, grad
        
        # Run L-BFGS
        try:
            res = minimize(
                obj_grad,
                x_init,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            x_opt = res.x
            val_opt = _metric(x_opt)
            
            if val_opt < best_val:
                best_val = val_opt
                best_x = x_opt.copy()
                x_init = x_opt.copy() # Continue from best found
        except Exception:
            pass
    
    # Final polish with exact metric and local search
    if time.time() < deadline - 5:
        # Simple local search to polish
        step_size = 0.1
        for i in range(1000):
            if time.time() > deadline - 2:
                break
                
            improved = False
            # Random coordinate descent
            idx = rng.integers(0, N)
            for delta in [-step_size, step_size]:
                x_test = best_x.copy()
                x_test[idx] += delta
                v = _metric(x_test)
                if v < best_val:
                    best_val = v
                    best_x = x_test
                    improved = True
                    break
            
            if not improved:
                step_size *= 0.9
                if step_size < 0.001:
                    break
    
    return best_x

def construct():
    return _construct()
# EVOLVE-BLOCK-END
