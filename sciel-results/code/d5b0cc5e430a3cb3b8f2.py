"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize

N = 600

# EVOLVE-BLOCK-START
def _autoconv_max(f):
    """Compute the normalized maximum of the autoconvolution of f."""
    # f is a nonnegative step function on N equal intervals on [-1/4, 1/4]
    # Autoconvolution g(t) = integral f(s)f(t-s) ds
    # For step functions on equal intervals, this is a discrete convolution
    # Normalization: max(g) / (integral(f))^2 * interval_width? 
    # Actually, let's think carefully.
    # 
    # Let dx = (1/4 - (-1/4)) / N = (1/2) / 600 = 1/1200
    # f_i = height on interval i
    # integral(f) ≈ sum(f_i) * dx
    # g(t) = integral f(s) f(t-s) ds
    # At the discrete level, the autoconvolution values are:
    # g_k = dx^2 * sum_{i+j=k} f_i * f_j  (for appropriate k)
    # Actually, more precisely:
    # The autoconvolution at shift k*dx is sum_i f_i * f_{i+k} * dx
    # Wait, let's be more careful.
    # 
    # g(t) = ∫ f(s) f(t-s) ds
    # If f is a step function with value f_i on interval [i*dx, (i+1)*dx),
    # then for t = k*dx, g(t) ≈ sum_i f_i * f_{i-k} * dx (approximately)
    # 
    # The normalized maximum is max_t g(t) / (∫f)^2
    # Since ∫f ≈ dx * sum(f_i), we have (∫f)^2 ≈ dx^2 * (sum(f_i))^2
    # And max_t g(t) ≈ dx * max_k sum_i f_i * f_{i-k}
    # So the ratio ≈ [dx * max_k sum_i f_i f_{i-k}] / [dx^2 * (sum f_i)^2]
    #            = max_k sum_i f_i f_{i-k} / [dx * (sum f_i)^2]
    # 
    # Hmm, that doesn't seem right. Let me reconsider.
    # 
    # Actually, for a step function, the autoconvolution is a piecewise constant
    # function. The maximum of the autoconvolution, normalized by (∫f)^2,
    # is what we want to minimize.
    # 
    # Let's just compute it numerically:
    dx = 1.0 / (2 * N)  # interval width
    # Autoconvolution via FFT
    f_ext = np.zeros(2 * N)
    f_ext[:N] = f
    conv = np.fft.irfft(np.fft.rfft(f_ext) ** 2, n=2 * N)
    # The autoconvolution at discrete points: conv[k] = sum_i f_i * f_{i-k} * dx
    # Actually, np.convolve(f, f, mode='full') gives sum_i f_i f_{i+k}
    # Let's use direct convolution for accuracy
    conv_full = np.convolve(f, f, mode='full')
    # conv_full[k] = sum_i f_i * f_{i-k} for k = 0..2N-2
    # The integral of f is sum(f) * dx
    integral_f = np.sum(f) * dx
    # The autoconvolution value at shift k*dx is conv_full[k] * dx
    # max autoconv = max(conv_full) * dx
    # Normalized max = max(conv_full) * dx / (integral_f)^2
    #                = max(conv_full) * dx / (sum(f)^2 * dx^2)
    #                = max(conv_full) / (sum(f)^2 * dx)
    
    denom = (np.sum(f) ** 2) * dx
    if denom < 1e-15:
        return 1e10
    return np.max(conv_full) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before 150s limit
    
    # Start with uniform distribution
    best_f = np.ones(N)
    best_val = _autoconv_max(best_f)
    
    # Try to improve using local optimization
    # The objective is non-smooth (max of convolution), so use Nelder-Mead or
    # minimize_scalar for 1D perturbations
    
    # Strategy: coordinate descent with exact 1D minimization per coordinate
    # Plus some random restarts
    
    np.random.seed(42)
    
    def objective(f_arr):
        f_arr = np.maximum(0, f_arr)  # Enforce nonnegativity
        return _autoconv_max(f_arr)
    
    # Coordinate descent with scipy.optimize.minimize_scalar
    improved = True
    max_cd_rounds = 50
    
    for round_idx in range(max_cd_rounds):
        if time.time() > deadline - 30:
            break
        progress = False
        for i in range(N):
            if time.time() > deadline - 30:
                break
            f_copy = best_f.copy()
            
            def obj_1d(x):
                f_test = f_copy.copy()
                f_test[i] = max(0, x)
                return _autoconv_max(f_test)
            
            result = minimize_scalar(obj_1d, bounds=(0, max(best_f) * 2 + 1), method='bounded',
                                     options={'maxiter': 100})
            new_val = result.fun
            if new_val < best_val - 1e-10:
                best_f[i] = max(0, result.x)
                best_val = new_val
                progress = True
        if not progress:
            break
    
    # Try some structured perturbations
    for round_idx in range(5):
        if time.time() > deadline - 10:
            break
        
        # Try sinusoidal perturbations
        for freq in [1, 2, 3, 5, 10]:
            if time.time() > deadline - 10:
                break
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = np.random.uniform(0.1, 0.5)
            perturbation = amplitude * np.sin(2 * np.pi * freq * np.arange(N) / N + phase)
            f_test = np.maximum(0, best_f + perturbation)
            val_test = _autoconv_max(f_test)
            if val_test < best_val:
                best_f = f_test
                best_val = val_test
    
    # Final polish: more iterations of coordinate descent
    for round_idx in range(10):
        if time.time() > deadline - 5:
            break
        progress = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            f_copy = best_f.copy()
            
            def obj_1d(x):
                f_test = f_copy.copy()
                f_test[i] = max(0, x)
                return _autoconv_max(f_test)
            
            result = minimize_scalar(obj_1d, bounds=(0, max(best_f) * 2 + 1), method='bounded',
                                     options={'maxiter': 200})
            new_val = result.fun
            if new_val < best_val - 1e-12:
                best_f[i] = max(0, result.x)
                best_val = new_val
                progress = True
        if not progress:
            break
    
    return list(best_f)
# EVOLVE-BLOCK-END
