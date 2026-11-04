import numpy as np
from scipy.optimize import minimize
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial guess: constant function
    # Normalize to sum to 1
    f = np.ones(N) / N
    
    # Precompute indices for convolution
    # g(t) = sum f(x) f(t-x)
    # We can use FFT for speed
    def compute_conv(f_vec):
        # Zero pad to 2*N to avoid circular convolution effects
        # Support of f is 0..N-1. Support of g is 0..2N-2
        # We want linear convolution
        F = np.fft.fft(f_vec, n=2*N)
        G = F * F
        g = np.fft.ifft(G).real[:2*N-1]
        return g

    def objective(params):
        # params are log(f) to enforce positivity
        log_f = params
        f = np.exp(log_f)
        # Normalize
        f = f / np.sum(f)
        
        g = compute_conv(f)
        
        # Soft max of g
        # We want to minimize the max
        # Use log-sum-exp
        max_g = np.max(g)
        # Scale for stability
        tau = 0.01 
        # If max_g is large, g/tau is large
        # log(sum(exp(g/tau))) * tau approximates max
        # But we want the actual max eventually.
        # For optimization, soft max is smoother.
        
        # To avoid overflow, subtract max
        g_scaled = g / tau
        g_scaled = g_scaled - np.max(g_scaled)
        log_sum_exp = np.log(np.sum(np.exp(g_scaled)))
        val = log_sum_exp * tau
        
        # Add penalty if time is running out? No, just return value.
        return val

    def jac(params):
        log_f = params
        f = np.exp(log_f)
        f = f / np.sum(f)
        
        g = compute_conv(f)
        tau = 0.01
        g_scaled = g / tau
        max_g_scaled = np.max(g_scaled)
        exp_g = np.exp(g_scaled - max_g_scaled)
        weights = exp_g / np.sum(exp_g)
        
        # Gradient of soft max
        # d/dx log(sum(exp(g(x)/tau))) = (1/tau) * (sum exp(g/tau) * dg/dx) / sum(exp(g/tau))
        # = sum(weights * dg/dx) / tau
        
        # dg/dx is the convolution kernel
        # g = f * f
        # dg/df = 2 * (f * I) ? No.
        # g(t) = sum_i f(i) f(t-i)
        # d g(t) / d f(k) = f(t-k) + f(k-t) (if linear conv defined appropriately)
        # Actually, gradient of sum_t w(t) g(t) wrt f(k)
        # = sum_t w(t) (f(t-k) + f(k-t))
        # This is a convolution of w with f, plus reverse.
        
        # Simpler: use numerical gradient or just use SLSQP with no jac?
        # SLSQP with finite diff is slow.
        # Let's use a simple gradient step manually or use L-BFGS-B with finite diff?
        # Given 600 variables, finite diff is 600 evals.
        # 140s / 600 = 0.23s per eval.
        # FFT conv is fast.
        # Let's try to compute analytic gradient.
        
        # w(t) are the weights from soft max.
        # grad_f[k] = sum_t w(t) * (f[t-k] + f[k-t])
        # This is (w * f)[k] + (w * f_rev)[k] ?
        # Let's just use finite differences for simplicity and robustness, 
        # but limit iterations.
        
        # Actually, let's just use scipy minimize with method='L-BFGS-B'
        # and let it handle gradients if we provide them, or finite diff.
        # Providing analytic gradient is better.
        
        # Convolution of w and f:
        # w is defined on 0..2N-2
        # f is defined on 0..N-1
        # We need the gradient wrt f.
        
        # Let's stick to a simpler approach:
        # Use a parametric family? No, "fresh design from scratch" and "Fourier" hint suggests full vector.
        
        # Let's use a few random restarts with a simple gradient descent.
        pass

    # Given the complexity and time limit, a robust approach is:
    # 1. Try a few parametric shapes (Gaussian, Box, Two-Box)
    # 2. Use scipy minimize on the full vector with a strict time limit.
    
    best_f = f.copy()
    best_val = objective(np.log(f))
    
    # Define constraints
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(np.exp(x)) - 1}]
    bounds = [(-10, 10)] * N # Log space
    
    # Run optimization
    # We will run a short optimization
    try:
        res = minimize(objective, np.log(f), method='L-BFGS-B', 
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': 50, 'disp': False})
        if res.success:
            f_opt = np.exp(res.x)
            f_opt = f_opt / np.sum(f_opt)
            val_opt = objective(res.x)
            if val_opt < best_val:
                best_f = f_opt
                best_val = val_opt
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        return list(best_f)
        
    # Try a second run with different initialization
    # e.g. Gaussian
    x = np.linspace(-0.25, 0.25, N)
    f_gauss = np.exp(-10 * x**2)
    f_gauss = f_gauss / np.sum(f_gauss)
    
    try:
        res2 = minimize(objective, np.log(f_gauss), method='L-BFGS-B', 
                        bounds=bounds, constraints=constraints,
                        options={'maxiter': 50, 'disp': False})
        if res2.success:
            f_opt2 = np.exp(res2.x)
            f_opt2 = f_opt2 / np.sum(f_opt2)
            val_opt2 = objective(res2.x)
            if val_opt2 < best_val:
                best_f = f_opt2
                best_val = val_opt2
    except Exception:
        pass

    return list(best_f)
