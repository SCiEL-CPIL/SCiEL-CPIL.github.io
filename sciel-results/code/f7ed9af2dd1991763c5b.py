import numpy as np
import time
from scipy.optimize import minimize

N = 600
L = 0.5
dx = L / N

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def compute_score(f):
        # f is a numpy array of length N
        # Compute linear autoconvolution
        f_pad = np.zeros(2 * N)
        f_pad[:N] = f
        F = np.fft.fft(f_pad)
        autoconv = np.fft.ifft(F * F).real
        peak = np.max(autoconv)
        integral = np.sum(f)
        if integral < 1e-12:
            return np.inf
        return peak / (integral * integral)

    # Candidate 1: Uniform
    f_uniform = np.ones(N)
    best_f = f_uniform.copy()
    best_score = compute_score(f_uniform)

    # Structural Change: Parametric Optimization
    # Represent f as a sum of cosine terms + constant, clipped to non-negative.
    # f(x) = max(0, c0 + sum_{k=1}^K c_k cos(2*pi*k*x/L))
    # x is the center of each bin.
    x = np.linspace(-L/2, L/2, N, endpoint=False) + dx/2
    
    K = 10  # Number of harmonics
    # Initial parameters: mostly zero, c0 = 1 (uniform-like)
    x0 = np.zeros(K + 1)
    x0[0] = 1.0
    
    def param_to_f(params):
        c0 = params[0]
        coeffs = params[1:]
        f = np.full(N, c0)
        for k in range(1, K + 1):
            f += coeffs[k-1] * np.cos(2 * np.pi * k * x / L)
        # Ensure non-negative
        return np.maximum(f, 0)

    def objective(params):
        if time.time() > deadline:
            return best_score
        f = param_to_f(params)
        if np.sum(f) < 1e-12:
            return np.inf
        return compute_score(f)

    # Run optimization
    # Use L-BFGS-B or Nelder-Mead
    # Nelder-Mead is robust for non-differentiable (due to max(0, .))
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < best_score:
            best_f = param_to_f(res.x)
            best_score = res.fun
    except Exception:
        pass

    # Also try a few random restarts with different initial params if time permits
    if time.time() < deadline - 10:
        for _ in range(5):
            x_rand = np.random.randn(K + 1) * 0.1
            x_rand[0] = 1.0 + np.random.randn() * 0.2
            try:
                res = minimize(objective, x_rand, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.fun < best_score:
                    best_f = param_to_f(res.x)
                    best_score = res.fun
            except Exception:
                pass

    return list(best_f)
