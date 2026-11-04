import numpy as np
from scipy.optimize import minimize
import time

N = 600
DELTA = (1.0 / 2.0) / N  # interval width = 1/1200

def _compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    We assume the normalization is by the square of the L1 norm (integral).
    Score = max(f*f) / (integral f)^2
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-9):
        return 1e10
    h = np.maximum(h, 0)
    
    integral = np.sum(h) * DELTA
    if integral <= 1e-10:
        return 1e10
        
    # Compute autoconvolution via FFT
    # The autoconvolution of a step function sampled at midpoints
    # approximates the continuous autoconvolution at knot points.
    # For non-negative functions, the max is at t=0.
    # (f*f)(0) = sum(h^2) * DELTA
    
    # However, to be safe and match potential discrete definitions, 
    # we compute the full discrete convolution.
    h_conv = np.convolve(h, h, mode='full')
    max_ac = np.max(h_conv) * DELTA
    
    return max_ac / (integral ** 2)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Strategy: Represent h(x) as |sum_{k=0}^{K} c_k e^{i 2 pi k x / L}|^2
    # This ensures h(x) >= 0.
    # We optimize the complex coefficients c_k.
    
    # Number of modes. More modes allow more flexibility.
    # 600 points. Let's use up to 20 modes for a smooth shape.
    n_modes = 20
    
    # Grid points
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    
    def params_to_h(params):
        # params: [re(c0), im(c0), re(c1), im(c1), ...]
        # c0 is real (DC component)
        c = np.zeros(n_modes, dtype=complex)
        for i in range(n_modes):
            re = params[2*i]
            im = params[2*i+1] if i > 0 else 0.0
            c[i] = re + 1j*im
        
        h = np.zeros(N, dtype=float)
        for k in range(n_modes):
            if k == 0:
                h += c[0].real**2
            else:
                # |sum c_k e^{ikx}|^2 = sum c_k c_l e^{i(k-l)x}
                # Efficient computation:
                pass
        
        # Direct computation for clarity and speed with small N_modes
        # h(x) = | \sum_{k=0}^{K} c_k e^{i 2 pi k x} |^2
        # Note: The frequency should be scaled to the domain.
        # Domain length 0.5. Period 0.5?
        # Let's use frequencies 2*pi*k / 0.5 = 4*pi*k.
        
        h_complex = np.zeros(N, dtype=complex)
        for k in range(n_modes):
            freq = 4.0 * np.pi * k
            h_complex += c[k] * np.exp(1j * freq * x)
        
        h = np.abs(h_complex)**2
        return h

    def objective(params):
        h = params_to_h(params)
        return _compute_score(h)
    
    # Initial guess: Uniform
    # Uniform corresponds to c[0] = 1, others 0.
    # h = 1.
    best_h = np.ones(N)
    best_score = _compute_score(best_h)
    
    # Optimization
    n_params = 2 * n_modes
    x0 = np.zeros(n_params)
    x0[0] = 1.0  # c[0] = 1
    
    # Try L-BFGS-B
    try:
        res = minimize(objective, x0, method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5})
        if res.fun < best_score:
            best_score = res.fun
            best_h = params_to_h(res.x)
    except Exception:
        pass
    
    # Multi-start with random Gaussian-like coefficients
    rng = np.random.default_rng(42)
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        
        x0 = np.zeros(n_params)
        x0[0] = 1.0
        # Add small random perturbations
        x0[2:] += rng.normal(0, 0.1, size=n_params-2)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 50, 'ftol': 1e-6})
            if res.fun < best_score:
                best_score = res.fun
                best_h = params_to_h(res.x)
        except Exception:
            pass

    # Final check: Ensure non-negative and return
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-10:
        best_h = np.ones(N)
        
    return list(best_h)
