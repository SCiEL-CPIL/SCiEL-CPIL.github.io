import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _score(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for step heights."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)  # enforce nonnegativity
    if np.all(h == 0):
        return 0.0
    # Interval width
    dx = 1.0 / N
    # L1 and L2 norms (continuous)
    l1 = np.sum(h) * dx
    l2_sq = np.sum(h**2) * dx
    if l1 == 0:
        return 0.0
    # Compute autoconvolution (h*h)(x) = integral of h(y)*h(x-y) dy
    # For step functions, we can compute this efficiently
    # The autoconvolution is a piecewise linear function.
    # We need its supremum.
    # 
    # For step functions on equal intervals, the autoconvolution at any point
    # can be computed. The maximum occurs at some point.
    # 
    # Efficient approach: use FFT-based convolution
    # Pad to avoid circular convolution issues
    n_fft = 2 * N + 1
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h
    # Linear convolution via FFT
    conv = np.fft.ifft(np.fft.fft(h_pad) ** 2).real
    # The autoconvolution in continuous terms:
    # (h*h)(x) for step functions: the discrete convolution gives us
    # the values at the interval boundaries, scaled by dx
    # Actually, let's think more carefully.
    #
    # h(x) = sum_i h[i] * 1_{[i*dx, (i+1)*dx)}(x + offset)
    # where the intervals cover [-1/4, 1/4]
    #
    # The autoconvolution (h*h)(t) = integral h(y) h(t-y) dy
    # For step functions, this is a piecewise linear function.
    #
    # Using the discrete convolution approach:
    # If we define the step function values, the autoconvolution at the
    # midpoints or boundaries can be approximated.
    #
    # Actually, for the purpose of finding the supremum, let's sample
    # the autoconvolution at many points.
    
    # Better: compute the autoconvolution exactly for step functions
    # The autoconvolution of two step functions on equal intervals is
    # a piecewise quadratic function (actually piecewise linear for 
    # the integral of product of two step functions).
    #
    # Let me use a direct approach: sample at many points
    # The support of h is [-1/4, 1/4], so h*h has support [-1/2, 1/2]
    # Sample at fine grid
    n_samples = 2000
    x = np.linspace(-0.5, 0.5, n_samples)
    dx_sample = x[1] - x[0]
    
    # h(x) for arbitrary x
    # h is defined on [-1/4, 1/4] with N intervals
    # Interval i covers [-1/4 + i*dx, -1/4 + (i+1)*dx)
    # h(x) = h[i] if x in interval i, else 0
    
    def h_func(x_arr):
        """Evaluate step function h at points x_arr."""
        # Shift so that intervals start at 0
        x_shifted = x_arr + 0.25
        # Map to interval indices
        idx = np.floor(x_shifted / dx).astype(int)
        # Valid if 0 <= idx < N and x in [-1/4, 1/4)
        valid = (idx >= 0) & (idx < N) & (x_arr >= -0.25) & (x_arr < 0.25)
        result = np.zeros_like(x_arr)
        result[valid] = h[idx[valid]]
        return result
    
    # Autoconvolution: (h*h)(t) = integral h(y) h(t-y) dy
    # Use numerical integration
    autoconv = np.zeros_like(x)
    for i, t in enumerate(x):
        # h(y) * h(t-y), integrate over y
        # y ranges such that both y and t-y are in [-1/4, 1/4]
        # y in [max(-1/4, t-1/4), min(1/4, t+1/4)]
        y_min = max(-0.25, t - 0.25)
        y_max = min(0.25, t + 0.25)
        if y_min >= y_max:
            autoconv[i] = 0.0
        else:
            # Sample for integration
            n_int = 500
            y = np.linspace(y_min, y_max, n_int)
            dy = y[1] - y[0] if n_int > 1 else (y_max - y_min)
            vals = h_func(y) * h_func(t - y)
            autoconv[i] = np.sum(vals) * dy
    
    sup_norm = np.max(autoconv)
    if sup_norm <= 0:
        return 0.0
    
    return l2_sq / (l1 * sup_norm)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Try some structured initial guesses
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Triangular
    t = np.linspace(0, 1, N)
    candidates.append(2 * np.abs(t - 0.5))
    candidates.append(1 - 2 * np.abs(t - 0.5))
    
    # Sinusoidal
    candidates.append(1 + 0.5 * np.sin(2 * np.pi * t))
    candidates.append(1 + 0.5 * np.cos(2 * np.pi * t))
    
    # Bump-like (concentrated in middle)
    candidates.append(np.exp(-10 * (t - 0.5)**2))
    candidates.append(np.exp(-5 * (t - 0.5)**2))
    
    # Sparse patterns
    sparse1 = np.zeros(N)
    sparse1[::5] = 1.0
    candidates.append(sparse1)
    
    sparse2 = np.zeros(N)
    sparse2[::10] = 1.0
    candidates.append(sparse2)
    
    # Alternating
    candidates.append(np.array([1.0, 0.0] * (N // 2) + [1.0] if N % 2 else [1.0, 0.0] * (N // 2)))
    
    # Evaluate all candidates
    for c in candidates:
        c = np.maximum(c, 0)
        s = _score(c)
        if s > best_score:
            best_score = s
            best_h = c.copy()
    
    # Now try local optimization from best candidate
    if time.time() < deadline:
        def neg_score(h):
            return -_score(h)
        
        # Use L-BFGS-B for bounded optimization
        bounds = [(0.0, 10.0)] * N
        try:
            res = minimize(neg_score, best_h, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success or res.fun < -best_score:
                h_opt = np.maximum(res.x, 0)
                s_opt = _score(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Try Nelder-Mead if time permits
    if time.time() < deadline - 20:
        try:
            res = minimize(neg_score, best_h, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            h_nm = np.maximum(res.x, 0)
            s_nm = _score(h_nm)
            if s_nm > best_score:
                best_score = s_nm
                best_h = h_nm.copy()
        except Exception:
            pass
    
    # Final check
    if time.time() > deadline:
        # Return best found so far
        return list(best_h)
    
    return list(best_h)
