"""Optimized program for first autocorrelation inequality."""
import numpy as np
from scipy.optimize import minimize
import time

N = 600
dx = 0.5 / N  # interval width = 1/300

def _full_autoconv_max_normalized(heights):
    """Compute max over all lags of autoconvolution, normalized by (integral f)^2."""
    int_f = dx * np.sum(heights)
    if int_f < 1e-15:
        return 1e10
    # Compute autoconvolution at all integer lags using FFT
    # Zero-pad to avoid circular convolution
    size = 2 * N
    h_padded = np.zeros(size)
    h_padded[:N] = heights
    # Autoconvolution via FFT
    fft_h = np.fft.fft(h_padded, n=size)
    autoconv = np.fft.ifft(fft_h * np.conj(fft_h)).real
    # autoconv[k] for k=0..N-1 gives sum_i h[i]*h[i+k] (circular, but zero-padded so linear)
    # Actually: ifft(|FFT|^2)[k] = sum_i h_padded[i] * h_padded[i+k mod size]
    # With zero padding, for k in [0, N-1], this is the linear cross-correlation
    # The full autoconvolution values are:
    # c[k] = dx * sum_i h[i] * h[i-k] for k in -(N-1)..(N-1)
    # Due to symmetry c[k] = c[-k], so we just need k >= 0
    # autoconv[0] = sum h[i]^2
    # autoconv[k] for k=1..N-1 = sum_i h[i]*h[i+k] (with h[j]=0 for j>=N)
    max_ac = np.max(autoconv[:N])  # max over non-negative lags (by symmetry)
    return max_ac / (int_f ** 2)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin
    
    # The problem: minimize max_k c[k] / (dx * sum h)^2 over h >= 0
    # By scale invariance, fix sum(h) = 1 and minimize max_k c[k] / dx
    # where c[k] = sum_i h[i]*h[i-k]
    # This is a convex problem: minimize t s.t. c[k] <= t for all k, h >= 0, sum(h)=1
    
    # Use scipy minimize with SLSQP or trust-constr for the convex problem
    # Variables: h[0..N-1] >= 0, sum(h) = 1
    # Objective: max_k c[k] (convex, piecewise quadratic)
    
    def _obj_and_grad(h):
        """Objective and gradient for the normalized autoconv max."""
        # Normalize
        s = np.sum(h)
        if s < 1e-15:
            return 1e10, np.zeros(N)
        hn = h / s  # normalized so sum = 1
        
        # Compute autoconvolution via FFT
        size = 2 * N
        h_padded = np.zeros(size)
        h_padded[:N] = hn
        fft_h = np.fft.fft(h_padded, n=size)
        autoconv = np.fft.ifft(fft_h * np.conj(fft_h)).real[:N]
        
        # Find the max lag
        k_max = np.argmax(autoconv)
        c_max = autoconv[k_max]
        
        # Objective: c_max (since we fixed sum=1, the normalized value is c_max / dx * dx = c_max... 
        # Wait: normalized = max_ac / (dx * sum h)^2 where sum h = s
        # If we fix sum(h)=1: normalized = max_ac / dx^2 ... no
        # Let me re-derive: 
        # int_f = dx * sum(h) = dx * 1 = dx
        # max_ac = dx * max_k sum_i h[i]*h[i-k] = dx * c_max
        # normalized = (dx * c_max) / (dx * 1)^2 = c_max / dx
        # So minimize c_max / dx, equivalently minimize c_max
        
        # Gradient: d/dh_j c_max where c_max = sum_i hn[i]*hn[i-k_max]
        # This is the correlation at lag k_max
        # c_max = sum_i hn[i] * hn[i-k_max] (with hn[j]=0 for j<0 or j>=N)
        # d c_max / d hn[j] = hn[j-k_max] + hn[j+k_max] (with boundary handling)
        # But hn = h/s, so we need chain rule
        # Actually, let's just use numerical gradient or a different approach
        
        # For SLSQP, we can use the objective without gradient
        return c_max / dx
    
    # Better approach: use the fact that this is a convex QP-like problem
    # Minimize t subject to:
    #   sum_i h[i]*h[i-k] <= t*dx for all k=0..N-1  (since normalized = max_ac/(dx*sum h)^2)
    #   sum(h) = 1
    #   h >= 0
    # 
    # This is a convex program. Let's use scipy's minimize with constraints.
    
    # Actually, let me use a simpler iterative approach:
    # 1. Start with a good initial point
    # 2. Use projected gradient or subgradient method
    # 3. Or use scipy minimize with the full problem
    
    # Let's use scipy minimize with SLSQP on the normalized problem
    # Variables: h[0..N-1], constraints: sum(h)=1, h>=0
    # Objective: max_k (dx * sum_i h[i]*h[i-k]) / (dx * sum(h))^2
    
    # For efficiency with N=600, let's use a reduced representation
    # Try: parameterize h as a sum of a few "bumps" or use a smooth parameterization
    
    # Alternative: use the insight that optimal solutions for such problems
    # often have a specific structure. Let's try a direct convex optimization.
    
    # Use scipy.optimize.minimize with method='trust-constr' or 'SLSQP'
    # But N=600 variables with equality constraint might be slow.
    
    # Let's try a different approach: use the dual or a specialized algorithm.
    
    # Actually, let's just do a careful implementation with scipy minimize:
    
    def _objective_full(h):
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        hn = h / s
        size = 2 * N
        h_padded = np.zeros(size)
        h_padded[:N] = hn
        fft_h = np.fft.fft(h_padded, n=size)
        autoconv = np.fft.ifft(fft_h * np.conj(fft_h)).real[:N]
        return np.max(autoconv) / dx
    
    # Start with uniform
    h0 = np.ones(N) / N
    
    # Try to improve using scipy minimize
    best_h = h0.copy()
    best_val = _objective_full(h0)
    
    # Use L-BFGS-B with bounds (no equality constraint, but we normalize)
    # Actually, let's add the equality constraint properly
    
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}]
    bounds = [(0, None)] * N
    
    # Try a few optimization methods
    methods_to_try = [
        ('L-BFGS-B', {'maxiter': 100, 'ftol': 1e-12}),
        ('SLSQP', {'maxiter': 50, 'ftol': 1e-10}),
    ]
    
    for method, opts in methods_to_try:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _objective_full, h0,
                method=method,
                constraints=constraints,
                bounds=bounds,
                options=opts
            )
            if res.success or res.fun < best_val:
                h_candidate = np.maximum(res.x, 0)
                if np.sum(h_candidate) > 1e-15:
                    h_candidate = h_candidate / np.sum(h_candidate)
                    val = _objective_full(h_candidate)
                    if val < best_val:
                        best_val = val
                        best_h = h_candidate
        except Exception:
            pass
    
    # Also try a water-filling inspired iterative approach:
    # Start with uniform, then iteratively reduce height where autoconv peaks
    if time.time() < deadline - 20:
        h_wf = np.ones(N) / N
        for iteration in range(200):
            if time.time() > deadline:
                break
            # Compute autoconv
            size = 2 * N
            h_padded = np.zeros(size)
            h_padded[:N] = h_wf
            fft_h = np.fft.fft(h_padded, n=size)
            autoconv = np.fft.ifft(fft_h * np.conj(fft_h)).real[:N]
            
            # Find where the autoconv is high relative to neighbors
            # "Water filling": reduce h where it contributes most to the peak
            # The gradient of max_k c[k] w.r.t. h[j] is approximately
            # the contribution of h[j] to the max lag
            k_max = np.argmax(autoconv)
            
            # Gradient of c[k_max] w.r.t. h[j]:
            # c[k] = sum_i h[i]*h[i-k]
            # dc[k_max]/dh[j] = h[j-k_max] + h[j+k_max] (with h out of range = 0)
            grad = np.zeros(N)
            # Contribution from h[j] as first factor: h[j-k_max] for j >= k_max
            if k_max > 0:
                grad[k_max:] += h_wf[:N - k_max]
            # Contribution from h[j] as second factor: h[j+k_max] for j < N-k_max
            if k_max > 0:
                grad[:N - k_max] += h_wf[k_max:]
            if k_max == 0:
                grad[:] = 2 * h_wf
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-15:
                grad /= grad_norm
            
            # Update: h -= step * grad, then project to simplex (h>=0, sum=1)
            step = 0.01
            h_new = h_wf - step * grad
            # Project to probability simplex
            h_new = np.maximum(h_new, 0)
            s = np.sum(h_new)
            if s > 1e-15:
                h_new /= s
            else:
                h_new = h_wf
            
            val_new = _objective_full(h_new)
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
            h_wf = h_new
    
    # Return the best heights
    return [max(0.0, h) for h in best_h]
