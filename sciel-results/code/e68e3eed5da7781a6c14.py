import numpy as np
import time
from scipy.optimize import minimize

N = 600
W = 1.0 / 1200.0

def _compute_ratio(h):
    """Compute max(autoconv) / l2_sq for step function h (nonnegative)."""
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    l2_sq = np.sum(h * h) * W
    if l2_sq < 1e-15:
        return float('inf')
    
    # Autoconvolution via FFT
    n_fft = 2048
    F = np.fft.rfft(h, n_fft)
    conv = np.fft.irfft(F * F, n_fft)
    # The autoconvolution of h with itself: (h*h)(x) = sum_j h(x-j*W)*h(j*W)*W
    # conv[k] corresponds to sum_j h[j]*h[k-j] * W (with appropriate scaling)
    # We need max over all shifts
    max_conv = np.max(conv) * W
    return max_conv / l2_sq

def _fourier_to_heights(coeffs, n_coeffs):
    """Convert Fourier coefficients to nonnegative step heights."""
    # coeffs[0] = DC, coeffs[1:] = complex pairs (real, imag) for k=1..n_coeffs
    h = np.zeros(N)
    # DC component
    dc = coeffs[0]
    h[:] = dc
    # Add harmonics
    for k in range(1, n_coeffs + 1):
        idx = 1 + 2*(k-1)
        if idx + 1 >= len(coeffs):
            break
        a = coeffs[idx]
        b = coeffs[idx+1]
        h += a * np.cos(2*np.pi*k*np.arange(N)/N) + b * np.sin(2*np.pi*k*np.arange(N)/N)
    return np.clip(h, 0.0, None)

def _objective(params, n_coeffs):
    """Objective: maximize ratio (minimize negative)."""
    h = _fourier_to_heights(params, n_coeffs)
    if np.sum(h) < 1e-15:
        return 0.0
    r = _compute_ratio(h)
    if not np.isfinite(r):
        return 0.0
    return -r

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_ratio = _compute_ratio(best_h)
    
    n_coeffs = 15  # Number of harmonic pairs to use
    
    # Generate diverse initial points in Fourier space
    np.random.seed(12345)
    
    # Strategy: many restarts with Nelder-Mead on Fourier parameters
    n_restarts = 30
    max_iters_per_restart = 200
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Generate initial Fourier coefficients
        if restart == 0:
            # Start near uniform (small perturbations)
            init_params = np.zeros(1 + 2*n_coeffs)
            init_params[0] = 1.0
            for k in range(1, n_coeffs+1):
                init_params[1+2*(k-1)] = np.random.randn() * 0.1
                init_params[2+2*(k-1)] = np.random.randn() * 0.1
        elif restart == 1:
            # Start with a cosine bump
            init_params = np.zeros(1 + 2*n_coeffs)
            init_params[0] = 0.5
            init_params[1] = 0.5  # cos(pi*x) component
        elif restart == 2:
            # Start with two bumps
            init_params = np.zeros(1 + 2*n_coeffs)
            init_params[0] = 0.3
            init_params[1] = 0.4
            init_params[3] = 0.3  # 2nd harmonic
        elif restart == 3:
            # Start with Gaussian-like (broad spectrum)
            init_params = np.zeros(1 + 2*n_coeffs)
            init_params[0] = 0.8
            for k in range(1, n_coeffs+1):
                init_params[1+2*(k-1)] = np.random.randn() * 0.3 / k
                init_params[2+2*(k-1)] = np.random.randn() * 0.3 / k
        else:
            # Random restarts with varying DC and harmonic content
            dc = np.random.uniform(0.2, 1.5)
            init_params = np.zeros(1 + 2*n_coeffs)
            init_params[0] = dc
            # Random amplitudes, decaying with frequency
            for k in range(1, n_coeffs+1):
                amp = np.random.uniform(-0.5, 0.5) / (1 + 0.5*k)
                init_params[1+2*(k-1)] = amp
                init_params[2+2*(k-1)] = amp * np.random.uniform(-1, 1)
        
        # Evaluate initial point
        h_init = _fourier_to_heights(init_params, n_coeffs)
        if np.sum(h_init) > 1e-15:
            r_init = _compute_ratio(h_init)
            if np.isfinite(r_init) and r_init < best_ratio:
                best_ratio = r_init
                best_h = h_init.copy()
        
        # Run Nelder-Mead
        try:
            result = minimize(
                _objective, 
                init_params,
                args=(n_coeffs,),
                method='Nelder-Mead',
                options={'maxiter': max_iters_per_restart, 
                         'xatol': 1e-6, 'fatol': 1e-8,
                         'adaptive': True}
            )
            
            if result.success or result.fun < 0:
                h_opt = _fourier_to_heights(result.x, n_coeffs)
                if np.sum(h_opt) > 1e-15:
                    r_opt = _compute_ratio(h_opt)
                    if np.isfinite(r_opt) and r_opt < best_ratio:
                        best_ratio = r_opt
                        best_h = h_opt.copy()
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 20.0:
            break
    
    # Local refinement in height space on the best candidate
    if time.time() < deadline - 5.0:
        h = best_h.copy()
        step_size = max(0.005, np.max(h) * 0.005)
        
        for iteration in range(200):
            if time.time() > deadline - 2.0:
                break
            
            improved = False
            n_trials = 50
            idxs = np.random.choice(N, size=n_trials, replace=False)
            
            for idx in idxs:
                h_old = h[idx]
                
                # Try increasing
                h[idx] = h_old + step_size
                r_plus = _compute_ratio(h)
                
                # Try decreasing
                h[idx] = h_old - step_size
                r_minus = _compute_ratio(h)
                
                h[idx] = h_old
                
                if r_plus < best_ratio:
                    h[idx] = h_old + step_size
                    best_ratio = r_plus
                    improved = True
                elif r_minus < best_ratio and h[idx] > 0:
                    h[idx] = h_old - step_size
                    best_ratio = r_minus
                    improved = True
            
            if not improved:
                step_size *= 0.7
                if step_size < 1e-7:
                    break
        
        final_r = _compute_ratio(h)
        if final_r < best_ratio:
            best_ratio = final_r
            best_h = h.copy()
    
    return best_h.tolist()
