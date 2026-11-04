import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _ratio_from_h(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h**2)
    f = np.convolve(h, h, mode='full')
    f_inf = np.max(f)
    if f_inf <= 0:
        return 0.0
    return l2_sq / (l1 * f_inf)

def _soft_max(f, temp=0.01):
    """Smooth approximation to max using log-sum-exp."""
    max_f = np.max(f)
    if max_f <= 0:
        return 0.0
    shifted = f / temp
    shifted -= np.max(shifted)
    return temp * (np.max(f) + np.log(np.sum(np.exp(shifted))))

def _neg_log_ratio_params(params, n=N):
    """
    Negative log of the ratio, parameterized by complex DFT coefficients.
    params: real vector of length 2*n (real and imaginary parts of n complex numbers)
    h = |DFT(params_complex)|^2, which ensures non-negativity.
    """
    # Unpack complex vector
    z = params[:n] + 1j * params[n:]
    
    # Compute h = |DFT(z)|^2
    # Using FFT for efficiency
    Z = np.fft.fft(z)
    h = np.abs(Z)**2
    
    # Ensure non-negative (should be by construction, but clip tiny negatives)
    h = np.maximum(h, 0)
    
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 1e10
    
    l2_sq = np.sum(h**2)
    
    # Compute autoconvolution via FFT
    # For full convolution of length n with itself, we need FFT of size >= 2n-1
    fft_size = 2 * n
    H = np.fft.fft(h, fft_size)
    F = H * H  # Autoconvolution in frequency domain
    f = np.real(np.fft.ifft(F))
    
    # Soft max for smoothness
    f_inf = _soft_max(f, temp=0.01)
    if f_inf <= 1e-15:
        return 1e10
    
    ratio = l2_sq / (l1 * f_inf)
    if ratio <= 0:
        return 1e10
    
    return -np.log(ratio)

def construct():
    start_time = time.time()
    timeout = 110  # Safety margin for 120s limit
    
    best_h = np.ones(N)
    best_ratio = _ratio_from_h(best_h)
    
    # Multiple random restarts with DFT parameterization
    n_restarts = 15
    
    for restart in range(n_restarts):
        if time.time() - start_time > timeout:
            break
        
        # Initialize complex vector with random values
        if restart == 0:
            # Start with a structured initial guess
            t = np.arange(N, dtype=float) / (N - 1)
            alpha = 1.5
            h_init = t ** alpha
            # Invert to get z such that |FFT(z)|^2 ≈ h_init
            # This is an inverse problem; use simple approach
            z = np.fft.ifft(np.sqrt(h_init / np.sum(h_init)) * np.exp(1j * np.random.randn(N)))
            z = z / np.sqrt(np.mean(np.abs(z)**2) + 1e-10)
        else:
            # Random initialization with varying scales
            scale = 0.5 + np.random.rand() * 2.0
            z = (np.random.randn(N) + 1j * np.random.randn(N)) * scale
            z = z / np.sqrt(np.mean(np.abs(z)**2) + 1e-10)
        
        # Pack into real parameter vector
        params = np.concatenate([z.real, z.imag])
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                _neg_log_ratio_params,
                params,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.success or result.fun < 1e10:
                # Extract h from optimized parameters
                z_opt = result.x[:N] + 1j * result.x[N:]
                Z_opt = np.fft.fft(z_opt)
                h_opt = np.abs(Z_opt)**2
                h_opt = np.maximum(h_opt, 0)
                
                # Normalize (ratio is scale-invariant)
                if np.sum(h_opt) > 0:
                    ratio = _ratio_from_h(h_opt)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_h = h_opt.copy()
        except Exception:
            pass
    
    # Coordinate descent refinement on the best solution found
    if time.time() - start_time < timeout - 30:
        h = best_h.copy()
        for _ in range(200):
            if time.time() - start_time > timeout - 5:
                break
            
            improved = False
            for i in range(N):
                old_val = h[i]
                # Try a range of values around current
                candidates = [old_val * 0.5, old_val * 0.75, old_val * 1.25, old_val * 1.5, old_val + 0.1, old_val - 0.1]
                candidates = [max(0, c) for c in candidates]
                
                best_local = h.copy()
                best_local_ratio = _ratio_from_h(h)
                
                for c in candidates:
                    test_h = h.copy()
                    test_h[i] = c
                    test_ratio = _ratio_from_h(test_h)
                    if test_ratio > best_local_ratio:
                        best_local_ratio = test_ratio
                        best_local = test_h.copy()
                
                if best_local_ratio > _ratio_from_h(h) + 1e-10:
                    h = best_local
                    improved = True
            
            if not improved:
                break
        
        final_ratio = _ratio_from_h(h)
        if final_ratio > best_ratio:
            best_ratio = final_ratio
            best_h = h
    
    return list(best_h)
