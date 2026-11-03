import numpy as np
from scipy.optimize import minimize
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def objective(h):
        h = np.maximum(h, 0)  # enforce nonnegativity
        s = np.sum(h)
        if s < 1e-10:
            return 1e10
        # Compute autocorrelation at all lags
        # autocorr[m] = sum_i h[i]*h[i+m] for m=0..N-1
        # Use FFT for efficiency
        n_fft = 2 * N
        H = np.fft.rfft(h, n=n_fft)
        autocorr = np.fft.irfft(H * np.conj(H), n=n_fft)[:N]
        # autocorr[m] = sum_i h[i]*h[i+m] (with circular wrap, but since we padded, it's linear)
        max_ac = np.max(autocorr)
        return max_ac / (s * s)
    
    # Initial guess: uniform
    h0 = np.ones(N)
    
    # Try a few different initializations and optimizations
    best_h = h0.copy()
    best_val = objective(h0)
    
    # Try uniform
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        res = minimize(objective, h0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < best_val:
            best_val = res.fun
            best_h = res.x
        elapsed = time.time() - start_time
    
    # Try a triangular or other shape
    if elapsed < deadline - 5:
        h_tri = np.abs(np.linspace(-1, 1, N))  # V-shaped
        h_tri = h_tri / np.sum(h_tri)
        res = minimize(objective, h_tri, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < best_val:
            best_val = res.fun
            best_h = res.x
        elapsed = time.time() - start_time
    
    # Try a Gaussian-like shape
    if elapsed < deadline - 5:
        x = np.linspace(-1, 1, N)
        h_gauss = np.exp(-x**2 * 5)
        h_gauss = h_gauss / np.sum(h_gauss)
        res = minimize(objective, h_gauss, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < best_val:
            best_val = res.fun
            best_h = res.x
        elapsed = time.time() - start_time
    
    # Final polish with more iterations if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 2:
        res = minimize(objective, best_h, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
        if res.fun < best_val:
            best_val = res.fun
            best_h = res.x
    
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0)
    
    return list(best_h)
