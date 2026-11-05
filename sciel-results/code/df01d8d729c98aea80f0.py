import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autocorr_fft(h):
    """Fast autocorrelation via FFT."""
    n = len(h)
    nfft = 2 ** (2 * n - 1).bit_length()
    H = np.fft.rfft(h, n=nfft)
    ac = np.fft.irfft(H * np.conj(H), n=nfft)[:2 * n - 1]
    return ac

def _value(h):
    """Compute the objective value."""
    S = float(np.sum(h))
    if abs(S) < 1e-10:
        return np.inf
    c = _autocorr_fft(h)
    return abs(2.0 * N * float(np.max(c)) / (S * S))

def _smooth_max_value(h, alpha=10.0):
    """Smooth maximum surrogate for optimization."""
    S = float(np.sum(h))
    if abs(S) < 1e-10:
        return np.inf
    c = _autocorr_fft(h)
    # Smooth max: log-sum-exp approximation
    c_max = np.max(c)
    if c_max <= 0:
        return 0.0
    # Use scaled log-sum-exp for smooth max
    c_scaled = c / c_max
    smooth_max = c_max * (1.0 + np.log(np.sum(np.exp(alpha * c_scaled - alpha))) / alpha)
    return abs(2.0 * N * smooth_max / (S * S))

def _golay_complementary_h():
    """Generate Golay complementary pair-based sequence."""
    # Golay complementary sequences have the property that their
    # autocorrelation sidelobes cancel each other
    # For N=400, we can use two length-200 Golay sequences
    
    # Generate a Golay complementary pair using recursive construction
    # Start with base sequences of length 2
    # G1 = [1, 1], G2 = [1, -1]
    # Recursively build up to length 256, then pad/trim to 200
    
    def golay_pair(length):
        """Generate Golay complementary pair of given power-of-2 length."""
        if length == 1:
            return np.array([1.0]), np.array([1.0])
        if length == 2:
            return np.array([1.0, 1.0]), np.array([1.0, -1.0])
        
        half = length // 2
        a, b = golay_pair(half)
        # Recursive construction
        a_new = np.concatenate([a, b])
        b_new = np.concatenate([a, -b])
        return a_new, b_new
    
    # Get length-256 Golay pair
    g1, g2 = golay_pair(256)
    
    # Pad to length 200 by trimming (or we could use different approach)
    # Actually, let's use a different strategy: create two chirps
    # that have complementary autocorrelation properties
    
    # Use Zadoff-Chu-like sequences with different rates
    i = np.arange(N, dtype=float)
    i_centered = i - N / 2.0
    
    # Two chirps with different quadratic phases
    rate1 = 2.0 * np.pi / N
    rate2 = 3.0 * np.pi / N
    
    h1 = np.cos(rate1 * i_centered ** 2)
    h2 = np.cos(rate2 * i_centered ** 2 + np.pi / 4)
    
    # Combine with optimal mixing
    # The sum h = h1 + h2 should have reduced peak autocorrelation
    h = h1 + h2
    
    # Normalize to have sum = N
    h = h * (N / np.sum(h))
    
    return h

def _dct_basis_matrix(N):
    """Create DCT basis matrix."""
    n = N
    k = np.arange(n)
    m = np.arange(n)
    # DCT-II basis
    B = np.cos(np.pi * (2 * m[:, None] + 1) * k[None, :] / (2 * n))
    return B

def _dct_h(h):
    """Compute DCT coefficients."""
    n = len(h)
    k = np.arange(n)
    m = np.arange(n)
    coeffs = np.cos(np.pi * (2 * m[None, :] + 1) * k[:, None] / (2 * n)) @ h
    return coeffs

def _idct_h(coeffs):
    """Inverse DCT."""
    n = len(coeffs)
    k = np.arange(n)
    m = np.arange(n)
    h = (2.0 / n) * (np.cos(np.pi * (2 * m[:, None] + 1) * k[None, :] / (2 * n)) @ coeffs)
    # First coefficient needs special handling
    h[0] = (1.0 / n) * np.sum(coeffs)
    return h

def _dct_smooth_obj(coeffs, alpha=10.0):
    """Smooth max objective in DCT domain."""
    h = _idct_h(coeffs)
    return _smooth_max_value(h, alpha)

def _zadoff_chu_h():
    """Generate windowed Zadoff-Chu sequence."""
    i = np.arange(N, dtype=float)
    i_centered = i - N / 2.0
    
    # Zadoff-Chu: exp(j * pi * u * i^2 / N)
    u = 1  # Chirp rate
    phase = np.pi * u * i_centered ** 2 / N
    h = np.cos(phase)
    
    # Apply a raised cosine window to reduce edge effects
    window = 0.5 * (1 + np.cos(2 * np.pi * i / N))
    h = h * window
    
    # Normalize
    h = h * (N / np.sum(h))
    
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_h = np.ones(N)
    best_val = _value(best_h)
    
    # Strategy 1: Golay complementary pair based
    h_golay = _golay_complementary_h()
    val_golay = _value(h_golay)
    if val_golay < best_val:
        best_val = val_golay
        best_h = h_golay.copy()
    
    # Strategy 2: Zadoff-Chu based
    h_zc = _zadoff_chu_h()
    val_zc = _value(h_zc)
    if val_zc < best_val:
        best_val = val_zc
        best_h = h_zc.copy()
    
    # Strategy 3: DCT spectral shaping optimization
    # Start from best candidate and optimize in DCT domain
    h0 = best_h.copy()
    coeffs0 = _dct_h(h0)
    
    # Use smooth max surrogate for optimization
    def smooth_obj(coeffs):
        return _dct_smooth_obj(coeffs, alpha=5.0)
    
    # Optimize DCT coefficients
    res = minimize(smooth_obj, coeffs0, method='L-BFGS-B', 
                   options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
    
    if time.time() < deadline:
        h_opt = _idct_h(res.x)
        # Ensure sum is reasonable
        if abs(np.sum(h_opt)) > 1e-10:
            val_opt = _value(h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
    
    # Strategy 4: Multi-phase refinement
    # Try different alpha values for smooth max
    for alpha in [3.0, 5.0, 10.0, 20.0]:
        if time.time() > deadline:
            break
        
        def smooth_obj_alpha(coeffs, a=alpha):
            return _dct_smooth_obj(coeffs, alpha=a)
        
        res = minimize(smooth_obj_alpha, coeffs0, method='L-BFGS-B',
                       options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6})
        
        if time.time() < deadline:
            h_opt = _idct_h(res.x)
            if abs(np.sum(h_opt)) > 1e-10:
                val_opt = _value(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
                    coeffs0 = _dct_h(h_opt)
    
    # Final direct refinement using gradient on the actual objective
    # Use the peak location to guide updates
    if time.time() < deadline:
        h_refine = best_h.copy()
        eta = 0.001
        for _ in range(1000):
            if time.time() > deadline:
                break
            
            s = np.sum(h_refine)
            if abs(s) < 1e-10:
                break
            
            c = _autocorr_fft(h_refine)
            max_idx = np.argmax(c)
            
            # Gradient of c[max_idx] w.r.t h
            grad = np.zeros(N)
            i_min = max(0, max_idx - N + 1)
            i_max = min(N - 1, max_idx)
            indices = np.arange(i_min, i_max + 1)
            grad[indices] = h_refine[max_idx - indices]
            
            # Update with projection to maintain sum
            h_new = h_refine - eta * grad
            h_new = h_new + (np.sum(h_refine) - np.sum(h_new)) / N
            
            val_new = _value(h_new)
            if val_new < _value(h_refine):
                h_refine = h_new
                if val_new < best_val:
                    best_val = val_new
                    best_h = h_refine.copy()
                eta = max(eta * 0.95, 1e-6)
            else:
                eta = max(eta * 0.5, 1e-6)
                if eta < 1e-7:
                    break
    
    return best_h
