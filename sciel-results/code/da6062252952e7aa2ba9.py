import numpy as np
from scipy.optimize import minimize
from scipy.fft import dct, idct
import time

N = 400

def _evaluate(h):
    """Compute the exact objective value."""
    s = np.sum(h)
    if abs(s) < 1e-10 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv)) / (s * s)

def _soft_max_and_weights(conv, beta):
    """Compute soft-max and its weights for the autoconvolution."""
    m = np.max(conv)
    if m <= 0:
        return m, np.zeros_like(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    soft_max = m + np.log(np.sum(exp_shifted)) / beta
    return soft_max, weights

def _grad_autoconv(w, h):
    """
    Gradient of sum_j w_j * conv_j w.r.t. h.
    conv = full autoconvolution of h.
    d(conv_j)/d(h_l) = 2 * h[j-l] if 0 <= j-l < N, else 0.
    So grad_h[l] = 2 * sum_j w_j * h[j-l] (for valid j-l).
    This is 2 * cross_correlation(w, h)[l] where cross_corr[l] = sum_j w[j] * h[j-l].
    Using FFT: cross_corr = ifft(fft(w) * conj(fft(h)))
    """
    # Pad to length 2N for full convolution/correlation
    L = 2 * N
    w_pad = np.zeros(L)
    w_pad[:len(w)] = w
    h_pad = np.zeros(L)
    h_pad[:N] = h
    
    W = np.fft.fft(w_pad, n=L)
    H = np.fft.fft(h_pad, n=L)
    
    # Cross-correlation: sum_j w[j] * h[j-l]
    # This is ifft(W * conj(H))
    cross = np.fft.ifft(W * np.conj(H)).real
    grad_h = 2.0 * cross[:N]
    return grad_h

def _objective_soft(h, beta=50.0):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    soft_max, _ = _soft_max_and_weights(conv, beta)
    return 2.0 * N * soft_max / (s * s)

def _objective_soft_grad(h, beta=50.0):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros_like(h)
    
    conv = np.convolve(h, h, mode='full')
    soft_max, weights = _soft_max_and_weights(conv, beta)
    
    if np.max(weights) < 1e-15:
        return np.zeros_like(h)
    
    grad_sm = _grad_autoconv(weights, h)
    
    # f = 2*N*soft_max/s^2
    # df/dh_i = 2*N * (grad_sm_i * s^2 - soft_max * 2*s) / s^4
    #         = 2*N * (grad_sm_i * s - 2*soft_max) / s^3
    return 2.0 * N * (grad_sm * s - 2.0 * soft_max) / (s * s * s)

def _spectral_penalty(h, lam=0.1):
    """Penalty for spectral concentration (L4 norm of DCT coefficients)."""
    dct_coeffs = dct(h, type=2, norm='ortho')
    return lam * np.sum(np.abs(dct_coeffs)**4) / N

def _spectral_penalty_grad(h, lam=0.1):
    """Gradient of spectral penalty."""
    # Use numerical differentiation for the DCT part as it's complex to derive analytically 
    # but N=400 is small enough for a few calls, or we can derive it.
    # d/dh_i sum_j |c_j|^4 = sum_j 4 |c_j|^2 sign(c_j) (dc_j/dh_i)
    # For DCT-II ortho: c_j = sqrt(2/N) * c_j_factor * sum_k h_k cos(pi*j*(k+0.5)/N)
    # Let's use the analytic form.
    # c_j = a_j * sum_k h_k cos(theta_jk)
    # dc_j/dh_i = a_j * cos(theta_ji)
    # grad_i = sum_j 4 |c_j|^2 sign(c_j) a_j cos(theta_ji)
    
    # Actually, simpler to just use the property that DCT is orthogonal.
    # Let C be the DCT matrix. c = C h.
    # Penalty P = lam/N * ||c||_4^4 = lam/N * sum c_j^4
    # dP/dh_i = lam/N * sum_j 4 c_j^3 * (C_{ji})
    # Since C is orthogonal (C^T C = I), C_{ji} = C_{ij} (if symmetric) or just the matrix element.
    # DCT-II is not symmetric but orthogonal.
    # grad = (lam/N) * 4 * C^T * (c**3)
    
    dct_coeffs = dct(h, type=2, norm='ortho')
    # We need the inverse DCT of (c**3) multiplied by 4*lam/N
    # idct(c**3) gives h_reconstructed_from_cubed
    # The gradient is 4*lam/N * idct(dct_coeffs**3)
    # Note: idct is the inverse of dct.
    # d/dh (sum c_j^4) = 4 * C^T (c^3) = 4 * idct(c^3)
    
    grad = 4.0 * lam / N * idct(dct_coeffs**3, type=2, norm='ortho')
    return grad

def _combined_obj(h, beta=50.0, lam=0.05):
    return _objective_soft(h, beta) + _spectral_penalty(h, lam)

def _combined_grad(h, beta=50.0, lam=0.05):
    return _objective_soft_grad(h, beta) + _spectral_penalty_grad(h, lam)

def _refine(h, deadline):
    """Refine a candidate h using a sequence of optimization phases."""
    h_curr = h.copy()
    
    # Phase 1: Moderate beta, spectral flattening
    if time.time() < deadline - 10.0:
        try:
            res = minimize(_combined_obj, h_curr, method='L-BFGS-B',
                           args=(30.0, 0.1), jac=_combined_grad,
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-7})
            h_curr = res.x
        except Exception:
            pass
    
    # Phase 2: Higher beta, less spectral penalty (focus on peak)
    if time.time() < deadline - 5.0:
        try:
            res = minimize(_combined_obj, h_curr, method='L-BFGS-B',
                           args=(80.0, 0.01), jac=_combined_grad,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9})
            h_curr = res.x
        except Exception:
            pass
    
    # Phase 3: Pure peak polishing with very high beta
    if time.time() < deadline - 2.0:
        try:
            res = minimize(_objective_soft, h_curr, method='L-BFGS-B',
                           args=(150.0,), jac=_objective_soft_grad,
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            h_curr = res.x
        except Exception:
            pass
            
    return h_curr

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    rng = np.random.default_rng(12345)
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    # Generate diverse initializations
    initializations = []
    i = np.arange(N)
    
    # 1. Seed
    initializations.append(np.ones(N))
    
    # 2. Low-frequency sinusoids with DC offset (ensures sum != 0)
    for k in [1, 2, 3, 4]:
        for a in [0.5, 1.0, 2.0]:
            h0 = a + np.sin(2*np.pi*k*i/N)
            initializations.append(h0)
            
    # 3. Comb patterns with small noise to break symmetry
    for period in [2, 3, 4, 5, 8, 10]:
        h0 = np.where((i % period) < period // 2, 1.0, -1.0)
        h0 += 0.1 * rng.standard_normal(N)
        if abs(np.sum(h0)) > 1.0:
            initializations.append(h0)
            
    # 4. High-frequency noise (low-pass filtered)
    for _ in range(5):
        h0 = rng.standard_normal(N)
        # Simple moving average as low-pass
        kernel = np.ones(5) / 5.0
        h0 = np.convolve(h0, kernel, mode='same')
        # Add DC to ensure sum is not zero
        h0 += 1.0
        initializations.append(h0)
        
    # 5. Random smooth functions
    for _ in range(5):
        h0 = np.sin(2*np.pi*rng.uniform(0.1, 2.0)*i/N + rng.uniform(0, 2*np.pi))
        h0 += 1.0
        initializations.append(h0)

    # Optimize each initialization
    for h0 in initializations:
        if time.time() > deadline - 15.0:
            break
            
        h_refined = _refine(h0, deadline)
        val = _evaluate(h_refined)
        if val < best_val:
            best_val = val
            best_h = h_refined.copy()
            
    # Final global check and return
    if time.time() < deadline - 1.0:
        # One last polish on the best found
        h_final = _refine(best_h, deadline)
        val_final = _evaluate(h_final)
        if val_final < best_val:
            best_h = h_final
            
    return best_h
