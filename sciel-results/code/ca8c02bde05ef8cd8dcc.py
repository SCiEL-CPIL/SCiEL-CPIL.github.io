"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.fft import dct, idct

def _fft_autoconv(h):
    """Compute full autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n
    fh = np.fft.rfft(h, size)
    conv = np.fft.irfft(fh * fh, size)[:n + n - 1]
    return conv

def _compute_value(h):
    s = np.sum(h)
    if s == 0:
        return np.inf
    conv = _fft_autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _grad_autoconv(w, h):
    """
    Gradient of sum_j w_j * conv_j w.r.t. h.
    conv = full autoconvolution of h.
    d(conv_j)/d(h_l) = 2 * h[j-l] if 0 <= j-l < N, else 0.
    So grad_h[l] = 2 * sum_j w_j * h[j-l] (for valid j-l).
    This is 2 * cross_correlation(w, h)[l] where cross_corr[l] = sum_j w[j] * h[j-l].
    Using FFT: cross_corr = ifft(fft(w) * conj(fft(h)))
    """
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

def _objective_soft(h, beta=50.0):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _fft_autoconv(h)
    soft_max, _ = _soft_max_and_weights(conv, beta)
    return 2.0 * N * soft_max / (s * s)

def _objective_soft_grad(h, beta=50.0):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros_like(h)
    
    conv = _fft_autoconv(h)
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
    dct_coeffs = dct(h, type=2, norm='ortho')
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
    if time.time() < deadline - 15.0:
        try:
            res = minimize(_combined_obj, h_curr, method='L-BFGS-B',
                           args=(30.0, 0.1), jac=_combined_grad,
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-7})
            h_curr = res.x
        except Exception:
            pass
    
    # Phase 2: Higher beta, less spectral penalty (focus on peak)
    if time.time() < deadline - 10.0:
        try:
            res = minimize(_combined_obj, h_curr, method='L-BFGS-B',
                           args=(80.0, 0.01), jac=_combined_grad,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9})
            h_curr = res.x
        except Exception:
            pass
    
    # Phase 3: Pure peak polishing with very high beta
    if time.time() < deadline - 5.0:
        try:
            res = minimize(_objective_soft, h_curr, method='L-BFGS-B',
                           args=(150.0,), jac=_objective_soft_grad,
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            h_curr = res.x
        except Exception:
            pass
            
    return h_curr

def _simulated_annealing_binary(h_init, deadline, rng):
    """
    Binary Simulated Annealing to find low-peak sequences.
    Represents h as binary {-1, 1} initially, then allows small perturbations.
    """
    # Start with a binary-like sequence derived from h_init
    h = np.sign(h_init)
    # Ensure sum is not zero
    if np.sum(h) == 0:
        h[0] = 1.0
        
    best_h = h.copy()
    best_val = _compute_value(h)
    
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.995
    
    steps = 0
    max_steps = 5000
    
    while time.time() < deadline - 5.0 and steps < max_steps:
        T = T_start * (T_end / T_start) ** (steps / max_steps)
        
        # Pick a random index to flip
        i = rng.integers(0, N)
        old_val = h[i]
        h[i] = -h[i]
        
        new_val = _compute_value(h)
        
        delta = new_val - best_val
        if delta < 0 or rng.random() < np.exp(-delta / (T + 1e-10)):
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            h[i] = old_val
            
        steps += 1
        
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    i = np.arange(N)
    
    # 1. Seed
    initializations.append(np.ones(N))
    
    # 2. Low-frequency sinusoids with DC offset
    for k in [1, 2, 3, 4]:
        for a in [0.5, 1.0, 2.0]:
            h0 = a + np.sin(2*np.pi*k*i/N)
            initializations.append(h0)
            
    # 3. Comb patterns
    for period in [2, 3, 4, 5, 8, 10]:
        h0 = np.where((i % period) < period // 2, 1.0, -1.0)
        h0 += 0.1 * rng.standard_normal(N)
        if abs(np.sum(h0)) > 1.0:
            initializations.append(h0)
            
    # 4. Random smooth functions
    for _ in range(5):
        h0 = np.sin(2*np.pi*rng.uniform(0.1, 2.0)*i/N + rng.uniform(0, 2*np.pi))
        h0 += 1.0
        initializations.append(h0)

    # Optimize each initialization with gradient descent
    for h0 in initializations:
        if time.time() > deadline - 30.0:
            break
            
        h_refined = _refine(h0, deadline)
        val = _compute_value(h_refined)
        if val < best_val:
            best_val = val
            best_h = h_refined.copy()
            
    # Try binary Simulated Annealing on the best continuous solution
    if time.time() < deadline - 10.0:
        h_sa, val_sa = _simulated_annealing_binary(best_h, deadline, rng)
        if val_sa < best_val:
            best_val = val_sa
            best_h = h_sa.copy()
            
            # Refine the binary result with continuous optimization
            if time.time() < deadline - 5.0:
                h_refined = _refine(best_h, deadline)
                val_refined = _compute_value(h_refined)
                if val_refined < best_val:
                    best_val = val_refined
                    best_h = h_refined.copy()

    # Final global check and return
    if time.time() < deadline - 1.0:
        h_final = _refine(best_h, deadline)
        val_final = _compute_value(h_final)
        if val_final < best_val:
            best_h = h_final
            
    s_final = np.sum(best_h)
    if abs(s_final) < 1e-10:
        return np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
