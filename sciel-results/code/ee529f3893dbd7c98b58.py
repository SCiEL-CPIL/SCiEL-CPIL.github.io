"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600

def _score_step(h):
    """Compute the score for a step function h (array of length N).
    Score = max(autoconv) / (sum(h))^2
    """
    if np.any(h < 0):
        return float('inf')
    s = h.sum()
    if s <= 0:
        return float('inf')
    # Compute autoconvolution via FFT
    # Linear convolution length is 2*N - 1
    size = 2 * N
    F = np.fft.rfft(h, size)
    auto = np.fft.irfft(F * F, size)[:2 * N - 1]
    peak = np.max(auto)
    return peak / (s * s)

def _make_gaussian(sigma_frac, amplitude):
    """Create a Gaussian step function."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    h = amplitude * np.exp(-0.5 * (x / sigma_frac) ** 2)
    h = np.maximum(h, 0)
    if h.sum() > 0:
        h = h / h.sum()
    else:
        h = np.ones(N) / N
    return h

def _make_triangular(width_frac, amplitude):
    """Create a triangular step function."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    h = np.maximum(1 - np.abs(x) / width_frac, 0) * amplitude
    h = np.maximum(h, 0)
    if h.sum() > 0:
        h = h / h.sum()
    else:
        h = np.ones(N) / N
    return h

def _make_box(width_frac):
    """Create a box step function with given width."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    h = np.where(np.abs(x) <= width_frac, 1.0, 0.0)
    if h.sum() > 0:
        h = h / h.sum()
    else:
        h = np.ones(N) / N
    return h

def _make_bump(p, width_frac):
    """Create a polynomial bump function of degree p."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    u = np.abs(x) / width_frac
    h = np.where(u < 1, (1 - u ** 2) ** p, 0)
    h = np.maximum(h, 0)
    if h.sum() > 0:
        h = h / h.sum()
    else:
        h = np.ones(N) / N
    return h

def _spectral_optimize(h_init, max_iters=200, lr=0.01, deadline=None):
    """
    Spectral amplitude shaping: optimize |H_k| (DFT magnitudes) to minimize
    peak-to-average ratio of autoconvolution, using log-sum-exp for smooth max.
    
    For zero phase, autoconv[k] = IDFT(|H|^2)[k] * (1/N) ... actually:
    If h has DFT H, then autoconv = IDFT(|H|^2) (circular). For linear, we pad.
    
    We work with circular autoconv on N points for efficiency, then verify with linear.
    Actually, let's work directly: score = max_k (h*h)[k] / (sum h)^2
    where (h*h)[k] = sum_j h[j] h[j-k] (circular) or linear.
    
    Let's use: if we define g = h / sum(h), then score = max(autoconv_g)
    and autoconv_g[k] = sum_j g[j] g[j-k]
    
    In frequency: G_k = DFT(g)_k, then autoconv_g = IDFT(|G|^2) / N (for circular)
    
    We optimize over |G_k| subject to g >= 0 and sum(g) = 1.
    """
    # Normalize initial h
    h = h_init.copy()
    h = np.maximum(h, 0)
    if h.sum() <= 0:
        h = np.ones(N) / N
    h = h / h.sum()
    
    # Get DFT
    G = np.fft.fft(h)
    # Work with log magnitudes for positivity
    log_mag = np.log(np.abs(G) + 1e-12)
    phase = np.angle(G)
    
    best_h = h.copy()
    best_score = _score_step(h)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Reconstruct h from current log_mag and phase
        mag = np.exp(log_mag)
        G_cur = mag * np.exp(1j * phase)
        h_cur = np.real(np.fft.ifft(G_cur))
        
        # Project to non-negative and normalize
        h_cur = np.maximum(h_cur, 0)
        s = h_cur.sum()
        if s <= 0:
            break
        h_cur = h_cur / s
        
        # Compute score and gradient using log-sum-exp for smooth max
        # autoconv = IDFT(|G_cur|^2) / N (circular)
        G_cur = np.fft.fft(h_cur)
        autoconv = np.real(np.fft.ifft(np.abs(G_cur) ** 2)) / N
        
        # Smooth max via log-sum-exp
        beta = 50.0  # temperature parameter
        log_max = np.logaddexp.reduce(np.log(autoconv + 1e-30) * beta) / beta
        
        # Gradient of log-sum-exp w.r.t. autoconv: softmax
        weights = np.exp((np.log(autoconv + 1e-30) * beta - log_max * beta))
        weights /= weights.sum()
        
        # d(autoconv)/d(G_k) and d(G_k)/d(log_mag_k)
        # autoconv = IDFT(|G|^2)/N
        # d(autoconv)/d(G_k) = conj(G_k) / N * exp(2*pi*i*k*n/N) for each n
        # In matrix form: autoconv_n = (1/N) sum_k |G_k|^2 exp(2*pi*i*k*n/N)
        # d(autoconv_n)/d(log_mag_k) = (1/N) * 2*|G_k|^2 * exp(2*pi*i*k*n/N) * exp(-i*phase_k) ... 
        # Actually |G_k|^2 = mag_k^2, d|G_k|^2/d(log_mag_k) = 2*mag_k^2
        # So d(autoconv_n)/d(log_mag_k) = (2*mag_k^2 / N) * exp(2*pi*i*k*n/N)
        
        # Gradient of log_max w.r.t. log_mag:
        # d(log_max)/d(autoconv_n) = weights[n]
        # d(autoconv_n)/d(log_mag_k) = (2/N) * mag_k^2 * exp(2*pi*i*k*n/N)
        # So d(log_max)/d(log_mag_k) = (2/N) * mag_k^2 * sum_n weights[n] * exp(2*pi*i*k*n/N)
        # = (2/N) * mag_k^2 * DFT(weights)_k (conjugate? let's be careful)
        
        # exp(2*pi*i*k*n/N) is the DFT kernel. sum_n weights[n] exp(2*pi*i*k*n/N) = DFT(weights)_k
        W = np.fft.fft(weights)
        grad = (2.0 / N) * mag ** 2 * np.real(W)
        
        # Update log_mag
        log_mag -= lr * grad
        
        # Clamp log_mag to prevent overflow
        log_mag = np.clip(log_mag, -20, 20)
        
        # Reconstruct and check
        mag_new = np.exp(log_mag)
        G_new = mag_new * np.exp(1j * phase)
        h_new = np.real(np.fft.ifft(G_new))
        h_new = np.maximum(h_new, 0)
        s_new = h_new.sum()
        if s_new <= 0:
            continue
        h_new = h_new / s_new
        
        s_new_val = _score_step(h_new)
        if s_new_val < best_score:
            best_score = s_new_val
            best_h = h_new.copy()
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 40s safety margin
    
    best_score = float('inf')
    best_h = np.ones(N) / N
    
    # Initialize with several parametric families
    candidates = []
    
    # Box functions
    for width_frac in [0.05, 0.1, 0.15, 0.2, 0.25]:
        candidates.append(_make_box(width_frac))
    
    # Gaussian functions
    for sigma_frac in [0.03, 0.05, 0.08, 0.12, 0.18, 0.25]:
        candidates.append(_make_gaussian(sigma_frac, 1.0))
    
    # Triangular functions
    for width_frac in [0.05, 0.1, 0.15, 0.2, 0.25]:
        candidates.append(_make_triangular(width_frac, 1.0))
    
    # Polynomial bumps
    for p in [2, 3, 4]:
        for width_frac in [0.1, 0.15, 0.2, 0.25]:
            candidates.append(_make_bump(p, width_frac))
    
    # Evaluate all candidates
    for h_cand in candidates:
        if time.time() > deadline:
            break
        s = _score_step(h_cand)
        if s < best_score:
            best_score = s
            best_h = h_cand.copy()
    
    # Spectral optimization from the best candidate
    if time.time() < deadline - 30:
        h_opt, s_opt = _spectral_optimize(best_h, max_iters=100, lr=0.005, deadline=deadline - 10)
        if s_opt < best_score:
            best_score = s_opt
            best_h = h_opt.copy()
    
    # Second spectral optimization with different initial conditions
    if time.time() < deadline - 20:
        # Try from a different initial point
        h_alt = _make_gaussian(0.1, 1.0)
        h_opt2, s_opt2 = _spectral_optimize(h_alt, max_iters=50, lr=0.003, deadline=deadline - 5)
        if s_opt2 < best_score:
            best_score = s_opt2
            best_h = h_opt2.copy()
    
    # Final local refinement with small perturbations
    if time.time() < deadline - 5 and best_score < float('inf'):
        for _ in range(20):
            if time.time() > deadline:
                break
            noise = np.random.randn(N)
            kernel = np.exp(-np.arange(-20, 21) ** 2 / 10)
            noise = np.convolve(noise, kernel, mode='same')
            noise = (noise - noise.mean()) * 0.005
            h_pert = best_h + noise
            h_pert = np.maximum(h_pert, 0)
            if h_pert.sum() <= 0:
                continue
            h_pert = h_pert / h_pert.sum()
            s = _score_step(h_pert)
            if s < best_score:
                best_score = s
                best_h = h_pert.copy()
    
    # Final check
    best_h = np.maximum(best_h, 0)
    if best_h.sum() <= 0:
        best_h = np.ones(N) / N
    
    return list(best_h)

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
