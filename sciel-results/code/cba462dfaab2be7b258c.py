"""Rewritten program: signed step heights minimizing max|autoconv|/(sum h)^2."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Precompute FFT size
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_fft(h):
    """Compute linear autoconvolution using FFT."""
    H = np.fft.rfft(h, _FFT_SIZE)
    ac = np.fft.irfft(H * H, _FFT_SIZE)
    return ac[:2 * N - 1]

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution."""
    ac = _autoconv_fft(h)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _soft_max_abs(ac, beta=100.0):
    """
    Smooth approximation of max(|ac|) via log-sum-exp.
    """
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -100, 100)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=100.0):
    """
    Smooth objective for optimization: soft_max_abs(autoconv(h)) / (sum h)^2.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _spectral_papr_objective(h, beta=100.0):
    """
    Spectral PAPR objective using Parseval identity.
    The autoconvolution is the inverse FFT of |H|^2.
    We minimize max|IFFT(|H|^2)| / (sum h)^2.
    Using Parseval: the peak of the time-domain autoconv relates to spectral properties.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    H = np.fft.rfft(h, _FFT_SIZE)
    power = np.abs(H) ** 2
    # The autoconvolution is IFFT(H^2), not IFFT(|H|^2)
    # But we can use the spectral representation for PAPR-like reduction
    ac = np.fft.irfft(H * H, _FFT_SIZE)
    ac = ac[:2 * N - 1]
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _spectral_papr_reduce(h, n_iter=50, gamma=0.5, deadline=None):
    """
    Iterative spectral PAPR reduction (Gilbert-Nettleton style).
    For the autoconvolution problem, we work in the frequency domain.
    The key idea: minimize peak of IFFT(H^2) by modifying H.
    
    We use the fact that if we scale H by exp(-gamma * Re(IFFT(|H|^2)) * H),
    we reduce peaks in the time-domain autoconvolution.
    """
    h = h.copy().astype(float)
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        H = np.fft.rfft(h, _FFT_SIZE)
        # Compute the autoconvolution (time domain peak)
        ac = np.fft.irfft(H * H, _FFT_SIZE)
        ac = ac[:2 * N - 1]
        
        # Compute the "error" signal: which time-domain samples exceed the mean
        ac_abs = np.abs(ac)
        ac_mean = np.mean(ac_abs)
        
        # Identify peaks
        peaks = ac_abs > ac_mean
        
        # In frequency domain, modify H to reduce peaks
        # The gradient of max|ac| w.r.t. h involves the conjugate of the peak location
        # For PAPR reduction: H_new = H * exp(-gamma * IFFT(peak_mask * conj(H)))
        
        # Simpler approach: use the time-domain peak information to adjust
        # h_new = h - gamma * IFFT(peak_indicator * conj(H))
        # This is the standard Gilber-Nettleton iteration
        
        # Create peak indicator in time domain
        peak_indicator = np.zeros(_FFT_SIZE, dtype=complex)
        # Map autoconv indices back to full FFT length
        # The autoconv is centered; for linear conv, index 0..2N-2
        # In the FFT of length _FFT_SIZE, the autoconv starts at index 0
        # but wraps around. For N=400, 2N-1=799, _FFT_SIZE=1024
        # The autoconv[0..798] corresponds to indices 0..798 in the FFT
        # But negative lags wrap to the end
        # For simplicity, use the direct time-domain approach
        
        # Alternative: use the gradient of the soft-max objective
        # d/dh [soft_max_abs(ac)] = sum_k (|ac_k|/sum|ac_k|) * sign(ac_k) * (da_k/dh)
        # where a_k = (h*h)[k]
        
        # Simpler: just do a projected gradient step
        if np.max(ac_abs) < 1e-15:
            break
        
        # Compute gradient of soft_max_abs w.r.t. h
        abs_ac = np.abs(ac)
        weights = abs_ac / (np.sum(abs_ac) + 1e-15)
        
        # Gradient of ac[k] w.r.t. h[i]:
        # ac[k] = sum_j h[j] * h[k-j]
        # d(ac[k])/d(h[i]) = h[k-i] + h[i-k] (for linear conv)
        # This is expensive to compute directly, so use FFT-based gradient
        
        # The gradient of sum_k w_k * |ac[k]| w.r.t. h is:
        # 2 * IFFT(FFT(w * ac) * conj(H)) approximately
        # More precisely: grad = 2 * IFFT(FFT(w * sign(ac) * |ac|) * conj(H))
        # But this is an approximation. Let's use the standard approach.
        
        # For the autoconvolution, the gradient of (h*h)[k] w.r.t. h[i] is:
        # h[k-i] + h[i-k] (with appropriate bounds)
        # In frequency domain: if A = H^2, then grad_h[sum w_k A_k] = 2 * IFFT(FFT(w) * conj(H))
        
        w_fft = np.fft.rfft(weights[:2*N-1], _FFT_SIZE)
        grad = 2.0 * np.fft.irfft(w_fft * np.conj(H), _FFT_SIZE)[:N]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        grad = grad / grad_norm
        
        # Step
        h = h - gamma * grad
        
        # Preserve sum (project onto sum-preserving subspace)
        s = np.sum(h)
        if abs(s) < 1e-9:
            break
        h = h + (1.0 - s / s) * 0  # no-op, sum is preserved by construction if grad has zero sum
        # Actually, let's explicitly project
        h = h - np.mean(h) + np.mean(h)  # no change
        # Better: subtract the mean to keep sum constant
        # But we want to preserve the sum, so:
        h = h + (np.sum(h_init) - np.sum(h)) / N if False else h
    
    return h

def _spectral_init(n):
    """
    Generate initial sequence using spectral flatness principles.
    A flat spectrum corresponds to a delta-like time-domain signal.
    For minimizing autoconv peaks, we want a signal whose |H|^2 is as flat as possible,
    which means h should be close to white noise but with controlled phase.
    """
    rng = np.random.default_rng(7)
    # Start with random phases (flat magnitude spectrum)
    freqs = np.fft.rfftfreq(_FFT_SIZE) * 2 * np.pi
    # Only use frequencies that correspond to our signal length
    valid_freqs = np.abs(freqs) <= np.pi  # Nyquist
    
    # Generate random phases
    phases = rng.uniform(0, 2 * np.pi, size=_FFT_SIZE // 2 + 1)
    
    # Start with flat magnitude
    H = np.exp(1j * phases)
    
    # Convert to time domain and truncate
    h = np.fft.irfft(H, _FFT_SIZE)[:n]
    
    # Normalize
    s = np.sum(h)
    if abs(s) < 1e-9:
        return np.ones(n)
    return h / s

def _gen_barker_like(n):
    """
    Generate a Barker-like sequence: +1/-1 with good autocorrelation properties.
    """
    rng = np.random.default_rng(123)
    h = rng.choice([-1.0, 1.0], size=n)
    kernel = np.array([0.2, 0.3, 0.3, 0.2])
    h = np.convolve(h, kernel, mode='same')
    return h

def _gen_chirp(n):
    """Generate a chirp sequence."""
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_gaussian(n):
    """Generate a Gaussian-shaped sequence."""
    x = np.arange(n)
    mu = n / 2
    sigma = n / 4
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def _gen_random_smooth(n, seed=42):
    """Generate a random smoothed sequence."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    kernel = np.ones(8) / 8
    return np.convolve(noise, kernel, mode='same')

def _local_search(h, max_passes=12, deadline=None):
    """
    Local search with batched evaluations for efficiency.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(42)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        # Try flipping signs at random positions (batched)
        for _ in range(N):
            if deadline is not None and time.time() > deadline:
                break
            i = rng.integers(N)
            orig = h[i]
            h[i] = -orig
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = orig
        
        # Try small multiplicative perturbations
        for _ in range(N):
            if deadline is not None and time.time() > deadline:
                break
            i = rng.integers(N)
            orig = h[i]
            for factor in [1.05, 0.95, 1.1, 0.9]:
                if deadline is not None and time.time() > deadline:
                    break
                h[i] = orig * factor
                new_val = _score(h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                    break
            if not improved:
                h[i] = orig
        
        if not improved:
            break
    
    return h, best_val

def _epigraph_objective(h, t, beta=100.0):
    """
    Epigraph formulation: minimize t subject to soft_max_abs(ac) <= t * (sum h)^2.
    This is handled by the optimizer; we just provide the objective.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    # Penalty for constraint violation
    violation = np.maximum(0, sm - t * s * s)
    return t + 10.0 * violation

def _high_precision_polish(h, deadline=None):
    """
    High-precision polish using tighter tolerances and more iterations.
    """
    h = h.copy()
    current_score = _score(h)
    
    # Use very tight tolerances
    try:
        def obj_tight(x):
            return _objective_smooth(x, beta=500.0)
        
        res = minimize(obj_tight, h, method='L-BFGS-B',
                       options={'maxiter': 500, 'disp': False, 'gtol': 1e-10, 'ftol': 1e-12})
        h_opt = res.x
        if abs(np.sum(h_opt)) > 1e-6:
            val = _score(h_opt)
            if val < current_score:
                h = h_opt
                current_score = val
    except Exception:
        pass
    
    # Final local search with very small steps
    rng = np.random.default_rng(7)
    for _ in range(20):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for _ in range(N):
            if deadline is not None and time.time() > deadline:
                break
            i = rng.integers(N)
            orig = h[i]
            for factor in [1.01, 0.99, 1.02, 0.98]:
                h[i] = orig * factor
                new_val = _score(h)
                if new_val < current_score:
                    current_score = new_val
                    improved = True
                    break
            if not improved:
                h[i] = orig
        if not improved:
            break
    
    return h, current_score

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s kill
    
    best_score = np.inf
    best_h = np.ones(N)
    
    base_score = _score(best_h)
    best_score = base_score
    
    inits = []
    
    # 1. Constant
    inits.append(np.ones(N))
    
    # 2. Gaussian
    inits.append(_gen_gaussian(N))
    
    # 3. Chirp
    inits.append(_gen_chirp(N))
    
    # 4. Barker-like
    inits.append(_gen_barker_like(N))
    
    # 5. Random smoothed
    inits.append(_gen_random_smooth(N, seed=1))
    
    # 6. Another random with different seed
    inits.append(_gen_random_smooth(N, seed=99))
    
    # 7. Alternating signs with envelope
    x = np.arange(N)
    envelope = np.exp(-0.5 * ((x - N/2) / (N/5))**2)
    inits.append(envelope * np.cos(np.pi * x / 5))
    
    # 8. Spectral flatness init
    inits.append(_spectral_init(N))
    
    for h_init in inits:
        if time.time() > deadline - 20:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        
        # Phase 1: Gradient optimization with smooth surrogate
        for beta in [50, 100, 200]:
            if time.time() > deadline - 10:
                break
            
            try:
                def obj(x):
                    return _objective_smooth(x, beta=beta)
                
                res = minimize(obj, h_norm, method='L-BFGS-B',
                               options={'maxiter': 300, 'disp': False, 'gtol': 1e-6, 'ftol': 1e-8})
                h_opt = res.x
                
                if abs(np.sum(h_opt)) < 1e-6:
                    continue
                    
                val = _score(h_opt)
                if val < best_score:
                    best_score = val
                    best_h = h_opt.copy()
            except Exception:
                pass
        
        # Phase 2: Spectral PAPR reduction
        if time.time() > deadline - 8:
            continue
        try:
            h_spectral = _spectral_papr_reduce(h_norm, n_iter=30, gamma=0.1, deadline=deadline - 5)
            if abs(np.sum(h_spectral)) > 1e-6:
                val = _score(h_spectral)
                if val < best_score:
                    best_score = val
                    best_h = h_spectral.copy()
        except Exception:
            pass
    
    # Phase 3: High-precision polish on best found
    if time.time() < deadline - 10:
        h_polished, val_polished = _high_precision_polish(best_h, deadline=deadline - 3)
        if val_polished < best_score:
            best_score = val_polished
            best_h = h_polished.copy()
    
    # Final local search
    if time.time() < deadline - 3:
        h_refined, val_refined = _local_search(best_h, max_passes=10, deadline=deadline - 1)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
    
    # Safety: if somehow we lost track, fall back
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N)
    
    return best_h

# EVOLVE-BLOCK-END
