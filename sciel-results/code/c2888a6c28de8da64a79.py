"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _spectral_score(H_mags):
    """
    Compute spectral score from magnitude spectrum.
    For a real sequence, H[k] and H[N-k] are conjugates.
    The autoconvolution peak is bounded by max_k |H[k]|^2 / (sum h)^2 * N.
    We minimize the peak of |H[k]|^2 subject to H[0] = sum(h).
    """
    # H_mags should be the full DFT magnitude array of length N
    H0 = H_mags[0]
    if H0 < 1e-10:
        return 1e10
    # The peak autoconv at lag 0 is sum(h^2) = (1/N) * sum_k |H[k]|^2
    # But max absolute autoconv is bounded by max over lags
    # For flat spectrum, peak is approximately (1/N) * sum_k |H[k]|^2 / N
    # Actually, by Parseval: sum_lag |ac[lag]|^2 = (1/N) * sum_k |H[k]|^4
    # The max |ac| is at least sqrt(mean of |ac|^2) and at most sqrt(sum |ac|^2)
    # For our purpose, we minimize the peak of |H[k]|^2 for k != 0
    # Score proxy: max_k |H[k]|^2 / H0^2
    peak = np.max(H_mags[1:] ** 2) if len(H_mags) > 1 else 0.0
    return peak / (H0 * H0)

def _optimize_spectral_phases():
    """
    Optimize phases of DFT components to minimize peak |H[k]|^2 / H0^2.
    Use gradient descent on the phase vector.
    """
    n = N
    # Start with random phases
    rng = np.random.default_rng(42)
    best_h = None
    best_score = np.inf
    
    for trial in range(5):
        # Initialize phases
        if trial == 0:
            # Chirp-like initialization
            i = np.arange(n)
            phases = 2.0 * np.pi * i**2 / n
        else:
            phases = rng.uniform(0, 2*np.pi, n)
        
        # Build initial H
        # H[k] = A_k * exp(j * phi_k)
        # For real h, H[k] = conj(H[n-k])
        # We parameterize using n//2 + 1 phases (for real sequences)
        # H[0] is real (DC), H[n/2] is real if n even
        
        # Simpler: work with complex H and enforce conjugate symmetry
        H = np.zeros(n, dtype=complex)
        H[0] = n  # DC component, will be adjusted
        
        # Set random phases for k = 1 to n//2
        for k in range(1, n//2 + 1):
            if k == n - k:  # Nyquist
                H[k] = np.exp(1j * phases[k])
            else:
                H[k] = np.exp(1j * phases[k])
                H[n-k] = np.conj(H[k])
        
        # Compute h
        h = np.fft.irfft(H, n)
        
        # Optimize phases using gradient descent
        t_start = time.time()
        deadline = t_start + 20.0
        
        lr = 0.1
        for step in range(500):
            if time.time() > deadline:
                break
            
            h = np.fft.irfft(H, n)
            H0 = np.abs(H[0])
            if H0 < 1e-10:
                break
            
            # Compute score
            peak = np.max(np.abs(H[1:]) ** 2)
            score = peak / (H0 * H0)
            
            if score < best_score:
                best_score = score
                best_h = h.copy()
            
            # Compute gradient of score w.r.t. phases
            # d(score)/d(phi_k) for k = 1 to n//2
            grad = np.zeros(n//2 + 1)
            eps = 0.01
            
            for k in range(1, n//2 + 1):
                if time.time() > deadline - 1:
                    break
                
                # Perturb phase k
                H_old_k = H[k]
                H_old_nk = H[n-k] if k != n-k else None
                
                H[k] = H_old_k * np.exp(1j * eps)
                if k != n-k:
                    H[n-k] = H_old_nk * np.exp(-1j * eps)
                
                h_plus = np.fft.irfft(H, n)
                H0_plus = np.abs(H[0])
                peak_plus = np.max(np.abs(H[1:]) ** 2)
                score_plus = peak_plus / (H0_plus * H0_plus) if H0_plus > 1e-10 else 1e10
                
                H[k] = H_old_k * np.exp(-1j * eps)
                if k != n-k:
                    H[n-k] = H_old_nk * np.exp(1j * eps)
                
                h_minus = np.fft.irfft(H, n)
                H0_minus = np.abs(H[0])
                peak_minus = np.max(np.abs(H[1:]) ** 2)
                score_minus = peak_minus / (H0_minus * H0_minus) if H0_minus > 1e-10 else 1e10
                
                H[k] = H_old_k
                if k != n-k:
                    H[n-k] = H_old_nk
                
                grad[k] = (score_plus - score_minus) / (2 * eps)
            
            # Update phases
            for k in range(1, n//2 + 1):
                phases[k] -= lr * grad[k]
                # Keep in [0, 2*pi)
                phases[k] = phases[k] % (2 * np.pi)
            
            # Rebuild H
            for k in range(1, n//2 + 1):
                if k == n - k:
                    H[k] = np.exp(1j * phases[k])
                else:
                    H[k] = np.exp(1j * phases[k])
                    H[n-k] = np.conj(H[k])
            
            # Adaptive learning rate
            h_new = np.fft.irfft(H, n)
            H0_new = np.abs(H[0])
            peak_new = np.max(np.abs(H[1:]) ** 2)
            score_new = peak_new / (H0_new * H0_new) if H0_new > 1e-10 else 1e10
            
            if score_new < score:
                lr = min(lr * 1.05, 0.5)
            else:
                lr = max(lr * 0.9, 0.01)
    
    if best_h is None:
        best_h = np.ones(N) / N
        best_score = _score(best_h)
    
    return best_h, best_score

def _construct_flat_spectrum_sequence():
    """
    Construct a real sequence with nearly flat non-DC spectrum.
    Uses quadratic phase (chirp) which has flat magnitude spectrum.
    """
    i = np.arange(N)
    # Quadratic phase chirp: h[i] = cos(2*pi*i^2/N)
    # This has approximately flat magnitude spectrum for prime N
    h = np.cos(2.0 * np.pi * i**2 / N)
    return h

def _construct_zadoff_chu_like():
    """
    Construct a Zadoff-Chu-like sequence for real-valued output.
    ZC sequences have constant amplitude and zero autocorrelation sidelobes.
    """
    i = np.arange(N)
    # For real sequences, use cos of quadratic phase
    # ZC: exp(-j*pi*k*i*(i+1)/N) for complex, take real part
    h = np.cos(np.pi * i * (i + 1) / N)
    return h

def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin before 120s limit
    
    best_score = np.inf
    best_h = np.ones(N) / N  # Baseline: constant sequence
    
    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()
    
    # Strategy 1: Flat spectrum chirp sequences
    candidates = []
    
    # Quadratic phase chirp
    i = np.arange(N)
    candidates.append(np.cos(2.0 * np.pi * i**2 / N))
    candidates.append(np.cos(np.pi * i**2 / N))
    candidates.append(np.cos(2.0 * np.pi * i**2 / (N+1)))
    
    # Zadoff-Chu like
    candidates.append(np.cos(np.pi * i * (i + 1) / N))
    candidates.append(np.cos(np.pi * i * (i + 1) / (N+1)))
    
    # Barker-like patterns
    for period in [4, 6, 8, 12]:
        pattern = np.array([1, -1, 1, -1])[:period]
        h = np.tile(pattern, N // period + 1)[:N]
        candidates.append(h)
    
    # Random smoothed
    rng = np.random.default_rng(42)
    for seed in range(3):
        rng_local = np.random.default_rng(seed)
        noise = rng_local.standard_normal(N)
        kernel = np.ones(5) / 5
        smooth = np.convolve(noise, kernel, mode='same')
        candidates.append(np.sign(smooth))
    
    # Evaluate all candidates
    for h_cand in candidates:
        if time.time() > deadline - 30:
            break
        s = np.sum(h_cand)
        if abs(s) < 1e-6:
            continue
        # Normalize to have sum = 1 for fair comparison, but score is scale-invariant
        val = _score(h_cand)
        if val < best_score:
            best_score = val
            best_h = h_cand.copy()
    
    # Strategy 2: Spectral phase optimization
    if time.time() < deadline - 25:
        h_spec, score_spec = _optimize_spectral_phases()
        if score_spec < best_score:
            best_score = score_spec
            best_h = h_spec.copy()
    
    # Local search refinement on best candidate
    if time.time() < deadline - 10:
        h_refined = best_h.copy()
        best_local = best_score
        for _ in range(3):
            improved = False
            for idx in range(N):
                if time.time() > deadline - 5:
                    break
                h_refined[idx] = -h_refined[idx]
                new_val = _score(h_refined)
                if new_val < best_local:
                    best_local = new_val
                    improved = True
                else:
                    h_refined[idx] = -h_refined[idx]
            if not improved:
                break
        
        if best_local < best_score:
            best_score = best_local
            best_h = h_refined.copy()
    
    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N
    
    return best_h

# EVOLVE-BLOCK-END
