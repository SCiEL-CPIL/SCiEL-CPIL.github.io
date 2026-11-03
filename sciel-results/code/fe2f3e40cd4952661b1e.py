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

def _spectral_score_from_H(H_full):
    """
    Given full complex DFT H, compute the score using spectral properties.
    H_full has length N. We use the relationship that linear autoconvolution
    peak is bounded by the max of |H[k]|^2 / N (approximately for large N).
    """
    # The DC component is H[0] = sum(h)
    dc = H_full[0]
    if abs(dc) < 1e-9:
        return 1e10
    
    # For linear autoconvolution, the peak is related to the time-domain 
    # reconstruction. We compute it directly for accuracy.
    h = np.fft.irfft(H_full, N)
    # Ensure real
    h = h.real
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    
    # Compute linear autoconvolution
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H_fft = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H_fft * H_fft, fsize)
    max_abs = np.max(np.abs(ac))
    return max_abs / (s * s)

def _optimize_phases_spectral():
    """
    Optimize the phases of the DFT components to minimize the peak of 
    the autoconvolution. We work in the frequency domain, keeping the 
    magnitudes fixed (or optimizing them) and adjusting phases.
    
    Strategy: Start with a chirp-like phase pattern, then use iterative 
    phase refinement to flatten the time-domain autoconvolution peak.
    """
    i = np.arange(N)
    
    # Start with quadratic phase (chirp) - has flat magnitude spectrum
    # H[k] = exp(-j * pi * k^2 / N) gives chirp in time domain
    phases = np.pi * i**2 / N
    H = np.exp(-1j * phases)
    
    # Ensure DC is real and positive
    H[0] = np.abs(H[0])
    
    best_h = np.fft.irfft(H, N).real
    best_score = _score(best_h)
    
    t_start = time.time()
    deadline = t_start + 90.0
    
    # Iterative phase optimization using subgradient-like updates
    lr = 0.1
    for step in range(100):
        if time.time() > deadline - 10:
            break
        
        h = np.fft.irfft(H, N).real
        current_score = _score(h)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
        
        # Compute gradient of score w.r.t. phases
        # Use finite differences on a subset of phases
        grad_phases = np.zeros(N)
        eps = 0.05
        # Sample phases to estimate gradient
        sample_indices = np.random.choice(N, size=min(50, N), replace=False)
        for idx in sample_indices:
            if time.time() > deadline - 5:
                break
            H_old = H[idx].copy()
            # Perturb phase
            angle = np.angle(H_old)
            mag = np.abs(H_old)
            H[idx] = mag * np.exp(1j * (angle + eps))
            h_plus = np.fft.irfft(H, N).real
            score_plus = _score(h_plus)
            
            H[idx] = mag * np.exp(1j * (angle - eps))
            h_minus = np.fft.irfft(H, N).real
            score_minus = _score(h_minus)
            
            H[idx] = H_old
            grad_phases[idx] = (score_plus - score_minus) / (2 * eps)
        
        # Update phases in direction of negative gradient
        angles = np.angle(H)
        mags = np.abs(H)
        # Only update non-DC components
        update_mask = np.ones(N, dtype=bool)
        update_mask[0] = False  # Keep DC fixed
        
        angles[update_mask] -= lr * grad_phases[update_mask]
        H = mags * np.exp(1j * angles)
        H[0] = mags[0]  # Keep DC real and positive
        
        # Adaptive learning rate
        h_new = np.fft.irfft(H, N).real
        new_score = _score(h_new)
        if new_score < best_score:
            lr = min(lr * 1.1, 0.5)
        else:
            lr = max(lr * 0.7, 0.01)
            H = H / np.abs(H) * mags  # Restore previous if worse
    
    return best_h, best_score

def _optimize_magnitudes_and_phases():
    """
    Jointly optimize both magnitudes and phases of the DFT to minimize 
    the autoconvolution peak. This is a more powerful approach.
    """
    i = np.arange(N)
    
    # Start with flat magnitude spectrum (all ones) and chirp phase
    mags = np.ones(N)
    phases = np.pi * i**2 / N
    H = mags * np.exp(-1j * phases)
    H[0] = np.abs(H[0])  # DC real and positive
    
    best_h = np.fft.irfft(H, N).real
    best_score = _score(best_h)
    
    t_start = time.time()
    deadline = t_start + 80.0
    
    # Use coordinate descent on phases and magnitudes
    for outer_iter in range(5):
        if time.time() > deadline - 10:
            break
        
        # Phase optimization
        for inner_iter in range(30):
            if time.time() > deadline - 5:
                break
            
            h = np.fft.irfft(H, N).real
            current_score = _score(h)
            
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
            
            # Estimate gradient for a few random phases
            n_samples = min(20, N)
            sample_indices = np.random.choice(N, size=n_samples, replace=False)
            for idx in sample_indices:
                if idx == 0:
                    continue
                if time.time() > deadline - 3:
                    break
                
                H_old = H[idx].copy()
                angle = np.angle(H_old)
                mag = np.abs(H_old)
                eps = 0.1
                
                H[idx] = mag * np.exp(1j * (angle + eps))
                h_plus = np.fft.irfft(H, N).real
                score_plus = _score(h_plus)
                
                H[idx] = mag * np.exp(1j * (angle - eps))
                h_minus = np.fft.irfft(H, N).real
                score_minus = _score(h_minus)
                
                H[idx] = H_old
                grad = (score_plus - score_minus) / (2 * eps)
                
                # Update phase
                new_angle = angle - 0.1 * grad
                H[idx] = mag * np.exp(1j * new_angle)
            
            # Magnitude optimization (keep DC fixed)
            if time.time() < deadline - 5:
                for idx in range(1, N, 5):  # Sample magnitudes
                    if time.time() > deadline - 3:
                        break
                    
                    H_old = H[idx].copy()
                    angle = np.angle(H_old)
                    mag = np.abs(H_old)
                    eps = 0.05 * mag
                    
                    H[idx] = (mag + eps) * np.exp(1j * angle)
                    h_plus = np.fft.irfft(H, N).real
                    score_plus = _score(h_plus)
                    
                    H[idx] = (mag - eps) * np.exp(1j * angle)
                    h_minus = np.fft.irfft(H, N).real
                    score_minus = _score(h_minus)
                    
                    H[idx] = H_old
                    grad = (score_plus - score_minus) / (2 * eps)
                    
                    # Update magnitude
                    new_mag = max(0, mag - 0.1 * grad)
                    H[idx] = new_mag * np.exp(1j * angle)
        
        # Enforce conjugate symmetry for real output
        for k in range(1, N // 2 + 1):
            if k < N - k:
                H[N - k] = np.conj(H[k])
    
    return best_h, best_score

def _local_search_sign_flips(h):
    """Local search by flipping signs of individual elements."""
    best_h = h.copy()
    best_score = _score(h)
    
    t_start = time.time()
    deadline = t_start + 30.0
    
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for idx in range(N):
            if time.time() > deadline - 2:
                break
            best_h[idx] = -best_h[idx]
            new_score = _score(best_h)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                best_h[idx] = -best_h[idx]
        
        if not improved:
            break
    
    return best_h, best_score

def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin before 120s limit
    
    best_score = np.inf
    best_h = np.ones(N) / N  # Baseline: constant sequence
    
    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()
    
    # Strategy 1: Spectral phase optimization
    if time.time() < deadline - 60:
        h1, score1 = _optimize_phases_spectral()
        if score1 < best_score:
            best_score = score1
            best_h = h1.copy()
    
    # Strategy 2: Joint magnitude and phase optimization
    if time.time() < deadline - 40:
        h2, score2 = _optimize_magnitudes_and_phases()
        if score2 < best_score:
            best_score = score2
            best_h = h2.copy()
    
    # Strategy 3: Local search refinement
    if time.time() < deadline - 20:
        h3, score3 = _local_search_sign_flips(best_h)
        if score3 < best_score:
            best_score = score3
            best_h = h3.copy()
    
    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N
    
    return best_h

# EVOLVE-BLOCK-END
