import numpy as np
import time

N = 600
START_TIME = time.time()
TIME_LIMIT = 135  # safety margin before 150s

# Precompute FFT setup
K = 32
M_STEP = N * K  # 19200
M_FFT = 1 << 15  # 32768

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution."""
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n  # width of each interval
    
    # Build fine-sampled function
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = np.repeat(heights, K)
    
    # Compute autoconvolution via FFT
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    max_conv = np.max(conv)
    integral_f = np.sum(heights) * w
    
    if integral_f <= 0:
        return np.inf
    
    # Normalized max: max_conv / (integral_f)^2
    score = max_conv / (integral_f ** 2)
    
    return score

def compute_score_fast(heights):
    """Faster version using precomputed structures for optimization."""
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return 1e10
    heights = np.maximum(heights, 0.0)
    
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = np.repeat(heights, K)
    
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    max_conv = np.max(conv)
    integral_f = np.sum(heights) * (0.5 / N)
    
    if integral_f <= 1e-10:
        return 1e10
    
    return max_conv / (integral_f ** 2)

def _spectral_params_to_heights(params):
    """
    Convert spectral parameters to time-domain heights.
    params: [sigma, amp0, amp1, amp2, amp3, amp4, phase1, phase2, phase3, phase4]
    where sigma controls the Gaussian width in frequency domain,
    amp_k are coefficients for polynomial modulation,
    phase_k are phase offsets.
    
    The power spectrum is: P(k) = amp0 * exp(-(k/sigma)^2) * (1 + sum amp_k * cos(phase_k + k * 2pi/N))
    Then we take the inverse DFT and ensure non-negativity.
    """
    n_params = len(params)
    sigma = max(params[0], 0.5)  # frequency spread
    amps = params[1:5]
    phases = params[5:9]
    
    # Frequency indices
    k = np.arange(N)
    
    # Base Gaussian spectrum
    base = np.exp(-(k / sigma) ** 2)
    
    # Modulate with cosine terms
    mod = np.ones(N)
    for i in range(4):
        mod += amps[i] * np.cos(phases[i] + 2 * np.pi * k / N * (i + 1))
    
    # Ensure non-negative spectrum
    spectrum = np.maximum(base * mod, 0.0)
    
    # Inverse DFT to get heights
    # We want real, non-negative heights
    # Use rfft/irfft for efficiency
    # Pad to even length if needed
    N_fft = N
    if N_fft % 2 != 0:
        N_fft += 1
    
    # Build full spectrum for irfft
    # irfft expects: [DC, positive freqs]
    # We have N frequencies from 0 to N-1
    # For N even: irfft input has N//2 + 1 elements
    # For N odd: irfft input has (N+1)//2 elements
    
    if N_fft % 2 == 0:
        # Even case
        half = N_fft // 2
        # spectrum has N elements, but we need N//2 + 1 for irfft
        # Take the first half + 1
        spec_half = spectrum[:half + 1].copy()
        # Ensure symmetry by setting Nyquist frequency properly
        heights = np.fft.irfft(spec_half, n=N_fft)[:N]
    else:
        # Odd case
        half = (N_fft + 1) // 2
        spec_half = spectrum[:half].copy()
        heights = np.fft.irfft(spec_half, n=N_fft)[:N]
    
    # Ensure non-negativity
    heights = np.maximum(heights, 0.0)
    
    # Normalize so that the integral is 1 (or some fixed value)
    integral = np.sum(heights) * (0.5 / N)
    if integral > 1e-10:
        heights = heights / integral  # normalize so integral = 1
    
    return heights

def _spectral_objective(params):
    """Objective for spectral parameter optimization."""
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return 1e10
    
    heights = _spectral_params_to_heights(params)
    return compute_score_fast(heights)

def _spectral_grad(params):
    """Finite difference gradient for spectral parameters."""
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return np.zeros_like(params)
    
    f0 = _spectral_objective(params)
    grad = np.zeros_like(params)
    h = 1e-4
    
    for i in range(len(params)):
        params_p = params.copy()
        params_p[i] += h
        fp = _spectral_objective(params_p)
        grad[i] = (fp - f0) / h
    
    return grad

def construct():
    global START_TIME
    if time.time() - START_TIME > TIME_LIMIT:
        return [1.0] * N
    
    # Start with uniform (known to give score 1.0)
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    x = np.linspace(0, 1, N)
    
    # Strategy 1: Spectral shaping via parameterized power spectrum
    # Parameters: [sigma, amp0, amp1, amp2, amp3, amp4, phase1, phase2, phase3, phase4]
    
    # Try a grid of spectral parameters
    best_spectral_score = best_score
    best_spectral_h = best_h.copy()
    
    sigma_values = [10, 20, 30, 50, 75, 100]
    amp_values = [0.0, 0.1, 0.2, 0.3, 0.5]
    
    for sigma in sigma_values:
        for amp in amp_values:
            # Try different phase combinations
            for phase_base in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
                if time.time() - START_TIME > TIME_LIMIT - 30:
                    break
                
                # Simple spectrum: Gaussian with one cosine modulation
                params = np.array([sigma, 1.0, amp, 0.0, 0.0, 0.0,
                                   phase_base, 0.0, 0.0, 0.0])
                heights = _spectral_params_to_heights(params)
                score = compute_score_fast(heights)
                
                if score < best_spectral_score:
                    best_spectral_score = score
                    best_spectral_h = heights.copy()
            
            # Two-modulation spectrum
            for amp2 in [0.0, 0.1, 0.2]:
                for phase2 in [0, np.pi/2, np.pi]:
                    if time.time() - START_TIME > TIME_LIMIT - 30:
                        break
                    
                    params = np.array([sigma, 1.0, amp, amp2, 0.0, 0.0,
                                       phase_base, phase2, 0.0, 0.0])
                    heights = _spectral_params_to_heights(params)
                    score = compute_score_fast(heights)
                    
                    if score < best_spectral_score:
                        best_spectral_score = score
                        best_spectral_h = heights.copy()
    
    if best_spectral_h is not None and best_spectral_score < best_score:
        best_score = best_spectral_score
        best_h = best_spectral_h.copy()
    
    # Strategy 2: Direct time-domain optimization with smooth basis
    # Use a small set of smooth basis functions to parameterize heights
    
    # Cosine basis (DCT-like)
    num_basis = 20
    basis = np.zeros((num_basis, N))
    for k in range(num_basis):
        basis[k] = np.cos(np.pi * k * x)
    
    # Optimize coefficients
    def _basis_objective(coeffs):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            return 1e10
        heights = np.sum(coeffs[:, None] * basis[None, :], axis=0)
        heights = np.maximum(heights, 0.0)
        return compute_score_fast(heights)
    
    # Grid search over a few coefficient patterns
    for c0 in [1.0, 1.5, 2.0]:
        for c1 in [-0.5, 0.0, 0.5]:
            for c2 in [-0.3, 0.0, 0.3]:
                if time.time() - START_TIME > TIME_LIMIT - 20:
                    break
                
                coeffs = np.zeros(num_basis)
                coeffs[0] = c0
                coeffs[1] = c1
                coeffs[2] = c2
                
                heights = np.sum(coeffs[:, None] * basis[None, :], axis=0)
                heights = np.maximum(heights, 0.0)
                score = compute_score_fast(heights)
                
                if score < best_score:
                    best_score = score
                    best_h = heights.copy()
    
    # Strategy 3: Try various smooth shapes as additional candidates
    for amp in [0.1, 0.2, 0.3, 0.4, 0.5]:
        h_cos = 1.0 - amp * 0.5 * (1 - np.cos(2 * np.pi * x))
        h_cos = np.maximum(h_cos, 0.0)
        score = compute_score(h_cos)
        if score < best_score:
            best_score = score
            best_h = h_cos.copy()
    
    for taper_frac in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        h_plateau = np.ones(N)
        taper_len = int(taper_frac * N)
        if taper_len > 0:
            ramp = np.linspace(0, 1, taper_len)
            h_plateau[:taper_len] = ramp
            h_plateau[-taper_len:] = ramp[::-1]
        score = compute_score(h_plateau)
        if score < best_score:
            best_score = score
            best_h = h_plateau.copy()
    
    for sigma in [0.1, 0.15, 0.2, 0.25, 0.3]:
        h_gauss = np.exp(-((x - 0.5) ** 2) / (2 * sigma ** 2))
        h_gauss = h_gauss / np.max(h_gauss)
        score = compute_score(h_gauss)
        if score < best_score:
            best_score = score
            best_h = h_gauss.copy()
    
    # Strategy 4: Random smooth perturbations around the best shape
    if time.time() - START_TIME < TIME_LIMIT - 10:
        rng = np.random.default_rng(42)
        for trial in range(30):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            
            # Create smooth perturbation using low-frequency Fourier modes
            g = np.zeros(N)
            num_modes = rng.integers(2, 6)
            for _ in range(num_modes):
                k = rng.integers(1, N // 4)
                phase = rng.uniform(0, 2 * np.pi)
                amp = rng.uniform(0.05, 0.15)
                g += amp * np.cos(2 * np.pi * k * x + phase)
            
            g -= np.mean(g)
            
            for eps in [0.05, 0.1, 0.15, 0.2]:
                h_trial = best_h + eps * g
                h_trial = np.maximum(h_trial, 0.0)
                if np.sum(h_trial) <= 0:
                    continue
                score = compute_score(h_trial)
                if score < best_score:
                    best_score = score
                    best_h = h_trial.copy()
    
    # Final: try scaling
    for scale in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        h_scaled = best_h * scale
        score = compute_score(h_scaled)
        if score < best_score:
            best_score = score
            best_h = h_scaled.copy()
    
    best_h = np.maximum(best_h, 0.0)
    return [max(0.0, float(v)) for v in best_h]

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
