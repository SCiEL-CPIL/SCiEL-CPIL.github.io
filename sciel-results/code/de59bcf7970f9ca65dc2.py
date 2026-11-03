"""Program for signed step heights minimizing max|autoconv|/(sum h)^2.

This is a rewrite with a fundamentally different optimization strategy:
- Use a spectral parameterization that directly controls the power spectrum
- Exploit the fact that autoconvolution is the inverse FFT of |H(f)|^2
- Use a "spectral shaping" approach: design H(f) to minimize the peak of |H(f)|^2
  while maintaining sum(h) = 1 (i.e., H(0) = 1)
- Then apply local refinement in the time domain

The key insight: if we can make the power spectrum as flat as possible (while keeping
H(0) = 1), the autoconvolution will be as close to a delta function as possible,
minimizing its maximum absolute value.
"""

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac[:size]))

def _autoconv_full(h):
    """Compute full autoconvolution and its max absolute value."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    return ac, np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _spectral_shaping_init(seed=42):
    """
    Initialize h using spectral shaping: design the power spectrum to be
    as flat as possible while maintaining H(0) = sum(h) = 1.
    
    We work in the frequency domain:
    - H(f) is the FFT of h
    - H(0) = sum(h) = 1 (we fix this)
    - We want |H(f)|^2 to be as flat as possible for f != 0
    - But h must be real, so H(f) = conj(H(N-f))
    
    Strategy: start with a white-noise-like spectrum, normalize H(0)=1,
    then take IFFT to get h.
    """
    rng = np.random.default_rng(seed)
    
    # FFT size for the autoconvolution
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    # rfft size: fsize // 2 + 1
    rsize = fsize // 2 + 1
    
    # Initialize H with random complex values
    # H[0] is real (DC component)
    H = np.zeros(rsize, dtype=complex)
    
    # DC component: H[0] = 1 (so sum(h) = 1)
    H[0] = 1.0
    
    # For other frequencies, use random amplitudes and phases
    # We want the power spectrum to be relatively flat
    # Use a low-pass filtered random spectrum to get smooth h
    freqs = np.arange(rsize)
    
    # Apply a smooth window to suppress very high frequencies (for stability)
    # This gives us a band-limited signal
    cutoff = int(0.8 * rsize)
    window = np.ones(rsize)
    window[cutoff:] = 0.0
    # Smooth transition
    transition = int(0.1 * rsize)
    if transition > 0:
        for i in range(transition):
            frac = i / transition
            window[cutoff - transition + i] = 0.5 * (1 + np.cos(np.pi * frac))
    
    # Random amplitudes with some structure
    # Use a mix of different frequency content
    for trial in range(5):
        # Different spectral shapes
        if trial == 0:
            # White noise
            amps = rng.standard_normal(rsize)
        elif trial == 1:
            # Low-frequency biased
            amps = rng.standard_normal(rsize) * (1.0 / (1.0 + freqs * 0.01))
        elif trial == 2:
            # Band-pass
            center = rsize // 2
            amps = rng.standard_normal(rsize) * np.exp(-((freqs - center)**2) / (2 * (rsize/8)**2))
        elif trial == 3:
            # Multi-band
            amps = rng.standard_normal(rsize)
            for k in [rsize//4, rsize//2, 3*rsize//4]:
                amps[max(0,k-5):k+5] += rng.standard_normal(11) * 2
        else:
            # Sinc-like (concentrated in frequency)
            amps = np.sinc(freqs / rsize * 0.5) + rng.standard_normal(rsize) * 0.1
        
        H_trial = np.zeros(rsize, dtype=complex)
        H_trial[0] = 1.0
        H_trial[1:] = amps[1:] * window[1:]
        
        # Ensure H is conjugate symmetric for real h
        # rfft handles this: we only store non-negative frequencies
        # The IFFT will automatically create the symmetric version
        
        # Take IFFT to get h
        h_trial = np.fft.irfft(H_trial, fsize)[:N]
        
        # Normalize so sum(h) = 1
        s = np.sum(h_trial)
        if abs(s) > 1e-9:
            h_trial = h_trial / s
            val = _score(h_trial)
        else:
            continue
        
        if trial == 0 or not hasattr(_spectral_shaping_init, '_best_val'):
            _spectral_shaping_init._best_val = val
            _spectral_shaping_init._best_h = h_trial.copy()
        elif val < _spectral_shaping_init._best_val:
            _spectral_shaping_init._best_val = val
            _spectral_shaping_init._best_h = h_trial.copy()
    
    if hasattr(_spectral_shaping_init, '_best_h'):
        return _spectral_shaping_init._best_h.copy()
    return np.ones(N) / N

def _local_search_flip(h, max_passes=10, deadline=None):
    """Local search by sign flipping."""
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(12345)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        idx_order = rng.permutation(N)
        for i in idx_order:
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]
        if not improved:
            break
    return h, best_val

def _local_search_perturb(h, max_passes=10, deadline=None, step_size=0.02):
    """Local search by small perturbations."""
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(67890)
    
    current_step = step_size
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        idx_order = rng.permutation(N)
        for i in idx_order:
            if deadline is not None and time.time() > deadline:
                break
            old_val = h[i]
            best_local = old_val
            best_local_val = best_val
            for delta in [current_step, -current_step, 0.5*current_step, -0.5*current_step]:
                h[i] = old_val + delta
                new_val = _score(h)
                if new_val < best_local_val:
                    best_local_val = new_val
                    best_local = h[i]
            if best_local != old_val:
                h[i] = best_local
                best_val = best_local_val
                improved = True
            else:
                h[i] = old_val
        if not improved:
            break
        current_step *= 0.85
    return h, best_val

def _spectral_optimization(h_init, deadline):
    """
    Optimize in the spectral domain: find H(f) that minimizes the peak of
    the inverse FFT of |H(f)|^2, subject to H(0) = 1.
    
    We parameterize the amplitude spectrum and use coordinate descent.
    """
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    rsize = fsize // 2 + 1
    
    # Get initial spectrum
    H = np.fft.rfft(h_init, fsize)
    
    # Work with amplitude and phase
    amp = np.abs(H)
    phase = np.angle(H)
    
    # Fix DC: H[0] should be real and equal to 1
    amp[0] = 1.0
    phase[0] = 0.0
    
    best_val = _score(h_init)
    best_amp = amp.copy()
    best_phase = phase.copy()
    
    rng = np.random.default_rng(54321)
    step_size = 0.1
    
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        improved = False
        # Shuffle frequency indices
        freq_order = rng.permutation(1, rsize)
        
        for k in freq_order:
            if time.time() > deadline:
                break
            
            old_amp = amp[k]
            best_local_amp = old_amp
            best_local_val = best_val
            
            # Try several amplitude changes
            for delta in [step_size, -step_size, 0.5*step_size, -0.5*step_size]:
                amp[k] = max(0.0, old_amp + delta)
                H_temp = amp * np.exp(1j * phase)
                h_temp = np.fft.irfft(H_temp, fsize)[:N]
                s_temp = np.sum(h_temp)
                if abs(s_temp) < 1e-9:
                    continue
                h_temp = h_temp / s_temp
                val_temp = _score(h_temp)
                if val_temp < best_local_val:
                    best_local_val = val_temp
                    best_local_amp = amp[k]
            
            if best_local_amp != old_amp:
                amp[k] = best_local_amp
                best_val = best_local_val
                improved = True
            else:
                amp[k] = old_amp
        
        if not improved:
            break
        step_size *= 0.8
    
    # Get final h
    H_final = amp * np.exp(1j * phase)
    h_final = np.fft.irfft(H_final, fsize)[:N]
    s_final = np.sum(h_final)
    if abs(s_final) > 1e-9:
        h_final = h_final / s_final
        val_final = _score(h_final)
    else:
        h_final = h_init
        val_final = best_val
    
    return h_final, val_final

def construct():
    t_start = time.time()
    deadline = t_start + 145.0
    
    best_score = np.inf
    best_h = np.ones(N) / N
    base_score = _score(best_h)
    best_score = base_score
    
    # Phase 1: Generate diverse initializations using spectral shaping
    inits = []
    for seed in [42, 123, 456, 789, 101, 202]:
        if time.time() > deadline - 60:
            break
        h_init = _spectral_shaping_init(seed=seed)
        val_init = _score(h_init)
        inits.append((h_init, val_init))
        if val_init < best_score:
            best_score = val_init
            best_h = h_init.copy()
    
    # Also try a few structured initializations
    # Alternating pattern
    h_alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    s_alt = np.sum(h_alt)
    if abs(s_alt) > 1e-9:
        h_alt = h_alt / s_alt
        val_alt = _score(h_alt)
        inits.append((h_alt.copy(), val_alt))
        if val_alt < best_score:
            best_score = val_alt
            best_h = h_alt.copy()
    
    # Block pattern
    block_size = N // 8
    h_block = np.zeros(N)
    for b in range(4):
        h_block[b*block_size:(b+1)*block_size] = 1.0
        h_block[(b+4)*block_size:(b+5)*block_size] = -1.0
    s_block = np.sum(h_block)
    if abs(s_block) > 1e-9:
        h_block = h_block / s_block
        val_block = _score(h_block)
        inits.append((h_block.copy(), val_block))
        if val_block < best_score:
            best_score = val_block
            best_h = h_block.copy()
    
    # Phase 2: Spectral optimization on each initialization
    for h_init, val_init in inits:
        if time.time() > deadline - 30:
            break
        h_opt, val_opt = _spectral_optimization(h_init, deadline=deadline - 25)
        if val_opt < best_score:
            best_score = val_opt
            best_h = h_opt.copy()
    
    # Phase 3: Local search refinement on the best result
    phase3_deadline = deadline - 15
    if time.time() < phase3_deadline:
        h_flip, val_flip = _local_search_flip(best_h, max_passes=10, deadline=phase3_deadline - 5)
        if val_flip < best_score:
            best_score = val_flip
            best_h = h_flip.copy()
    
    if time.time() < phase3_deadline - 2:
        h_pert, val_pert = _local_search_perturb(best_h, max_passes=10, deadline=phase3_deadline - 1, step_size=0.02)
        if val_pert < best_score:
            best_score = val_pert
            best_h = h_pert.copy()
    
    # Phase 4: More aggressive local search
    phase4_deadline = deadline - 5
    if time.time() < phase4_deadline:
        h_flip2, val_flip2 = _local_search_flip(best_h, max_passes=5, deadline=phase4_deadline - 2)
        if val_flip2 < best_score:
            best_score = val_flip2
            best_h = h_flip2.copy()
    
    if time.time() < phase4_deadline - 1:
        h_pert2, val_pert2 = _local_search_perturb(best_h, max_passes=5, deadline=phase4_deadline - 0.5, step_size=0.01)
        if val_pert2 < best_score:
            best_score = val_pert2
            best_h = h_pert2.copy()
    
    # Phase 5: Final spectral optimization on the best result
    if time.time() < deadline - 2:
        h_spec_final, val_spec_final = _spectral_optimization(best_h, deadline=deadline - 1)
        if val_spec_final < best_score:
            best_score = val_spec_final
            best_h = h_spec_final.copy()
    
    # Final sign flip polish
    if time.time() < deadline - 0.5:
        h_final_flip, val_final_flip = _local_search_flip(best_h, max_passes=3, deadline=deadline - 0.2)
        if val_final_flip < best_score:
            best_score = val_final_flip
            best_h = h_final_flip.copy()
    
    return best_h

# EVOLVE-BLOCK-END
