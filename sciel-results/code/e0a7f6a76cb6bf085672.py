"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr_peak_sidelobe(h):
    """Compute max |R[k]| for k != 0 using FFT-based linear autocorrelation."""
    n = len(h)
    nfft = 1 << (2 * n - 1).bit_length()
    H = np.fft.rfft(h, n=nfft)
    R = np.fft.irfft(H * np.conj(H), n=nfft)
    sidelobes_pos = R[1:n]
    sidelobes_neg = R[nfft - n + 1:nfft]
    all_sidelobes = np.concatenate([sidelobes_pos, sidelobes_neg])
    return np.max(np.abs(all_sidelobes))

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n, mapped to ±1."""
    bits = np.zeros(n, dtype=np.int8)
    for i in range(n):
        x = i
        cnt = 0
        while x:
            cnt += x & 1
            x >>= 1
        bits[i] = 1 if cnt % 2 == 0 else -1
    return bits

def _m_sequence(n, seed):
    """Generate an m-sequence-like pattern using LFSR."""
    result = np.ones(n, dtype=np.int8)
    state = seed
    for i in range(n):
        result[i] = 1 if (state & 1) == 0 else -1
        # Simple LFSR with polynomial x^8 + x^6 + x^5 + x^4 + 1
        feedback = ((state >> 7) ^ (state >> 5) ^ (state >> 4) ^ (state >> 3)) & 1
        state = ((state << 1) | feedback) & 0xFF
    return result

def _frequency_guided_optimize(h, n_steps, deadline, rng):
    """Optimize h by identifying dominant frequency components causing sidelobes
    and adjusting time-domain samples to suppress them."""
    best_h = h.copy()
    best_score = _autocorr_peak_sidelobe(h)
    n = len(h)
    nfft = 1 << (2 * n - 1).bit_length()
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Compute current spectrum
        H = np.fft.rfft(h, n=nfft)
        power = np.abs(H) ** 2
        
        # Identify frequency bins with highest power (excluding DC)
        # These correspond to the main lobes causing sidelobes
        power_no_dc = power.copy()
        power_no_dc[0] = 0
        top_freqs = np.argsort(power_no_dc)[-5:]
        
        # For each top frequency, compute the corresponding time-domain pattern
        # and try to anti-correlate with it
        for freq_idx in top_freqs:
            if time.time() > deadline:
                break
            
            # Create the sinusoidal pattern at this frequency
            freq = freq_idx / nfft
            pattern = np.cos(2 * np.pi * freq * np.arange(n))
            
            # Compute correlation of h with this pattern
            corr = np.dot(h, pattern)
            
            # If correlation is high, flip signs where pattern is strong
            # to reduce the contribution at this frequency
            if abs(corr) > 0.5:
                # Find positions where pattern has strongest amplitude
                threshold = np.abs(pattern).max() * 0.8
                strong_positions = np.where(np.abs(pattern) > threshold)[0]
                
                # Flip a subset of these positions
                if len(strong_positions) > 0:
                    n_flip = max(1, len(strong_positions) // 4)
                    flip_positions = rng.choice(strong_positions, size=min(n_flip, len(strong_positions)), replace=False)
                    
                    h_new = h.copy()
                    h_new[flip_positions] *= -1
                    
                    score_new = _autocorr_peak_sidelobe(h_new)
                    if score_new < best_score:
                        best_score = score_new
                        best_h = h_new.copy()
                        h = h_new
        
        # Also try random small perturbations to escape local minima
        if step % 10 == 0:
            h_new = h.copy()
            n_perturb = rng.integers(1, 5)
            perturb_positions = rng.choice(n, size=n_perturb, replace=False)
            h_new[perturb_positions] *= -1
            score_new = _autocorr_peak_sidelobe(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new.copy()
                h = h_new
    
    return best_h, best_score

def construct():
    t_start = time.time()
    deadline = t_start + 135.0  # 15s safety margin
    
    # Multiple diverse initializations
    initializations = [
        _thue_morse(N).astype(np.float64),
        _m_sequence(N, 0x55).astype(np.float64),
        _m_sequence(N, 0xAA).astype(np.float64),
        _m_sequence(N, 0x33).astype(np.float64),
        np.random.default_rng(0).choice([-1, 1], size=N).astype(np.float64),
    ]
    
    # Pick the best initialization
    best_h = None
    best_score = float('inf')
    for init in initializations:
        score = _autocorr_peak_sidelobe(init)
        if score < best_score:
            best_score = score
            best_h = init.copy()
    
    h = best_h.copy()
    
    # Phase 1: Frequency-guided optimization
    rng = np.random.default_rng(42)
    h, best_score = _frequency_guided_optimize(h, 500, deadline, rng)
    
    # Phase 2: Local search with targeted flips based on residual sidelobes
    if time.time() < deadline:
        n = N
        nfft = 1 << (2 * n - 1).bit_length()
        T_init = 10.0
        T_final = 0.01
        n_iterations = 30000
        
        for it in range(n_iterations):
            if time.time() > deadline:
                break
            
            T = T_init * (T_final / T_init) ** (it / n_iterations)
            
            # Compute current autocorrelation to find worst sidelobe
            H = np.fft.rfft(h, n=nfft)
            R = np.fft.irfft(H * np.conj(H), n=nfft)
            sidelobes_pos = R[1:n]
            sidelobes_neg = R[nfft - n + 1:nfft]
            all_sidelobes = np.concatenate([sidelobes_pos, sidelobes_neg])
            worst_idx = np.argmax(np.abs(all_sidelobes))
            
            # Determine if this is positive or negative lag
            if worst_idx < n - 1:
                lag = worst_idx + 1
            else:
                lag = -(nfft - n + 1 - worst_idx)
            
            # Try flipping positions that would reduce this specific sidelobe
            # The sidelobe at lag k is sum(h[i]*h[i+k]) for valid i
            # Flipping h[j] changes the sidelobe by -2*h[j]*h[j-k] (for positive lag)
            candidate_positions = []
            for j in range(n):
                if 0 <= j - lag < n:
                    # Flipping h[j] changes contribution at lag by -2*h[j]*h[j-lag]
                    change = -2 * h[j] * h[j - lag]
                    if change < 0:  # This flip would reduce the sidelobe magnitude
                        candidate_positions.append((j, abs(change)))
            
            if candidate_positions:
                # Sort by magnitude of change and try the most impactful flips
                candidate_positions.sort(key=lambda x: -x[1])
                for j, _ in candidate_positions[:5]:
                    h_new = h.copy()
                    h_new[j] *= -1
                    score_new = _autocorr_peak_sidelobe(h_new)
                    
                    delta = score_new - best_score
                    if delta < 0 or (T > 0 and np.log(rng.random() + 1e-30) > -delta / T):
                        h = h_new
                        if score_new < best_score:
                            best_score = score_new
                            best_h = h_new.copy()
                        break
    
    return best_h
# EVOLVE-BLOCK-END
