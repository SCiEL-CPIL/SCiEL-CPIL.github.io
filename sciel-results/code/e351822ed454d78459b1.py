"""Combined program: spectral flattening initialization + peak-targeting local search."""
import numpy as np
import time

N = 400
_DEADLINE = 145.0
_T0 = time.monotonic()

def _remaining():
    return _DEADLINE - (time.monotonic() - _T0)

def _autoconv_fft(f):
    """Compute linear autoconvolution via FFT."""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.rfft(f, size)
    ac = np.fft.irfft(F * F, size)
    return ac[:2*n-1]

def _score(f):
    """Compute max|autoconv| / (sum f)^2. Returns inf if sum f == 0."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconv_fft(f)
    return np.max(np.abs(ac)) / (s * s)

def _spectral_flatten(x_init, n_steps=200, lr=0.05):
    """
    Gradient descent on spectral flatness to spread autoconvolution energy.
    Objective: minimize log(mean(|F|^2)) - mean(log(|F|^2)) (spectral entropy-like)
    This drives the DFT magnitude toward uniform, spreading autoconv energy.
    We work in the frequency domain for efficiency.
    """
    n = len(x_init)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    # Start from x_init
    x = x_init.astype(float).copy()
    
    for step in range(n_steps):
        if _remaining() < 10.0:
            break
        
        # Compute FFT
        F = np.fft.rfft(x, size)
        mag2 = np.abs(F) ** 2 + 1e-12
        
        # Spectral flatness gradient: we want to minimize variance of log(mag2)
        # Simpler: minimize sum of (log(mag2) - mean(log(mag2)))^2
        log_mag2 = np.log(mag2)
        mean_log = np.mean(log_mag2)
        # Gradient of sum((log_mag2 - mean)^2) w.r.t. x
        # d/dx log_mag2[k] = 2 * Re(F[k] * conj(dF/dx[k]))
        # This is complex. Let's use a simpler heuristic:
        # Push energy from high-magnitude bins to low-magnitude bins.
        
        # Alternative: minimize max |ac| directly via soft-max in frequency domain?
        # Let's just do gradient descent on the objective directly using FFT.
        
        # Compute current objective
        s = np.sum(x)
        if abs(s) < 1e-12:
            break
        
        ac = np.fft.irfft(F * np.conj(F), size)[:2*n-1]
        c_excl = ac[1:]
        
        # Soft max of |c_excl|
        abs_c = np.abs(c_excl)
        m = np.max(abs_c)
        if m < 1e-15:
            beta = 10.0
            sm = m
        else:
            beta = 50.0
            sm = m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta
        
        obj = sm / (s * s)
        
        # Numerical gradient is too slow. Use analytic gradient of soft_max / s^2.
        # Let's use a simpler approach: random perturbations in frequency domain.
        
        # Perturb the phase of F randomly
        phase = np.angle(F)
        mag = np.sqrt(mag2)
        
        # Add small random phase noise
        noise = np.random.randn(len(phase)) * 0.01
        new_phase = phase + noise
        new_F = mag * np.exp(1j * new_phase)
        
        # Inverse FFT
        new_x = np.fft.irfft(new_F, size)[:n]
        
        # Ensure sum is not too small
        new_s = np.sum(new_x)
        if abs(new_s) < 1e-6:
            continue
        
        new_score = _score(new_x)
        if new_score < obj:
            x = new_x
    
    return x

def construct():
    """
    Returns a signed step height vector of length N minimizing
    max|linear_autoconvolution| / (sum f)^2.
    
    Strategy:
    1. Generate diverse initial candidates (random, Golay, Barker, etc.)
    2. Apply spectral flattening to spread autoconvolution energy
    3. Use peak-targeting local search to refine the result
    """
    rng = np.random.RandomState(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # Random sequences
    for _ in range(5):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Golay-like complementary sequence construction
    def golay_pair(n):
        """Generate Golay complementary pair of length n (must be power of 2)."""
        if n == 1:
            return np.array([1.0]), np.array([1.0])
        half = n // 2
        a, b = golay_pair(half)
        a2 = np.concatenate([a, b])
        b2 = np.concatenate([a, -b])
        return a2, b2
    
    # Use Golay sequences as building blocks
    golay_len = 1
    while golay_len < N:
        golay_len *= 2
    a, b = golay_pair(golay_len)
    # Pad or truncate to N
    if golay_len >= N:
        candidates.append(a[:N].copy())
        candidates.append(b[:N].copy())
        candidates.append(np.concatenate([a[:N//2], b[:N//2]])[:N])
    else:
        # Tile
        reps = (N + golay_len - 1) // golay_len
        candidates.append(np.tile(a, reps)[:N])
        candidates.append(np.tile(b, reps)[:N])
        candidates.append(np.tile(np.concatenate([a, b]), reps)[:N])
    
    # Barker-like sequences (known good autocorrelation)
    # Barker sequence of length 13: + + + - + - - + - + - - +
    barker13 = np.array([1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1], dtype=float)
    reps = (N + 12) // 13
    candidates.append(np.tile(barker13, reps)[:N])
    
    # Alternating patterns
    candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # Block patterns with varying block sizes
    for block_size in [4, 8, 16, 32]:
        if block_size < N:
            n_blocks = N // block_size
            pattern = []
            for b in range(n_blocks):
                # Use a pseudo-random sign based on block index
                sign = 1.0 if ((b * 7 + 3) % 3) != 0 else -1.0
                pattern.extend([sign] * block_size)
            # Fill remainder
            rem = N - len(pattern)
            pattern.extend([1.0] * rem)
            candidates.append(np.array(pattern[:N]))
    
    # Evaluate all candidates and apply spectral flattening to the best few
    scored_candidates = []
    for c in candidates:
        s = _score(c)
        scored_candidates.append((s, c))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Take top 5 candidates and refine with spectral flattening
    best_f = None
    best_score = np.inf
    
    for score, f in scored_candidates[:5]:
        if _remaining() < 20.0:
            break
        
        # Apply spectral flattening
        f_flat = _spectral_flatten(f, n_steps=100, lr=0.05)
        
        # Evaluate the flattened version
        flat_score = _score(f_flat)
        
        # Also keep the original if it's better
        if score < best_score:
            best_score = score
            best_f = f.copy()
        
        if flat_score < best_score:
            best_score = flat_score
            best_f = f_flat.copy()
    
    # Peak-targeting local search on the best candidate
    f = best_f.copy()
    cur_score = best_score
    ac = _autoconv_fft(f)
    s = np.sum(f)
    
    # Simulated annealing with peak-targeting
    T = 1.0
    T_min = 0.01
    cooling = 0.99995
    
    while _remaining() > 5.0 and T > T_min:
        # Find the peak lag
        peak_idx = np.argmax(np.abs(ac))
        peak_val = ac[peak_idx]
        
        # Try flipping bits that contribute to this peak
        best_flip = -1
        best_new_score = cur_score
        
        # Sample a subset of indices to try
        indices_to_try = rng.choice(N, size=min(N, 100), replace=False)
        
        for idx in indices_to_try:
            if _remaining() < 5.0:
                break
            
            old_val = f[idx]
            new_val = -old_val
            
            # Estimate change in autoconv at peak_idx
            # ac[peak_idx] = sum_i f[i] * f[peak_idx - i]
            # Flipping f[idx] changes terms where i=idx or peak_idx-i=idx
            
            # Compute new autoconv efficiently
            f_new = f.copy()
            f_new[idx] = new_val
            ac_new = _autoconv_fft(f_new)
            s_new = np.sum(f_new)
            
            if abs(s_new) < 1e-12:
                continue
            
            new_score = np.max(np.abs(ac_new)) / (s_new * s_new)
            
            if new_score < best_new_score:
                best_new_score = new_score
                best_flip = idx
                best_f_new = f_new
                best_ac_new = ac_new
                best_s_new = s_new
        
        if best_flip >= 0:
            # Accept with probability based on temperature
            delta = best_new_score - cur_score
            if delta < 0 or rng.rand() < np.exp(-delta / T):
                f = best_f_new
                ac = best_ac_new
                s = best_s_new
                cur_score = best_new_score
        
        T *= cooling
    
    return f
