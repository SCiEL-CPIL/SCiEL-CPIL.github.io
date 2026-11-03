"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _get_fft_size(n):
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    return fsize, size

def _autoconv_fft(h):
    """Compute full linear autoconvolution using FFT."""
    n = len(h)
    fsize, size = _get_fft_size(n)
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

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

def _soft_max_abs(ac, beta=50.0):
    """
    Smooth approximation of max(|ac|) using log-sum-exp.
    """
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=50.0):
    """
    Smooth objective: soft_max_abs(autoconv(h)) / (sum(h))^2
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _gen_chirp(n):
    """Generate a chirp sequence."""
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_sinc(n, beta=5.0):
    """Generate a sinc-like sequence."""
    i = np.arange(n)
    x = (i - n/2) / (n * beta)
    h = np.sinc(x)
    return h

def _gen_gaussian(n, sigma_frac=0.25):
    """Generate a Gaussian window."""
    i = np.arange(n)
    mu = n / 2
    sigma = n * sigma_frac
    return np.exp(-(i - mu)**2 / (2 * sigma**2))

def _gen_barker_like(n):
    """Generate a sequence with alternating pattern."""
    h = np.zeros(n)
    for i in range(n):
        if i % 4 in [0, 2]:
            h[i] = 1.0
        else:
            h[i] = -1.0
    return h

def _gen_msequence_like(n, period=7):
    """Generate an m-sequence-like pattern."""
    h = np.zeros(n)
    for i in range(n):
        val = ((i // period) % 2) ^ ((i % period) % 2)
        h[i] = 1.0 if val == 0 else -1.0
    return h

def _gen_random_smoothed(n, seed=42, kernel_size=10):
    """Generate a random smoothed sequence."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(noise, kernel, mode='same')

def _gen_binary_random(n, seed=42):
    """Generate a random binary (+1/-1) sequence."""
    rng = np.random.default_rng(seed)
    return np.where(rng.random(n) < 0.5, 1.0, -1.0)

def _compute_ac_delta(h, i, sign):
    """
    Compute the change in autoconvolution when h[i] changes by delta.
    If we flip h[i] to -h[i], delta = -2*h[i].
    The change in autoconv is delta * (h shifted by i) + h * (delta shifted by i)
    Actually, ac_new[k] = ac_old[k] + delta * (h[k-i] + h[k-i] for the two terms)
    More precisely: (h+dh)*(h+dh) = h*h + dh*h + h*dh + dh*dh
    So ac_new = ac_old + dh * h + h * dh + dh * dh
    where dh is a vector with delta at position i and 0 elsewhere.
    dh * h is delta * h shifted right by i (i.e., (dh*h)[k] = delta * h[k-i] if 0<=k-i<N)
    h * dh is delta * h shifted left by i (i.e., (h*dh)[k] = delta * h[k-i] if 0<=k-i<N)
    Wait, let me be more careful.
    (h * h)[k] = sum_j h[j] h[k-j]
    If we change h[i] by delta, then:
    (h_new * h_new)[k] = sum_j h_new[j] h_new[k-j]
    = sum_j (h[j] + delta*1[j=i]) (h[k-j] + delta*1[k-j=i])
    = sum_j h[j]h[k-j] + delta * sum_j h[j] 1[k-j=i] + delta * sum_j h[k-j] 1[j=i] + delta^2 * 1[j=i]1[k-j=i]
    = ac_old[k] + delta * h[k-i] + delta * h[k-i] + delta^2 * 1[k=2i]
    = ac_old[k] + 2*delta*h[k-i] + delta^2 * (1 if k==2i else 0)
    
    So the change is:
    - For all k where 0 <= k-i < N: ac[k] += 2*delta*h[k-i]
    - Additionally, ac[2*i] += delta^2 (if 2*i < 2*N-1)
    """
    n = len(h)
    size = 2 * n - 1
    ac_change = np.zeros(size)
    
    # 2*delta*h[k-i] for k in [i, i+n-1]
    # h[k-i] for k-i in [0, n-1]
    ac_change[i:i+n] += 2 * sign * h
    
    # delta^2 at position 2*i
    if 2 * i < size:
        ac_change[2 * i] += sign * sign
    
    return ac_change

def _local_search_flip(h, max_passes=5, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Uses incremental autoconvolution update for efficiency.
    Multiple passes until no improvement.
    """
    h = h.copy()
    ac = _autoconv_fft(h)
    s = np.sum(h)
    best_val = np.max(np.abs(ac)) / (s * s)
    cur_val = best_val
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            # Try flipping h[i]
            delta = -2 * h[i]
            ac_change = _compute_ac_delta(h, i, delta)
            ac_new = ac + ac_change
            s_new = s + delta
            
            if abs(s_new) < 1e-9:
                continue
            
            new_val = np.max(np.abs(ac_new)) / (s_new * s_new)
            
            if new_val < cur_val:
                # Accept the flip
                ac = ac_new
                s = s_new
                h[i] = -h[i]
                cur_val = new_val
                improved = True
                if cur_val < best_val:
                    best_val = cur_val
            # else: reject, no change needed
        
        if not improved:
            break
    
    return h, best_val

def _local_search_perturb(h, max_passes=3, deadline=None, step_size=0.1):
    """
    Local optimization with small perturbations.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(123)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            old_val = h[i]
            h[i] = old_val + step_size * rng.standard_normal()
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = old_val
        if not improved:
            break
    return h, best_val

def _simulated_annealing(h, t_start, deadline, max_iter=50000):
    """
    Simulated annealing on sign flips with incremental AC update.
    """
    h = h.copy()
    ac = _autoconv_fft(h)
    s = np.sum(h)
    best_h = h.copy()
    best_ac = ac.copy()
    best_s = s
    best_val = np.max(np.abs(ac)) / (s * s)
    cur_val = best_val
    
    T = 1.0
    T_min = 1e-4
    cooling = 0.9995
    
    rng = np.random.default_rng(999)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick random index to flip
        i = rng.integers(0, N)
        delta = -2 * h[i]
        ac_change = _compute_ac_delta(h, i, delta)
        ac_new = ac + ac_change
        s_new = s + delta
        
        if abs(s_new) < 1e-9:
            T *= cooling
            continue
        
        new_val = np.max(np.abs(ac_new)) / (s_new * s_new)
        
        delta_val = new_val - cur_val
        if delta_val < 0 or rng.random() < np.exp(-delta_val / max(T, 1e-15)):
            # Accept
            ac = ac_new
            s = s_new
            h[i] = -h[i]
            cur_val = new_val
            if cur_val < best_val:
                best_val = cur_val
                best_h = h.copy()
                best_ac = ac.copy()
                best_s = s
        else:
            # Reject - no change needed
            pass
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_h, best_val

def _iterative_spectral_phase_correction(h, max_iter=5000, deadline=None):
    """
    Iterative spectral phase correction to minimize max|autoconv|.
    
    Key insight: By Parseval, the autoconvolution in time domain corresponds
    to |H(f)|^2 in frequency domain. The max of the time-domain autoconv is
    related to the peak of |H(f)|^2. We use an iterative algorithm similar to
    PAPR reduction:
    
    1. Compute FFT of h
    2. In frequency domain, modify phase to reduce peaks
    3. Transform back, enforce constraints (e.g., sum=1)
    4. Repeat
    
    This is much faster per iteration than time-domain local search.
    """
    h = h.copy().astype(float)
    n = len(h)
    s0 = np.sum(h)
    if abs(s0) < 1e-9:
        return h, _score(h)
    
    # Normalize to sum = 1
    h = h / s0
    
    # Precompute FFT size
    fsize, size = _get_fft_size(n)
    
    best_val = _score(h)
    best_h = h.copy()
    
    # Get initial FFT
    H = np.fft.rfft(h, fsize)
    
    # Track the peak locations in time domain
    ac = _autoconv_fft(h)
    peak_idx = np.argmax(np.abs(ac))
    
    # Iterative refinement
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute current autoconv and find peaks
        ac = _autoconv_fft(h)
        abs_ac = np.abs(ac)
        max_val = np.max(abs_ac)
        cur_val = max_val / (np.sum(h) ** 2)
        
        if cur_val < best_val:
            best_val = cur_val
            best_h = h.copy()
        
        # Find the time-domain peak that we want to reduce
        # The autoconvolution peak at lag k corresponds to correlation at that lag
        peak_lag = np.argmax(abs_ac)
        
        # In frequency domain, the peak at lag k in autoconv is influenced by
        # the phase of H(f). We adjust the phase to reduce this peak.
        
        # Method: Gradient descent on phase in frequency domain
        # The autoconv at lag k is: sum_j h[j+k]*h[j] (with appropriate bounds)
        # In FFT domain, this is related to the Fourier transform of |H|^2
        
        # Alternative approach: Use the fact that we want to minimize
        # max_k |sum_j h[j]h[j+k]|. 
        # 
        # A simpler iterative method: 
        # 1. Compute H = FFT(h)
        # 2. Compute |H|^2 (power spectrum)
        # 3. The inverse FFT of |H|^2 gives the autoconvolution
        # 4. To reduce peaks in autoconv, we can modify the phase of H
        
        # Phase-only modification approach:
        # We want to keep |H| fixed (to preserve energy) but adjust phase
        # to reduce time-domain peaks.
        
        # Compute current power spectrum
        power = np.abs(H) ** 2
        
        # Compute autoconv from power spectrum
        ac_from_power = np.fft.irfft(power, fsize)[:size]
        
        # Find which lags have high values
        # We want to reduce the maximum, so focus on the peak
        peak_lag = np.argmax(np.abs(ac_from_power))
        
        # The contribution to autoconv at lag k from frequency f is:
        # |H(f)|^2 * exp(-i*2*pi*f*k/N)
        # To reduce the peak at lag k, we want to add a phase shift that
        # makes the contributions at that lag cancel out.
        
        # Simple phase adjustment: add a linear phase in frequency
        # that corresponds to a shift in time, then re-optimize
        # Actually, let's use a different approach:
        
        # Iterative water-filling in frequency domain:
        # We want to minimize max_k |IFFT(|H|^2)[k]|
        # This is equivalent to making |H|^2 as flat as possible in a sense
        
        # Let's try: adjust the amplitude of H to reduce peaks
        # While keeping the total energy (sum of |H|^2) constant
        
        # Current autoconv
        ac_cur = np.fft.irfft(np.abs(H)**2, fsize)[:size]
        max_ac = np.max(np.abs(ac_cur))
        
        if max_ac < 1e-10:
            break
        
        # For each frequency bin, we can scale |H(f)|
        # We want to reduce the peak in time domain
        # 
        # Gradient of max|ac| with respect to |H(f)|:
        # d(max|ac|)/d|H(f)| = sign(ac_peak) * 2*H(f)*conj(H(f)) / |H(f)|
        #                     * (contribution of frequency f to ac at peak lag)
        #
        # This is complex. Let's use a simpler heuristic:
        # Reduce the amplitude of frequencies that contribute most to the peak
        
        # Compute the contribution of each frequency to the peak lag
        freqs = np.fft.rfftfreq(fsize, 1.0/fsize)
        k_peak = peak_lag
        # Contribution of frequency f to ac at lag k:
        # |H(f)|^2 * cos(2*pi*f*k) for real signals
        contributions = np.abs(H)**2 * np.cos(2*np.pi*freqs*k_peak)
        
        # Frequencies with positive contribution increase the peak
        # Frequencies with negative contribution decrease it
        # We want to reduce positive contributions
        
        # Scale down frequencies with positive contribution
        scale = 1.0 - 0.01 * np.clip(contributions / (np.max(np.abs(contributions)) + 1e-15), 0, 1)
        
        # Apply scaling to H
        H_new = H * scale
        
        # Transform back
        h_new = np.fft.irfft(H_new, fsize)[:n]
        
        # Enforce sum constraint: adjust to maintain sum = 1
        s_new = np.sum(h_new)
        if abs(s_new) > 1e-9:
            h_new = h_new / s_new
        else:
            h_new = h  # Keep old if sum is too small
        
        # Check if this improved things
        ac_new = _autoconv_fft(h_new)
        val_new = np.max(np.abs(ac_new)) / (np.sum(h_new)**2)
        
        if val_new < cur_val:
            h = h_new
            H = np.fft.rfft(h, fsize)
        else:
            # Try the opposite: scale up frequencies with negative contribution
            scale = 1.0 + 0.01 * np.clip(-contributions / (np.max(np.abs(contributions)) + 1e-15), 0, 1)
            H_new = H * scale
            h_new = np.fft.irfft(H_new, fsize)[:n]
            s_new = np.sum(h_new)
            if abs(s_new) > 1e-9:
                h_new = h_new / s_new
            else:
                break
            
            ac_new = _autoconv_fft(h_new)
            val_new = np.max(np.abs(ac_new)) / (np.sum(h_new)**2)
            
            if val_new < cur_val:
                h = h_new
                H = np.fft.rfft(h, fsize)
            else:
                # No improvement, break
                break
    
    # Final evaluation
    final_val = _score(h)
    if final_val < best_val:
        best_val = final_val
        best_h = h.copy()
    
    return best_h, best_val


def _spectral_optimization(h_init, max_iter=2000, deadline=None):
    """
    Full spectral optimization: combine iterative phase correction with
    local search refinement.
    """
    h = h_init.copy()
    s0 = np.sum(h)
    if abs(s0) < 1e-9:
        return h, _score(h)
    
    h = h / s0
    
    # Phase 1: Iterative spectral phase correction
    h, val = _iterative_spectral_phase_correction(h, max_iter=max_iter, deadline=deadline)
    
    # Phase 2: Local search refinement
    if deadline is not None and time.time() < deadline - 2:
        h, val = _local_search_flip(h, max_passes=2, deadline=deadline)
    
    return h, val


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s limit

    best_score = np.inf
    best_h = np.ones(N) / N  # Baseline: constant sequence

    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    # Initialization strategies - diverse set
    inits = []
    
    # 1. All ones
    inits.append(np.ones(N))
    
    # 2. Cosine windows
    x = np.arange(N)
    inits.append(np.cos(np.pi * x / N))
    inits.append(np.cos(np.pi * x / (2*N))**2)
    
    # 3. Gaussian windows
    inits.append(_gen_gaussian(N, sigma_frac=0.25))
    inits.append(_gen_gaussian(N, sigma_frac=0.15))
    inits.append(_gen_gaussian(N, sigma_frac=0.35))
    
    # 4. Triangle window
    mu = N/2
    inits.append(np.maximum(0, 1 - np.abs(x - mu)/mu))
    
    # 5. Chirp sequence
    inits.append(_gen_chirp(N))
    
    # 6. Sinc sequences
    inits.append(_gen_sinc(N, beta=3.0))
    inits.append(_gen_sinc(N, beta=5.0))
    
    # 7. Barker-like sequence
    inits.append(_gen_barker_like(N))
    
    # 8. M-sequence-like
    inits.append(_gen_msequence_like(N, period=7))
    
    # 9. Random smoothed sequences
    for seed in [42, 123, 456, 789, 101, 202, 303]:
        inits.append(_gen_random_smoothed(N, seed=seed))
    
    # 10. Random binary sequences
    for seed in [7, 17, 27, 37]:
        inits.append(_gen_binary_random(N, seed=seed))
    
    # Phase 1: Spectral optimization on diverse initializations
    phase1_deadline = t_start + 100.0  # 100s for spectral optimization
    
    for h_init in inits:
        if time.time() > phase1_deadline:
            break
            
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        # Normalize to sum 1
        h_norm = h_init / s0
        
        try:
            h_opt, val = _spectral_optimization(h_norm, max_iter=1000, deadline=phase1_deadline - 5)
            if val < best_score:
                best_score = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Phase 2: Intensive discrete search on the best solution
    phase2_deadline = t_start + 135.0  # Leave 5s buffer
    
    # Simulated annealing - the main workhorse
    if time.time() < phase2_deadline - 2:
        h_sa, val_sa = _simulated_annealing(best_h, t_start, phase2_deadline - 1, max_iter=100000)
        if val_sa < best_score:
            best_score = val_sa
            best_h = h_sa.copy()
    
    # Final local search polish
    if time.time() < phase2_deadline:
        h_final, val_final = _local_search_flip(best_h, max_passes=5, deadline=phase2_deadline)
        if val_final < best_score:
            best_score = val_final
            best_h = h_final.copy()
    
    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N

    return best_h

# EVOLVE-BLOCK-END
