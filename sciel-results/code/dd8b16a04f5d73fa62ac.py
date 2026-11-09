"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy import signal


def _autoconv_excl_lag0(f):
    """Compute autoconvolution excluding lag 0"""
    ac = np.convolve(f, f, mode='full')
    # Exclude lag 0 (index N-1 in full convolution)
    n = len(f)
    ac_no_lag0 = np.concatenate([ac[:n-1], ac[n:]])
    return np.max(np.abs(ac_no_lag0))


def _compute_objective(f):
    """Compute max |autoconv|_{k!=0} / (sum f)^2"""
    S = np.sum(f)
    if abs(S) < 1e-12:
        return np.inf
    max_ac = _autoconv_excl_lag0(f)
    return max_ac / (S * S)


def _spectral_shape(N, alpha=0.5, beta=1.0):
    """
    Design a frequency-domain amplitude profile that minimizes 
    max|IDFT| by shaping the power spectrum.
    
    The key insight: if we want a time-domain signal with small max absolute
    value, we need a frequency domain with controlled peaks. We use a 
    smooth envelope that suppresses high-frequency components while 
    maintaining low-frequency energy for a good sum.
    
    Parameters:
    - alpha: controls the roll-off (higher = more aggressive suppression)
    - beta: controls the shape of the transition band
    """
    # Frequency indices for N-point DFT
    freqs = np.fft.fftfreq(N)
    
    # Create a smooth envelope that is 1 at DC and decays away
    # Using a raised cosine / window-like function
    # Normalize frequencies to [0, 1] range
    abs_freqs = np.abs(freqs)
    max_freq = abs_freqs.max() if abs_freqs.max() > 0 else 1.0
    normalized = abs_freqs / max_freq
    
    # Smooth transition: 1 near DC, decays to ~0 at Nyquist
    # Using a power-law decay with smooth cutoff
    envelope = np.ones(N)
    for i in range(N):
        if normalized[i] > 0:
            # Smooth decay: starts at 1, goes to 0
            # Use a sigmoid-like function for smooth transition
            x = normalized[i]
            envelope[i] = 1.0 / (1.0 + alpha * (x / (1.0 - x + 1e-10)) ** beta)
    
    # Ensure DC component is exactly 1
    envelope[0] = 1.0
    
    # The signal in frequency domain: we want F[k] such that 
    # IDFT gives us something close to ±1
    # Start with a flat spectrum (all ones) and apply the envelope
    F = np.ones(N, dtype=complex)
    F *= envelope
    
    # Make it Hermitian symmetric (for real-valued time domain)
    # F[k] = conj(F[N-k]) for k=1,...,N-1
    for k in range(1, N // 2 + 1):
        if k < N - k:
            F[N - k] = np.conj(F[k])
        elif k == N - k:
            F[k] = np.real(F[k])
    
    # If N is even, the Nyquist frequency (k=N/2) must be real
    if N % 2 == 0:
        F[N // 2] = np.real(F[N // 2])
    
    # Inverse DFT to get time domain
    f_cont = np.fft.ifft(F).real
    
    return f_cont, F


def _round_to_signed(f_cont):
    """
    Round continuous values to signed integers ±1.
    Use a thresholding strategy: values above 0 → +1, below 0 → -1.
    But also try different thresholds and sign flips to find best.
    """
    # Basic thresholding at 0
    f_rounded = np.sign(f_cont)
    # Where sign is 0, assign +1
    f_rounded[f_rounded == 0] = 1.0
    
    # Ensure positive sum
    if np.sum(f_rounded) < 0:
        f_rounded = -f_rounded
    
    return f_rounded


def _refine_by_flipping(f, deadline, max_iters=50000):
    """Local search: try flipping signs to minimize objective"""
    best = f.copy()
    best_obj = _compute_objective(best)
    n = len(f)
    
    # Track which indices we've tried recently for diversity
    recent_flips = {}
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try flipping a random sign, avoiding recently flipped indices
        idx = np.random.randint(n)
        trial = best.copy()
        trial[idx] = -trial[idx]
        obj = _compute_objective(trial)
        
        if obj < best_obj:
            best = trial
            best_obj = obj
            recent_flips[idx] = it
        else:
            # Occasionally try a different index if stuck
            if it % 100 == 0:
                idx = np.random.randint(n)
                trial = best.copy()
                trial[idx] = -trial[idx]
                obj = _compute_objective(trial)
                if obj < best_obj:
                    best = trial
                    best_obj = obj
    
    return best, best_obj


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    np.random.seed(42)
    
    candidates = []
    
    # Strategy 1: Spectral shaping with different parameters
    for alpha in [0.3, 0.5, 0.7, 1.0, 1.5]:
        for beta in [0.5, 1.0, 1.5, 2.0]:
            f_cont, F = _spectral_shape(N, alpha=alpha, beta=beta)
            f_rounded = _round_to_signed(f_cont)
            candidates.append(f_rounded)
    
    # Strategy 2: Walsh-Hadamard based (keep as fallback)
    p = 1
    while p * 2 <= N:
        p *= 2
    wh = np.array([1.0])
    while len(wh) < p:
        wh = np.concatenate([wh, wh, wh, -wh])
    f_wh = np.zeros(N)
    for i in range(N):
        f_wh[i] = wh[i % p]
    if np.sum(f_wh) < 0:
        f_wh = -f_wh
    candidates.append(f_wh)
    
    # Strategy 3: All ones (baseline)
    candidates.append(np.ones(N))
    
    # Strategy 4: Random ±1 sequences
    for _ in range(3):
        f_rand = np.random.choice([-1.0, 1.0], size=N)
        if np.sum(f_rand) < 0:
            f_rand = -f_rand
        candidates.append(f_rand)
    
    # Evaluate all candidates and pick the best starting point
    best_f = candidates[0].copy()
    best_obj = _compute_objective(best_f)
    
    for f_init in candidates[1:]:
        obj = _compute_objective(f_init)
        if obj < best_obj:
            best_f = f_init.copy()
            best_obj = obj
    
    # Local search from the best candidate
    f_opt, obj_opt = _refine_by_flipping(best_f, deadline, max_iters=50000)
    if obj_opt < best_obj:
        best_f = f_opt
        best_obj = obj_opt
    
    # Additional refinement passes if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        f_polish = best_f.copy()
        obj_polish = best_obj
        max_polish_iters = int(remaining_time * 300)  # ~300 iters per second
        for _ in range(min(max_polish_iters, 20000)):
            if time.time() > deadline - 1.0:
                break
            idx = np.random.randint(N)
            trial = f_polish.copy()
            trial[idx] = -trial[idx]
            obj = _compute_objective(trial)
            if obj < obj_polish:
                f_polish = trial
                obj_polish = obj
        if obj_polish < best_obj:
            best_f = f_polish
            best_obj = obj_polish
    
    return best_f
# EVOLVE-BLOCK-END
