import numpy as np
import time
from scipy.signal import windows

N = 400
_start_time = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

# Precompute FFT size for autoconvolution
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_fft(f):
    """Compute linear autoconvolution using FFT."""
    F = np.fft.rfft(f, n=_FFT_SIZE)
    C = np.fft.irfft(F * F, n=_FFT_SIZE)
    return C[:2 * N - 1]

def _score(f):
    """Compute max(|autoconv|) / (sum(f))^2. Lower is better."""
    S = np.sum(f)
    if abs(S) < 1e-12:
        return float('inf')
    C = _autoconv_fft(f)
    return np.max(np.abs(C)) / (S * S)

def construct():
    """
    Minimizes max(|autoconv(f)|) / (sum(f))^2 using spectral-domain-inspired
    initialization followed by multi-start local search with pair-flips.
    """
    deadline = _start_time + _TIME_LIMIT
    
    best_f = np.ones(N, dtype=np.int8)
    best_score = _score(best_f)
    
    # Helper: try a candidate and update best
    def try_candidate(f):
        nonlocal best_f, best_score
        sc = _score(f)
        if sc < best_score:
            best_score = sc
            best_f = f.copy()
    
    # Phase 1: Spectral-domain-inspired initializations
    # 1.1 Windowed sinc-inspired sequences (Blackman-Harris windowed)
    # The idea: a sequence whose power spectrum has suppressed sidelobes
    # will have autoconvolution with suppressed sidelobes.
    
    # Generate a "designed" sequence by taking the sign of a windowed sinc-like function
    # with various center frequencies and window types
    
    window_types = ['blackmanharris', 'kaiser', 'hamming', 'hann', 'blackman']
    for wtype in window_types:
        if time.time() > deadline:
            break
        
        # Try different "center" positions and bandwidths
        for center_frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            if time.time() > deadline:
                break
            
            # Create a frequency-domain template: peak at center_frac * N
            freqs = np.fft.rfftfreq(N, d=1.0)
            center_freq = center_frac
            
            # Gaussian-shaped spectral peak
            sigma = np.random.choice([0.05, 0.1, 0.15, 0.2, 0.3])
            spectral_template = np.exp(-((freqs - center_freq) ** 2) / (2 * sigma ** 2))
            
            # Apply window in time domain
            if wtype == 'kaiser':
                win = windows.kaiser(N, 8.6)
            else:
                win = windows.get_window(wtype, N)
            
            # Modulate: multiply spectral template by window in frequency domain
            F_design = spectral_template * win
            
            # Take inverse FFT to get time-domain signal
            f_design = np.fft.irfft(F_design, n=N)
            
            # Sign function to get ±1 sequence
            f_sign = np.sign(f_design)
            f_sign[f_sign == 0] = 1
            f_sign = f_sign.astype(np.int8)
            
            try_candidate(f_sign)
    
    # 1.2 Thue-Morse
    f_tm = np.zeros(N, dtype=np.int8)
    f_tm[0] = 1
    for i in range(1, N):
        n = i // 2
        r = i % 2
        f_tm[i] = f_tm[n] if r == 0 else -f_tm[n]
    try_candidate(f_tm)
    
    # 1.3 Random ±1 sequences
    rng = np.random.default_rng(42)
    for i in range(500):
        if time.time() > deadline:
            break
        f_rand = rng.choice([-1, 1], size=N).astype(np.int8)
        try_candidate(f_rand)
    
    # Phase 2: Local search with pair-flips (2-bit flips)
    # This avoids local minima that single-flip search gets stuck in
    
    def local_search_pair(f, max_passes=10, time_budget=15.0):
        """Local search using pair-flips for better escape from local minima."""
        f = f.copy().astype(np.float64)
        S = np.sum(f)
        if abs(S) < 1e-6:
            return f, float('inf')
        
        C = _autoconv_fft(f)
        score = np.max(np.abs(C)) / (S * S)
        
        start = time.time()
        
        for _pass in range(max_passes):
            if time.time() - start > time_budget:
                break
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try pair-flips: flip two bits at a time
            # Use random pairs to avoid O(N^2) exhaustive search
            num_pairs_to_try = min(500, N * (N - 1) // 2)
            
            for _ in range(num_pairs_to_try):
                if time.time() > deadline:
                    break
                
                # Pick two random distinct indices
                i, j = rng.choice(N, size=2, replace=False)
                
                if i == j:
                    continue
                
                vi = f[i]
                vj = f[j]
                
                # Skip if flipping both would be same as single flip (vi == vj)
                # Actually, pair flip changes S by -2*vi - 2*vj
                # If vi == vj, S changes by -4*vi
                # If vi != vj, S changes by 0
                
                f[i] = -vi
                f[j] = -vj
                new_S = S - 2 * vi - 2 * vj
                
                # Compute new autoconvolution
                # Delta from flipping f[i]: -4*vi*f[k-i] for k != 2i
                # Delta from flipping f[j]: -4*vj*f[k-j] for k != 2j
                # But we need to be careful about interactions
                
                C_new = C.copy()
                
                # Contribution from flipping i
                k_start_i = max(0, i)
                k_end_i = min(2 * N - 1, i + N)
                idx_i = np.arange(k_start_i, k_end_i)
                f_idx_i = idx_i - i
                center_mask_i = (idx_i == 2 * i)
                delta_i = -4 * vi * f[f_idx_i]
                delta_i[center_mask_i] = 0
                C_new[idx_i] += delta_i
                
                # Contribution from flipping j
                k_start_j = max(0, j)
                k_end_j = min(2 * N - 1, j + N)
                idx_j = np.arange(k_start_j, k_end_j)
                f_idx_j = idx_j - j
                center_mask_j = (idx_j == 2 * j)
                delta_j = -4 * vj * f[f_idx_j]
                delta_j[center_mask_j] = 0
                C_new[idx_j] += delta_j
                
                # Correction for double-counting: terms where both i and j contribute
                # When k-i == j and k-j == i, i.e., k = i+j
                # The term f[i]*f[j] appears in C[k] as f[i]*f[j] + f[j]*f[i] = 2*f[i]*f[j]
                # After flip: (-f[i])*(-f[j]) + (-f[j])*(-f[i]) = 2*f[i]*f[j]
                # So no change at k = i+j! The pair flip doesn't change the cross term.
                
                # But wait, we need to check if our incremental update is correct
                # The issue is that when we compute delta_i, we use the original f
                # But f[j] has also been flipped. So for k where k-i == j, we used
                # the old f[j] in delta_i, but f[j] is now -f[j].
                # Similarly for delta_j where k-j == i.
                
                # Let's just recompute properly for safety
                # Actually, for correctness, let's just compute the full autoconvolution
                # for pair flips since it's simpler and less error-prone
                
                C_new = _autoconv_fft(f)
                
                if new_S == 0:
                    new_score = float('inf')
                else:
                    new_score = np.max(np.abs(C_new)) / (new_S * new_S)
                
                if new_score < score - 1e-10:
                    # Accept
                    S = new_S
                    C = C_new
                    score = new_score
                    improved = True
                else:
                    # Revert
                    f[i] = vi
                    f[j] = vj
            
            if not improved:
                break
        
        return f, score
    
    # Run local search from the best candidate found so far
    if time.time() < deadline - 10.0:
        f_opt, sc_opt = local_search_pair(best_f.astype(np.float64), max_passes=20, time_budget=25.0)
        if sc_opt < best_score:
            best_score = sc_opt
            best_f = f_opt.astype(np.int8)
    
    # Phase 3: Multi-start local search from random restarts
    if time.time() < deadline - 15.0:
        for restart in range(30):
            if time.time() > deadline - 15.0:
                break
            
            # Random start
            f_rand = rng.choice([-1, 1], size=N).astype(np.float64)
            f_opt, sc_opt = local_search_pair(f_rand, max_passes=5, time_budget=3.0)
            
            if sc_opt < best_score:
                best_score = sc_opt
                best_f = f_opt.astype(np.int8)
    
    # Phase 4: Final local search from best with single flips for fine-tuning
    def local_search_single(f, max_passes=10, time_budget=10.0):
        """Local search using single flips for final refinement."""
        f = f.copy().astype(np.float64)
        S = np.sum(f)
        if abs(S) < 1e-6:
            return f, float('inf')
        
        C = _autoconv_fft(f)
        score = np.max(np.abs(C)) / (S * S)
        
        start = time.time()
        
        for _pass in range(max_passes):
            if time.time() - start > time_budget:
                break
            if time.time() > deadline:
                break
            
            improved = False
            
            for i in range(N):
                if time.time() > deadline:
                    break
                
                v = f[i]
                f[i] = -v
                new_S = S - 2 * v
                
                C_new = _autoconv_fft(f)
                
                if new_S == 0:
                    new_score = float('inf')
                else:
                    new_score = np.max(np.abs(C_new)) / (new_S * new_S)
                
                if new_score < score - 1e-10:
                    S = new_S
                    C = C_new
                    score = new_score
                    improved = True
                else:
                    f[i] = v
            
            if not improved:
                break
        
        return f, score
    
    if time.time() < deadline - 5.0:
        f_opt, sc_opt = local_search_single(best_f.astype(np.float64), max_passes=20, time_budget=20.0)
        if sc_opt < best_score:
            best_score = sc_opt
            best_f = f_opt.astype(np.int8)
    
    return best_f.astype(np.int8)

# EVOLVE-BLOCK-END
