"""Search for a low-autocorrelation signed step function (400 equal intervals).

Uses incremental autocorrelation updates to evaluate flips in O(N) instead of
O(N log N), enabling far more search iterations within the time budget.
Combines simulated annealing with block flips, greedy coordinate descent
(single, pair, triple flips), and a deterministic Halton-based initial seed.
"""
import numpy as np
import time
import math

N = 400
_HARD_DEADLINE = 135.0  # 15 s safety margin below 150 s kill


def _halton_initial(seed_base=2):
    """Generate a deterministic +-1 sequence using Halton-like construction.
    
    Uses fractional parts of n * log(2) to create quasi-random sign assignments
    that avoid periodicity and provide a good starting point for optimization.
    """
    h = np.zeros(N)
    for n in range(1, N + 1):
        # Use multiple bases for better distribution
        frac = (n * 0.6180339887498949) % 1.0  # golden ratio conjugate
        frac2 = (n * 0.7548776662466927) % 1.0  # another irrational
        combined = (frac + frac2) % 1.0
        h[n - 1] = 1.0 if combined < 0.5 else -1.0
    return h


def _init_autocorr(h):
    """Compute full aperiodic autocorrelation r[k] for k in [-(N-1), N-1].
    
    r[k] = sum_{i} h[i] * h[i+k] for k >= 0
    r[k] = r[-k] for k < 0 (by symmetry for real sequences)
    
    We store r[0..N-1] where r[k] = sum_i h[i]*h[i+k] for k>=0.
    """
    r = np.zeros(N)
    for k in range(N):
        r[k] = np.dot(h[:N - k], h[k:])
    return r


def _peak_from_autocorr(r):
    """Peak sidelobe magnitude from autocorrelation array r[0..N-1]."""
    if N <= 1:
        return 0.0
    return float(np.max(np.abs(r[1:])))


def _flip_update_autocorr(r, h, i):
    """Update autocorrelation r after flipping h[i]. Returns new r (in-place).
    
    When h[i] flips from v to -v:
    For k >= 0: r[k] = sum_j h[j]*h[j+k]
    The term involving h[i] appears when j = i (gives h[i]*h[i+k]) 
    or when j+k = i i.e. j = i-k (gives h[i-k]*h[i])
    
    So flipping h[i] by factor -1 changes:
    - For k >= 0: the term h[i]*h[i+k] changes by -2*h[i]*h[i+k] (if i+k < N)
    - For k <= 0 (i.e., r[k] where k represents negative lag): 
      r[k] for k>0 includes h[i-k]*h[i] when i-k >= 0, changes by -2*h[i-k]*h[i]
    
    Actually, let's think more carefully. r[k] = sum_{j=0}^{N-1-k} h[j]*h[j+k] for k >= 0.
    
    Flipping h[i]: h[i] -> -h[i]
    
    For r[k] with k >= 0:
    - If i <= N-1-k: term h[i]*h[i+k] is present, changes by -2*h[i]*h[i+k]
    - If i-k >= 0 and i-k <= N-1-k: term h[i-k]*h[i] is present (when j=i-k), 
      changes by -2*h[i-k]*h[i]
    
    Note these are the same term when we consider that r[k] = sum h[j]h[j+k],
    the index i contributes to r[k] if:
    - j = i, so i + k < N, contributing h[i]*h[i+k]
    - j + k = i, so j = i - k >= 0, contributing h[i-k]*h[i]
    
    Both conditions can hold simultaneously (when 0 <= i-k and i+k < N).
    
    For r[k] with k >= 0, flipping h[i] changes r[k] by:
    delta = -2 * h[i] * (h[i+k] if i+k < N else 0) - 2 * h[i] * (h[i-k] if i-k >= 0 else 0)
    
    But wait - we need to be careful. The autocorrelation is r[k] = sum_j h[j]*h[j+k].
    After flipping h[i], the new r'[k] = r[k] - 2*h[i]*h[i+k] (if i+k<N) - 2*h[i]*h[i-k] (if i-k>=0)
    
    Actually no. Let's reconsider. h[i] appears in r[k] in two ways:
    1. As the first factor: h[i]*h[i+k] when i+k < N
    2. As the second factor: h[i-k]*h[i] when i-k >= 0
    
    When we flip h[i] to -h[i]:
    1. h[i]*h[i+k] becomes -h[i]*h[i+k], change = -2*h[i]*h[i+k]
    2. h[i-k]*h[i] becomes h[i-k]*(-h[i]), change = -2*h[i-k]*h[i]
    
    So total change to r[k] = -2*h[i]*(h[i+k] + h[i-k]) with appropriate bounds.
    
    But note: for k=0, both terms are h[i]*h[i] = h[i]^2 = 1, so change = -2*h[i]*(1+1) = -4*h[i]
    Wait, that's not right either. For k=0: r[0] = sum h[j]^2 = N. Flipping h[i] doesn't change r[0] since h[i]^2 is still 1.
    
    Let me redo this. For k=0: r[0] = sum_j h[j]*h[j] = N. Flipping h[i] changes h[i]^2 from 1 to 1, no change. So delta for k=0 is 0.
    
    For k > 0: 
    - Term 1: h[i]*h[i+k] exists if i+k < N. Change: -2*h[i]*h[i+k]
    - Term 2: h[i-k]*h[i] exists if i-k >= 0. Change: -2*h[i-k]*h[i]
    
    But wait, when k>0, these are distinct terms in the sum (different j values: j=i and j=i-k).
    
    So for k > 0: delta_r_k = -2*h[i]*(h[i+k] if i+k<N else 0) - 2*h[i]*(h[i-k] if i-k>=0 else 0)
    
    Hmm, but this doesn't seem right dimensionally. Let me verify with a small example.
    
    h = [1, -1, 1, -1], N=4
    r[0] = 1+1+1+1 = 4
    r[1] = 1*(-1) + (-1)*1 + 1*(-1) = -1-1-1 = -3
    r[2] = 1*1 + (-1)*(-1) = 1+1 = 2
    r[3] = 1*(-1) = -1
    
    Flip h[1] (index 1, value -1 -> 1):
    new h = [1, 1, 1, -1]
    r'[0] = 4
    r'[1] = 1*1 + 1*1 + 1*(-1) = 1+1-1 = 1
    r'[2] = 1*1 + 1*(-1) = 0
    r'[3] = 1*(-1) = -1
    
    Change in r[1]: 1 - (-3) = 4
    Using formula: -2*h[1]*(h[2] + h[0]) = -2*(-1)*(1+1) = 4. Correct!
    
    Change in r[2]: 0 - 2 = -2
    Using formula: -2*h[1]*(h[3] + h[0]) = -2*(-1)*(-1+1) = 0. Wrong!
    
    Let me recheck. For k=2, i=1:
    - h[i+k] = h[3] = -1, exists since 3 < 4
    - h[i-k] = h[-1], doesn't exist since -1 < 0
    
    So delta = -2*(-1)*(-1) - 0 = -2. Correct!
    
    I made an arithmetic error. h[i-k] = h[1-2] = h[-1] doesn't exist.
    
    OK so the formula is:
    For k > 0:
      delta = 0
      if i + k < N: delta += -2 * h[i] * h[i+k]
      if i - k >= 0: delta += -2 * h[i] * h[i-k]
    
    For k = 0: delta = 0
    
    This is O(1) per k value, so O(N) total to update all r[k].
    """
    v = h[i]
    # For each k >= 1, update r[k]
    for k in range(1, N):
        delta = 0.0
        if i + k < N:
            delta -= 2.0 * v * h[i + k]
        if i - k >= 0:
            delta -= 2.0 * v * h[i - k]
        if delta != 0.0:
            r[k] += delta
    return r


def _flip_update_autocorr_fast(r, h, i):
    """Vectorized version of autocorrelation update for flipping h[i]."""
    v = h[i]
    # For k = 1 to N-1:
    # delta_k = -2*v*h[i+k] (if i+k < N) - 2*v*h[i-k] (if i-k >= 0)
    
    # Create array of deltas for all k
    k_vals = np.arange(1, N)
    
    # h[i+k] for valid i+k < N
    idx_pos = i + k_vals  # these are the indices for h[i+k]
    valid_pos = idx_pos < N
    h_pos = np.zeros(N - 1)
    h_pos[valid_pos] = h[idx_pos[valid_pos]]
    
    # h[i-k] for valid i-k >= 0
    idx_neg = i - k_vals
    valid_neg = idx_neg >= 0
    h_neg = np.zeros(N - 1)
    h_neg[valid_neg] = h[idx_neg[valid_neg]]
    
    deltas = -2.0 * v * (h_pos + h_neg)
    r[1:] += deltas
    return r


def _eval_flip(r, h, i):
    """Evaluate peak sidelobe after flipping h[i], without modifying r or h.
    Returns (new_peak, delta_array) where delta_array can be used to update r.
    """
    v = h[i]
    k_vals = np.arange(1, N)
    idx_pos = i + k_vals
    valid_pos = idx_pos < N
    h_pos = np.zeros(N - 1)
    h_pos[valid_pos] = h[idx_pos[valid_pos]]
    
    idx_neg = i - k_vals
    valid_neg = idx_neg >= 0
    h_neg = np.zeros(N - 1)
    h_neg[valid_neg] = h[idx_neg[valid_neg]]
    
    deltas = -2.0 * v * (h_pos + h_neg)
    new_r_sidelobes = r[1:] + deltas
    new_peak = float(np.max(np.abs(new_r_sidelobes)))
    return new_peak, deltas


def _anneal(seed, time_budget):
    """Simulated annealing with fast incremental autocorrelation updates."""
    rng = np.random.default_rng(seed)
    h = rng.choice([-1.0, 1.0], size=N)
    r = _init_autocorr(h)
    cur = _peak_from_autocorr(r)
    best_h = h.copy()
    best = cur
    T = 3.0
    T_min = 1e-8
    start = time.time()
    flip_count = 0
    
    while time.time() - start < time_budget:
        progress = min(1.0, (time.time() - start) / max(time_budget, 1e-9))
        max_len = max(1, int(20.0 * (1.0 - 0.6 * progress)))
        L = int(rng.integers(1, max_len + 1))
        i = int(rng.integers(0, N - L + 1))
        
        # Flip block h[i:i+L]
        # We need to update r for each flip in the block
        # For efficiency, do all flips and recompute r once
        h[i:i + L] *= -1.0
        r = _init_autocorr(h)
        new_peak = _peak_from_autocorr(r)
        
        d = new_peak - cur
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur = new_peak
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            h[i:i + L] *= -1.0
            r = _init_autocorr(h)
        
        T = max(T_min, T * 0.99995)
        flip_count += 1
    
    return best_h, best, flip_count


def _anneal_fast_single(seed, time_budget):
    """Annealing with single-bit flips using O(N) incremental updates."""
    rng = np.random.default_rng(seed)
    h = rng.choice([-1.0, 1.0], size=N)
    r = _init_autocorr(h)
    cur = _peak_from_autocorr(r)
    best_h = h.copy()
    best = cur
    T = 2.0
    T_min = 1e-8
    start = time.time()
    flip_count = 0
    
    while time.time() - start < time_budget:
        i = int(rng.integers(0, N))
        
        # Evaluate flip at i
        new_peak, deltas = _eval_flip(r, h, i)
        
        d = new_peak - cur
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            # Accept: update h and r
            h[i] *= -1.0
            r[1:] += deltas
            cur = new_peak
            if cur < best:
                best = cur
                best_h = h.copy()
        # If rejected, h and r unchanged
        
        T = max(T_min, T * 0.99998)
        flip_count += 1
    
    return best_h, best, flip_count


def _greedy_polish_fast(h, time_budget):
    """Greedy polish with single-flip and pair-flip using fast evaluation."""
    h = h.copy()
    r = _init_autocorr(h)
    cur = _peak_from_autocorr(r)
    best = cur
    best_h = h.copy()
    start = time.time()
    
    while time.time() - start < time_budget:
        improved = False
        
        # --- Single-flip sweep ---
        best_i = -1
        best_c = cur
        for i in range(N):
            new_peak, deltas = _eval_flip(r, h, i)
            if new_peak < best_c:
                best_c = new_peak
                best_i = i
                best_deltas = deltas
        if best_i >= 0:
            h[best_i] *= -1.0
            r[1:] += best_deltas
            cur = best_c
            if cur < best:
                best = cur
                best_h = h.copy()
            improved = True
        
        # --- Adjacent-pair-flip sweep ---
        best_pair = -1
        best_c2 = cur
        best_pair_deltas = None
        for i in range(N - 1):
            # Evaluate flipping both h[i] and h[i+1]
            # First flip h[i]
            peak1, d1 = _eval_flip(r, h, i)
            # Then flip h[i+1] on the modified state
            # This is tricky; let's just recompute for pairs
            h[i] *= -1.0
            r[1:] += d1
            peak2, d2 = _eval_flip(r, h, i + 1)
            h[i] *= -1.0
            r[1:] -= d1
            
            if peak2 < best_c2:
                best_c2 = peak2
                best_pair = i
                # Store the combined effect
                # We'll recompute this when applying
                best_pair_d1 = d1.copy()
                best_pair_d2 = d2.copy()
        
        if best_pair >= 0:
            # Apply the pair flip
            h[best_pair] *= -1.0
            r[1:] += best_pair_d1
            # Now flip h[best_pair+1]
            _, d2_apply = _eval_flip(r, h, best_pair + 1)
            h[best_pair + 1] *= -1.0
            r[1:] += d2_apply
            cur = best_c2
            if cur < best:
                best = cur
                best_h = h.copy()
            improved = True
        
        if not improved:
            break
    
    return best_h, best


def _greedy_single_only_fast(h, time_budget):
    """Tighter single-flip polish using fast evaluation."""
    h = h.copy()
    r = _init_autocorr(h)
    cur = _peak_from_autocorr(r)
    best = cur
    best_h = h.copy()
    start = time.time()
    
    while time.time() - start < time_budget:
        best_i = -1
        best_c = cur
        best_deltas = None
        for i in range(N):
            new_peak, deltas = _eval_flip(r, h, i)
            if new_peak < best_c:
                best_c = new_peak
                best_i = i
                best_deltas = deltas
        if best_i < 0:
            break
        h[best_i] *= -1.0
        r[1:] += best_deltas
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()
    
    return best_h, best


def _greedy_triple_fast(h, time_budget):
    """Greedy with single and triple flips using fast evaluation."""
    h = h.copy()
    r = _init_autocorr(h)
    cur = _peak_from_autocorr(r)
    best = cur
    best_h = h.copy()
    start = time.time()
    
    while time.time() - start < time_budget:
        improved = False
        
        # Single-flip sweep
        best_i = -1
        best_c = cur
        best_deltas = None
        for i in range(N):
            new_peak, deltas = _eval_flip(r, h, i)
            if new_peak < best_c:
                best_c = new_peak
                best_i = i
                best_deltas = deltas
        if best_i >= 0:
            h[best_i] *= -1.0
            r[1:] += best_deltas
            cur = best_c
            if cur < best:
                best = cur
                best_h = h.copy()
            improved = True
        
        # Triple-flip sweep (flip 3 consecutive bits)
        best_triple = -1
        best_c3 = cur
        for i in range(N - 2):
            # Evaluate flipping h[i], h[i+1], h[i+2]
            # Apply sequentially
            p1, d1 = _eval_flip(r, h, i)
            h[i] *= -1.0
            r[1:] += d1
            p2, d2 = _eval_flip(r, h, i + 1)
            h[i + 1] *= -1.0
            r[1:] += d2
            p3, d3 = _eval_flip(r, h, i + 2)
            
            # Undo
            h[i + 2] *= -1.0  # no change needed since we didn't apply
            r[1:] -= d2
            h[i + 1] *= -1.0
            r[1:] -= d1
            h[i] *= -1.0
            
            if p3 < best_c3:
                best_c3 = p3
                best_triple = i
                # Store deltas for application
                bt_d1 = d1.copy()
                bt_d2 = d2.copy()
                bt_d3 = d3.copy()
        
        if best_triple >= 0:
            # Apply triple flip
            h[best_triple] *= -1.0
            r[1:] += bt_d1
            h[best_triple + 1] *= -1.0
            r[1:] += bt_d2
            h[best_triple + 2] *= -1.0
            r[1:] += bt_d3
            cur = best_c3
            if cur < best:
                best = cur
                best_h = h.copy()
            improved = True
        
        if not improved:
            break
    
    return best_h, best


def _verify_peak(h):
    """Verify peak sidelobe using full FFT computation (for final check)."""
    FFT_SIZE = 1
    while FFT_SIZE < 2 * N:
        FFT_SIZE *= 2
    H = np.fft.rfft(h, FFT_SIZE)
    ac = np.fft.irfft(H * H.conj(), FFT_SIZE, n=N)
    return float(np.max(np.abs(ac[1:])))


def construct():
    try:
        start = time.time()

        def _remaining():
            return _HARD_DEADLINE - (time.time() - start)

        # Initialize with Halton-based construction for a good deterministic start
        global_best_h = _halton_initial()
        global_best = _verify_peak(global_best_h)

        # Phase 1: Fast single-flip annealing (most iterations per second)
        budget = min(30.0, _remaining() - 15.0)
        if budget > 2.0:
            h, c, fc = _anneal_fast_single(42, budget)
            if c < global_best:
                global_best = c
                global_best_h = h

        # Phase 2: Block-flip annealing for larger moves
        budget2 = min(20.0, _remaining() - 15.0)
        if budget2 > 2.0:
            h, c, fc = _anneal(12345, budget2)
            if c < global_best:
                global_best = c
                global_best_h = h

        # Phase 3: Greedy polish with single and pair flips
        remaining = _remaining() - 12.0
        if remaining > 1.0:
            h, c = _greedy_polish_fast(global_best_h, remaining)
            if c < global_best:
                global_best = c
                global_best_h = h

        # Phase 4: Triple-flip polish
        remaining2 = _remaining() - 8.0
        if remaining2 > 1.0:
            h, c = _greedy_triple_fast(global_best_h, remaining2)
            if c < global_best:
                global_best = c
                global_best_h = h

        # Phase 5: Final tight single-flip polish
        remaining3 = _remaining() - 3.0
        if remaining3 > 0.5:
            h, c = _greedy_single_only_fast(global_best_h, remaining3)
            if c < global_best:
                global_best = c
                global_best_h = h

        # Phase 6: One more round of polish if time allows
        remaining4 = _remaining() - 1.5
        if remaining4 > 0.5:
            h, c = _greedy_polish_fast(global_best_h, remaining4)
            if c < global_best:
                global_best = c
                global_best_h = h

        # Final verification with FFT
        final_peak = _verify_peak(global_best_h)
        # Use the FFT-verified value as the true score
        return np.ascontiguousarray(global_best_h, dtype=np.float64)

    except Exception:
        return np.ones(N)
