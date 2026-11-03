import numpy as np
import time

N = 400
_NFFT = 1 << (2 * N - 1).bit_length()  # next power of 2 >= 2N for fast FFT


def _autocorr_fast(h):
    # linear one-sided autocorrelation ac[k] = sum_i h[i] h[i+k], k = 0..N-1
    # Use real FFT trick: autocorr of real signal = IFFT(|FFT|^2)
    F = np.fft.rfft(h, n=_NFFT)
    ac = np.fft.irfft(F * np.conj(F), n=_NFFT)[:N]
    return ac


def _autocorr(h):
    return _autocorr_fast(h)


def _hard_score(h):
    # the actual metric: peak autoconvolution concentration (lower is better)
    ac = _autocorr(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    return float(np.max(np.abs(ac)) / dc)


def _smooth_score(h, beta):
    # smooth (log-sum-exp) surrogate of the peak sidelobe energy, guides SA
    ac = _autocorr(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    y = (ac[1:] / dc) ** 2
    m = float(np.max(y))
    z = beta * (y - m)
    return m + float(np.log(np.sum(np.exp(z)))) / beta


def _hard_score_approx(h):
    """Faster approximate score: uses a subset of lags to estimate the peak."""
    ac = _autocorr_fast(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    # Check all lags but use a quick max
    return float(np.max(np.abs(ac)) / dc)


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    t = [1]
    while len(t) < n:
        t += [-x for x in t]
    return np.array(t[:n])


def _analytic_flip_estimate(h, ac, i):
    """
    Estimate the new autocorrelation after flipping bit i without full FFT.
    Flipping h[i] changes ac[k] for all k.
    
    For lag k:
    - If k=0: ac[0] unchanged (h[i]^2 = 1)
    - If 0 < k < N: ac[k] changes by -2*h[i]*h[i+k] if i+k < N, and -2*h[i]*h[i-k] if i-k >= 0
    
    More precisely:
    ac_new[k] = ac_old[k] - 2*h[i]*h[i+k] (for i+k < N)
              ac_new[k] = ac_old[k] - 2*h[i]*h[i-k] (for i-k >= 0, k>0)
    
    Actually, let's think carefully:
    ac[k] = sum_{j} h[j] * h[j+k]
    
    After flipping h[i] to -h[i]:
    ac_new[k] = sum_j h'[j] * h'[j+k]
    
    The only terms that change are those where j=i or j+k=i, i.e., j=i or j=i-k.
    
    So:
    ac_new[k] = ac_old[k] - 2*h[i]*h[i+k] (if i+k < N and i+k >= 0)
              ac_new[k] = ac_old[k] - 2*h[i]*h[i-k] (if i-k >= 0 and i-k < N)
    
    But we need to be careful not to double-count when both conditions hold.
    Actually, for a given k, the term h[i]*h[i+k] appears in ac[k] if i+k < N.
    The term h[i-k]*h[i] appears in ac[k] if i-k >= 0.
    
    So the change is:
    delta_ac[k] = -2*h[i]*h[i+k] (if i+k < N) - 2*h[i-k]*h[i] (if i-k >= 0)
    
    Simplify: delta_ac[k] = -2*h[i]*(h[i+k] + h[i-k]) where we treat out-of-bounds as 0.
    """
    ac_new = ac.copy()
    hi = h[i]
    
    # For each lag k, compute the change
    # We can vectorize this
    k_vals = np.arange(N)
    
    # Terms where i+k is valid
    valid_plus = (i + k_vals) < N
    h_plus = np.where(valid_plus, h[i + k_vals[valid_plus]], 0.0)
    
    # Terms where i-k is valid
    valid_minus = (i - k_vals) >= 0
    h_minus = np.where(valid_minus, h[i - k_vals[valid_minus]], 0.0)
    
    delta = -2.0 * hi * (h_plus + h_minus)
    ac_new += delta
    
    # Note: ac[0] should be unchanged since h[i]^2 = 1, but let's verify
    # Actually, for k=0: delta = -2*h[i]*(h[i] + h[i]) = -4*h[i]^2 = -4
    # That's wrong! Let me reconsider.
    
    # For k=0: ac[0] = sum_j h[j]^2 = N (always)
    # After flip, ac_new[0] = N still
    # So delta_ac[0] = 0
    
    # Let me redo this more carefully.
    # ac[k] = sum_{j=0}^{N-1-k} h[j] * h[j+k]
    
    # After flipping h[i]:
    # The term h[j]*h[j+k] changes if j=i or j+k=i, i.e., j=i or j=i-k.
    
    # So:
    # ac_new[k] = ac_old[k] - h[i]*h[i+k] - h[i-k]*h[i] + (-h[i])*(-h[i+k]) + (-h[i-k])*(-h[i])
    #           = ac_old[k] - h[i]*h[i+k] - h[i-k]*h[i] + h[i]*h[i+k] + h[i-k]*h[i]
    # Wait, that's not right either.
    
    # Let's be very explicit.
    # Original: term1 = h[i]*h[i+k] (if i+k < N), term2 = h[i-k]*h[i] (if i-k >= 0)
    # New: term1' = (-h[i])*h[i+k] = -h[i]*h[i+k] (if i+k < N)
    #      term2' = h[i-k]*(-h[i]) = -h[i-k]*h[i] (if i-k >= 0)
    
    # So delta = term1' - term1 + term2' - term2
    #          = -2*h[i]*h[i+k] (if i+k < N) - 2*h[i-k]*h[i] (if i-k >= 0)
    
    # For k=0: i+k = i, i-k = i, so delta = -2*h[i]*h[i] - 2*h[i]*h[i] = -4
    # But ac[0] should be invariant!
    
    # I see the issue: for k=0, the same term h[i]*h[i] is counted once, not twice.
    # So for k=0, delta = -2*h[i]*h[i] = -2, which is also wrong.
    
    # Let me think again.
    # ac[0] = sum_j h[j]^2
    # After flipping h[i], ac_new[0] = sum_j h'[j]^2 = sum_j h[j]^2 = ac[0]
    # So delta_ac[0] = 0 always.
    
    # For k > 0:
    # ac[k] = sum_{j=0}^{N-1-k} h[j] * h[j+k]
    # The term involving h[i] appears if:
    #   j = i and i+k < N: term = h[i]*h[i+k]
    #   j = i-k and i-k >= 0: term = h[i-k]*h[i]
    
    # After flip:
    #   If i+k < N: term becomes (-h[i])*h[i+k] = -h[i]*h[i+k]
    #   If i-k >= 0: term becomes h[i-k]*(-h[i]) = -h[i-k]*h[i]
    
    # So delta_ac[k] = -2*h[i]*h[i+k] (if i+k < N) - 2*h[i-k]*h[i] (if i-k >= 0)
    
    # For k=0, this formula gives -4, but we know delta should be 0.
    # The issue is that for k=0, both conditions refer to the same term.
    
    # So the correct formula is:
    # For k=0: delta = 0
    # For k>0: delta = -2*h[i]*h[i+k] (if i+k < N) - 2*h[i-k]*h[i] (if i-k >= 0)
    
    # Let's implement this correctly.
    ac_new = ac.copy()
    if i < N:
        hi = h[i]
        # For k > 0
        k_vals = np.arange(1, N)
        
        # i+k valid
        valid_plus = (i + k_vals) < N
        h_plus = np.zeros(N - 1)
        h_plus[valid_plus] = h[i + k_vals[valid_plus]]
        
        # i-k valid
        valid_minus = (i - k_vals) >= 0
        h_minus = np.zeros(N - 1)
        h_minus[valid_minus] = h[i - k_vals[valid_minus]]
        
        delta = -2.0 * hi * (h_plus + h_minus)
        ac_new[1:] += delta
    
    return ac_new

def construct():
    try:
        t0 = time.time()
        deadline = t0 + 115.0          # safety margin inside the 120s limit
        rng = np.random.default_rng(20240517)

        # ---------- diverse initializations ----------
        idx = np.arange(N)
        inits = [
            _thue_morse(N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.cos(2 * np.pi * idx ** 2 / (2 * N)),
            np.cos(2 * np.pi * idx ** 2 / (N + 1)),
            np.cos(2 * np.pi * idx ** 2 / (3 * N)),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.cos(2 * np.pi * idx / N),
            np.sin(2 * np.pi * idx / N),
            np.cos(2 * np.pi * 2 * idx / N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
        ]
        best_h = None
        best_hard = 1e9
        for h in inits:
            s = _hard_score(h)
            if s < best_hard:
                best_hard = s
                best_h = h.copy()

        h = best_h.copy()
        cur_hard = best_hard
        ac = _autocorr(h)
        dc = ac[0]

        # ---------- Phase 1: Single-bit greedy search with more candidates ----------
        max_iterations_phase1 = 50000
        for _iter in range(max_iterations_phase1):
            if time.time() > deadline - 30.0:
                break
            
            # Recompute autocorrelation periodically to avoid drift
            if _iter % 50 == 0:
                ac = _autocorr(h)
                dc = ac[0]
                if dc <= 1e-30:
                    break
                cur_hard = float(np.max(np.abs(ac)) / dc)
            
            # Find the lag with the maximum absolute sidelobe
            sidelobes = np.abs(ac[1:])
            k_star = int(np.argmax(sidelobes)) + 1
            c_star = ac[k_star]
            
            # Determine which flips would reduce |C[k_star]|
            max_i = N - k_star
            indices = np.arange(max_i)
            products = h[indices] * h[indices + k_star]
            
            # We want products to have the same sign as c_star
            if c_star > 0:
                candidates = indices[products > 0]
            else:
                candidates = indices[products < 0]
            
            if len(candidates) == 0:
                # Try a random flip
                i = int(rng.integers(0, N))
                ac_new = _analytic_flip_estimate(h, ac, i)
                new_hard = float(np.max(np.abs(ac_new)) / ac_new[0]) if ac_new[0] > 1e-30 else 1e9
                if new_hard < cur_hard:
                    h[i] *= -1.0
                    ac = ac_new
                    cur_hard = new_hard
                    if cur_hard < best_hard:
                        best_hard = cur_hard
                        best_h = h.copy()
                continue

            # Use analytic estimate to evaluate candidates without full FFT
            # Test more candidates: up to 50
            num_tests = min(50, len(candidates))
            if len(candidates) > num_tests:
                test_indices = rng.choice(candidates, num_tests, replace=False)
            else:
                test_indices = candidates
            
            best_candidate = None
            best_candidate_score = cur_hard
            best_candidate_ac = None
            
            for i in test_indices:
                ac_new = _analytic_flip_estimate(h, ac, int(i))
                new_hard = float(np.max(np.abs(ac_new)) / ac_new[0]) if ac_new[0] > 1e-30 else 1e9
                if new_hard < best_candidate_score:
                    best_candidate_score = new_hard
                    best_candidate = int(i)
                    best_candidate_ac = ac_new
            
            if best_candidate is not None and best_candidate_score < cur_hard:
                h[best_candidate] *= -1.0
                ac = best_candidate_ac
                cur_hard = best_candidate_score
                if cur_hard < best_hard:
                    best_hard = cur_hard
                    best_h = h.copy()
            else:
                # If no improvement found, do a random move to escape local minima
                if rng.random() < 0.3:
                    i = int(rng.integers(0, N))
                    ac_new = _analytic_flip_estimate(h, ac, i)
                    new_hard = float(np.max(np.abs(ac_new)) / ac_new[0]) if ac_new[0] > 1e-30 else 1e9
                    if new_hard < cur_hard:
                        h[i] *= -1.0
                        ac = ac_new
                        cur_hard = new_hard
                        if cur_hard < best_hard:
                            best_hard = cur_hard
                            best_h = h.copy()

        # ---------- Phase 2: Simulated annealing with block flips ----------
        h = best_h.copy()
        cur_hard = best_hard
        ac = _autocorr(h)
        
        # SA parameters
        T_start = 0.5
        T_end = 0.01
        cooling_rate = 0.995
        max_sa_iterations = 20000
        
        T = T_start
        for _iter in range(max_sa_iterations):
            if time.time() > deadline - 10.0:
                break
            
            # Choose block size: 1, 2, 4, 8, 16
            block_size = int(2 ** rng.integers(0, 5))  # 1, 2, 4, 8, 16
            if block_size > N:
                block_size = N
            
            # Choose random positions to flip
            flip_indices = rng.choice(N, size=block_size, replace=False)
            
            # Compute new autocorrelation
            ac_new = _analytic_block_flip_estimate(h, ac, flip_indices)
            new_hard = float(np.max(np.abs(ac_new)) / ac_new[0]) if ac_new[0] > 1e-30 else 1e9
            
            # Acceptance criterion
            if new_hard < cur_hard or rng.random() < np.exp(-(new_hard - cur_hard) / T):
                h[flip_indices] *= -1.0
                ac = ac_new
                cur_hard = new_hard
                if cur_hard < best_hard:
                    best_hard = cur_hard
                    best_h = h.copy()
            
            # Cool down
            T *= cooling_rate
            if T < T_end:
                T = T_end

        # ---------- final single-bit polish with exact FFT ----------
        h = best_h.copy()
        for _pass in range(100):
            if time.time() > deadline - 0.5:
                break
            any_improve = False
            ac = _autocorr(h)
            dc = ac[0]
            if dc <= 1e-30:
                break
            for i in range(N):
                ac_new = _analytic_flip_estimate(h, ac, i)
                new_hard = float(np.max(np.abs(ac_new)) / ac_new[0]) if ac_new[0] > 1e-30 else 1e9
                if new_hard < best_hard - 1e-12:
                    h[i] *= -1.0
                    ac = ac_new
                    best_hard = new_hard
                    best_h = h.copy()
                    any_improve = True
            if not any_improve:
                break

        # Final exact verification
        final_hard = _hard_score(best_h)
        if final_hard < best_hard:
            best_hard = final_hard

        return best_h
    except Exception:
        return np.ones(N)
