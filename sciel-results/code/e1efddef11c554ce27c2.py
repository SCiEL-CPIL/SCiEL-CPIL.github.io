import numpy as np
import time

N = 400
_NFFT = 1 << (2 * N - 1).bit_length()  # next power of 2 >= 2N-1 for fast FFT


def _autocorr_fast(h):
    # linear one-sided autocorrelation ac[k] = sum_i h[i] h[i+k], k = 0..N-1
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
    if m < 1e-30:
        return 0.0
    z = beta * (y - m)
    return m + float(np.log(np.sum(np.exp(z)))) / beta


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    t = [1]
    while len(t) < n:
        t += [-x for x in t]
    return np.array(t[:n])


def _analytic_flip_estimate(h, ac, i):
    """
    Estimate the new autocorrelation after flipping bit i without full FFT.
    Vectorized: computes delta for all lags at once.
    """
    ac_new = ac.copy()
    if i < N:
        hi = h[i]
        # For lag k (1..N-1): contribution to ac[k] from pair (i, i+k) and (i-k, i)
        # Flipping h[i] changes ac[k] by -2*h[i]*(h[i+k] + h[i-k]) for valid indices
        k = np.arange(1, N)
        
        # h[i+k] valid when i+k < N, i.e., k < N-i
        h_plus = np.zeros(N - 1)
        valid_plus = k < (N - i)
        h_plus[valid_plus] = h[i + k[valid_plus]]
        
        # h[i-k] valid when i-k >= 0, i.e., k <= i
        h_minus = np.zeros(N - 1)
        valid_minus = k <= i
        h_minus[valid_minus] = h[i - k[valid_minus]]
        
        delta = -2.0 * hi * (h_plus + h_minus)
        ac_new[1:] += delta
        
    return ac_new

def _analytic_block_flip_estimate(h, ac, flip_indices):
    """
    Estimate the new autocorrelation after flipping a set of bits.
    Vectorized over flip indices.
    """
    ac_new = ac.copy()
    n_flips = len(flip_indices)
    if n_flips == 0:
        return ac_new
        
    k = np.arange(1, N)
    flips = np.array(flip_indices, dtype=int)
    
    # For each flip i and each lag k, compute contribution
    # h[i+k] valid when i+k < N
    # h[i-k] valid when i-k >= 0
    
    # Build arrays: for each flip i_j and lag k_l:
    #   plus_val[j,l] = h[i_j + k_l] if i_j + k_l < N else 0
    #   minus_val[j,l] = h[i_j - k_l] if i_j - k_l >= 0 else 0
    
    i_idx = flips[:, None]  # (n_flips, 1)
    k_idx = k[None, :]       # (1, N-1)
    
    i_plus_k = i_idx + k_idx  # (n_flips, N-1)
    i_minus_k = i_idx - k_idx
    
    # Initialize with zeros
    h_plus = np.zeros((n_flips, N - 1))
    h_minus = np.zeros((n_flips, N - 1))
    
    valid_plus = i_plus_k < N
    valid_minus = i_minus_k >= 0
    
    # Clip indices for safe indexing
    i_plus_k_clipped = np.clip(i_plus_k, 0, N - 1)
    i_minus_k_clipped = np.clip(i_minus_k, 0, N - 1)
    
    h_plus[valid_plus] = h[i_plus_k_clipped[valid_plus]]
    h_minus[valid_minus] = h[i_minus_k_clipped[valid_minus]]
    
    # Sum contributions: delta[k] = -2 * sum_j h[i_j] * (h_plus[j,k] + h_minus[j,k])
    hi_vals = h[flips]  # (n_flips,)
    delta = -2.0 * np.sum(hi_vals[:, None] * (h_plus + h_minus), axis=0)
    ac_new[1:] += delta
    
    return ac_new

def _continuous_gradient_init(rng, time_budget=2.0):
    """
    Use continuous gradient descent on L2 norm of autocorrelation sidelobes
    to find a good initial point, then threshold to binary.
    """
    t0 = time.time()
    # Start with random continuous values
    h = rng.normal(0, 1, N)
    
    lr = 0.01
    for step in range(2000):
        if time.time() - t0 > time_budget:
            break
        
        ac = _autocorr(h)
        dc = ac[0]
        if dc <= 1e-30:
            break
        
        # Objective: minimize sum of squared sidelobes (normalized)
        # gradient of sum_k (ac[k]/dc)^2 w.r.t. h
        # d ac[k] / d h[i] = h[i+k] + h[i-k] (for valid indices)
        
        # Compute gradient efficiently
        grad = np.zeros(N)
        sidelobes = ac[1:] / dc
        
        for k in range(1, N):
            dk = sidelobes[k - 1] / dc  # derivative contribution
            # d(ac[k])/d(h[i]) = h[i+k] + h[i-k]
            valid_plus = np.arange(N) + k < N
            valid_minus = np.arange(N) - k >= 0
            
            idx_plus = np.where(valid_plus)[0]
            idx_minus = np.where(valid_minus)[0]
            
            if len(idx_plus) > 0:
                grad[idx_plus] += 2 * dk * h[idx_plus + k]
            if len(idx_minus) > 0:
                grad[idx_minus] += 2 * dk * h[idx_minus - k]
        
        h -= lr * grad
        
        # Periodically threshold to keep values near binary
        if step % 50 == 0:
            h = np.sign(h)
            h[h == 0] = rng.choice([-1.0, 1.0], size=np.sum(h == 0))
    
    # Final threshold
    h = np.sign(h)
    h[h == 0] = 1.0
    return h


def construct():
    try:
        t0 = time.time()
        deadline = t0 + 115.0
        rng = np.random.default_rng(20240517)

        idx = np.arange(N)
        
        # Generate initial candidates including continuous gradient init
        inits = []
        
        # Thue-Morse
        inits.append(_thue_morse(N))
        
        # Random binary
        for _ in range(3):
            inits.append(rng.choice(np.array([-1.0, 1.0]), size=N))
        
        # Chirp-like sequences
        inits.append(np.sign(np.cos(2 * np.pi * idx ** 2 / (2 * N))))
        inits.append(np.sign(np.cos(2 * np.pi * idx ** 2 / (N + 1))))
        inits.append(np.sign(np.cos(2 * np.pi * idx ** 2 / (3 * N))))
        
        # Sine/cosine
        inits.append(np.sign(np.cos(2 * np.pi * idx / N)))
        inits.append(np.sign(np.sin(2 * np.pi * idx / N)))
        inits.append(np.sign(np.cos(2 * np.pi * 2 * idx / N)))
        
        # Continuous gradient descent init (key improvement)
        if time.time() < deadline - 100.0:
            h_cont = _continuous_gradient_init(rng, time_budget=2.0)
            inits.append(h_cont)
        
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

        # Phase 1: Greedy single-bit search
        max_iterations_phase1 = 50000
        for _iter in range(max_iterations_phase1):
            if time.time() > deadline - 30.0:
                break
            
            if _iter % 50 == 0:
                ac = _autocorr(h)
                dc = ac[0]
                if dc <= 1e-30:
                    break
                cur_hard = float(np.max(np.abs(ac)) / dc)
            
            sidelobes = np.abs(ac[1:])
            k_star = int(np.argmax(sidelobes)) + 1
            c_star = ac[k_star]
            
            max_i = N - k_star
            indices = np.arange(max_i)
            products = h[indices] * h[indices + k_star]
            
            if c_star > 0:
                candidates = indices[products > 0]
            else:
                candidates = indices[products < 0]
            
            if len(candidates) == 0:
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

        # Phase 2: Simulated Annealing with block flips
        h = best_h.copy()
        cur_hard = best_hard
        ac = _autocorr(h)
        cur_smooth = _smooth_score(h, 30.0)
        
        T_start = 0.3
        T_end = 0.005
        cooling_rate = 0.997
        max_sa_iterations = 30000
        beta = 30.0
        sizes = np.array([1, 2, 3, 4, 8, 12, 16, 20, 24, 32])
        
        T = T_start
        for _iter in range(max_sa_iterations):
            if time.time() > deadline - 10.0:
                break
            
            # Adaptive block size: smaller blocks at low temperature
            if T > 0.1:
                block_size = int(rng.choice(sizes))
            else:
                block_size = int(2 ** rng.integers(0, 3))
            if block_size > N:
                block_size = N
            
            flip_indices = rng.choice(N, size=block_size, replace=False)
            
            ac_new = _analytic_block_flip_estimate(h, ac, flip_indices)
            new_hard = float(np.max(np.abs(ac_new)) / ac_new[0]) if ac_new[0] > 1e-30 else 1e9
            new_smooth = _smooth_score(h[flip_indices] * -1.0 * 0 + h, beta)  # placeholder, need proper computation
            
            # Compute smooth score properly
            h_test = h.copy()
            h_test[flip_indices] *= -1.0
            new_smooth = _smooth_score(h_test, beta)
            
            dS = new_smooth - cur_smooth
            if new_hard < cur_hard or (dS <= 0 or rng.random() < np.exp(-dS / T)):
                h[flip_indices] *= -1.0
                ac = ac_new
                cur_hard = new_hard
                cur_smooth = new_smooth
                if cur_hard < best_hard:
                    best_hard = cur_hard
                    best_h = h.copy()
            
            T *= cooling_rate
            if T < T_end:
                T = T_end

        # Phase 3: High-precision polish
        h = best_h.copy()
        for _pass in range(200):
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

        final_hard = _hard_score(best_h)
        if final_hard < best_hard:
            best_hard = final_hard

        return best_h
    except Exception:
        return np.ones(N)
