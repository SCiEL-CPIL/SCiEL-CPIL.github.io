import numpy as np
import time

N = 400
DX = 0.5 / N  # interval width: total length 0.5, N intervals

def _linear_acf(h):
    """
    Compute the linear autoconvolution coefficients c_k = sum_i h[i]*h[i+k]
    for k = -(N-1), ..., N-1 using FFT.
    Returns array of length 2N-1 where index k+N-1 corresponds to lag k.
    """
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    power = np.abs(H) ** 2
    acf_full = np.fft.irfft(power, n=fft_size)
    # acf_full[0] = c_0, acf_full[1] = c_1, ..., acf_full[n-1] = c_{n-1}
    # acf_full[fft_size-1] = c_{-1}, acf_full[fft_size-2] = c_{-2}, ...
    # We want c_k for k = -(n-1) to n-1
    # c_k for k>=0: acf_full[k]
    # c_k for k<0: acf_full[fft_size+k]
    acf = np.empty(2 * n - 1, dtype=np.float64)
    acf[n-1] = acf_full[0]
    for k in range(1, n):
        acf[n-1+k] = acf_full[k]
        acf[n-1-k] = acf_full[fft_size - k]
    return acf

def _score(h):
    """
    Compute the upper bound on C3: dx * max_k |c_k| for k != 0.
    This is the max |autoconvolution| of the step function at grid points.
    """
    acf = _linear_acf(h)
    # Exclude lag 0 (center)
    n = len(h)
    center = n - 1
    # Lags 1 to n-1 on both sides
    right = acf[center+1:]  # lags 1 to n-1
    left = acf[:center]     # lags -(n-1) to -1
    max_abs = max(np.max(np.abs(right)), np.max(np.abs(left)))
    return float(DX * max_abs)

def _compute_acf(h):
    """Compute full linear autocorrelation via FFT."""
    return _linear_acf(h)

def _psl_value(acf, n):
    """Get PSL value from acf array (max abs at non-zero lags)."""
    if n <= 1:
        return 0.0
    center = n - 1
    right = acf[center+1:]
    left = acf[:center]
    return float(max(np.max(np.abs(right)), np.max(np.abs(left))))

def _gradient_step(h, acf):
    """
    Find the single best sign flip to reduce PSL.
    For flipping h[i], the linear ACF change at lag k is:
      delta[k] = -2 * h[i] * (contribution of h[i] to c_k)
    c_k = sum_j h[j] * h[j+k]
    Flipping h[i] changes c_k by:
      -2*h[i] * (h[i-k] if i-k in [0,n-1] else 0)
      -2*h[i] * (h[i+k] if i+k in [0,n-1] else 0)
      +4*h[i]^2 if k=0 and i in range (but we don't consider k=0 for PSL)
    Actually for k != 0:
      delta_c_k = -2*h[i] * (h[i-k] * 1_{0<=i-k<n} + h[i+k] * 1_{0<=i+k<n})
    """
    n = len(h)
    current_psl = _psl_value(acf, n)
    # acf is length 2n-1, center at n-1
    # We work with the full acf array
    best_i = -1
    best_new_psl = float('inf')

    for i in range(n):
        # Compute delta for each lag
        # delta[k] for lag k (k from -(n-1) to n-1, k != 0)
        # c_k = sum_j h[j]*h[j+k]
        # After flipping h[i]: c_k' = c_k - 2*h[i]*(h[i-k]*1_{valid} + h[i+k]*1_{valid})
        # but only if i is one of the indices in the sum for lag k
        
        # For lag k > 0: c_k = sum_{j=0}^{n-1-k} h[j]*h[j+k]
        #   h[i] appears if i is in [0, n-1-k] (as h[j]) or i+k is in [0, n-1] (as h[j+k])
        #   i.e., j=i means i in [0, n-1-k], or j+k=i means j=i-k in [0, n-1-k]
        #   Contribution: h[i]*h[i+k] (if i+k < n) + h[i-k]*h[i] (if i-k >= 0)
        #   But these are the same term when both valid? No, they're different pairs.
        #   Actually c_k = sum_j h[j] h[j+k]. The terms involving h[i] are:
        #     j=i: h[i] h[i+k] (if i+k < n)
        #     j=i-k: h[i-k] h[i] (if i-k >= 0)
        #   So delta_c_k = -2 h[i] (h[i+k] * 1_{i+k<n} + h[i-k] * 1_{i-k>=0})
        
        # For lag k < 0: c_k = sum_j h[j] h[j+k] = sum_j h[j-k] h[j] (by substitution)
        #   = c_{|k|} by symmetry of autoconvolution
        #   So we can just use |k| and the same formula
        
        # Let's compute delta for all lags efficiently
        # delta[k] for k in 1..n-1 (positive lags)
        delta = np.zeros(2 * n - 1, dtype=np.float64)
        
        # For positive lag k (1 to n-1):
        #   delta[k] = -2*h[i] * (h[i+k] if i+k < n else 0) 
        #             - 2*h[i] * (h[i-k] if i-k >= 0 else 0)
        #   Note: these are two separate terms unless i+k and i-k refer to same index (impossible for k>0)
        
        for k in range(1, n):
            term1 = 0.0
            term2 = 0.0
            if i + k < n:
                term1 = h[i + k]
            if i - k >= 0:
                term2 = h[i - k]
            delta[k] = -2.0 * h[i] * (term1 + term2)
            delta[-k] = delta[k]  # symmetry
        
        new_acf = acf + delta
        new_psl = _psl_value(new_acf, n)
        
        if new_psl < best_new_psl:
            best_new_psl = new_psl
            best_i = i

    return best_i, float(best_new_psl), current_psl

def _fast_psl_optimize(h, max_passes=300, time_budget=None, start_time=None):
    """
    Fast PSL optimization using Gordon's gradient method.
    Repeatedly find the single best flip and apply it.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    if start_time is None:
        start_time = time.time()

    best_h = h.copy()
    acf = _compute_acf(h)
    best_psl = _psl_value(acf, n)

    passes = 0
    while passes < max_passes:
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break

        i, new_psl, cur_psl = _gradient_step(h, acf)
        if i == -1 or new_psl >= cur_psl:
            break

        h[i] = -h[i]
        passes += 1
        acf = _compute_acf(h)
        cur_psl = _psl_value(acf, n)

        if cur_psl < best_psl:
            best_psl = cur_psl
            best_h = h.copy()

        if best_psl < 1e-10:
            break

    return best_h, best_psl

def _chirp_family_candidates(rng):
    """
    Generate diverse candidates from the chirp family.
    """
    candidates = []
    t = np.arange(N, dtype=np.float64)

    # Zadoff-Chu-like: pure quadratic chirp with various roots
    for root in [1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]:
        val = np.cos(2.0 * np.pi * root * t * t / N)
        candidates.append(np.sign(val))

    # Mixed linear + quadratic chirp
    for trial in range(40):
        alpha = rng.uniform(0.01, 0.5)
        beta = rng.uniform(0.001, 0.1)
        gamma = rng.uniform(0.0, 2.0 * np.pi)
        delta = rng.uniform(-0.5, 0.5)
        val = (np.cos(alpha * t * t)
               + beta * np.cos(2.0 * np.pi * t / N + gamma)
               + delta)
        candidates.append(np.sign(val))

    # Pure quadratic with phase offset
    for trial in range(20):
        alpha = rng.uniform(0.05, 0.3)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        val = np.cos(alpha * t * t + phase)
        candidates.append(np.sign(val))

    # Cubic chirp
    for trial in range(15):
        beta = rng.uniform(0.005, 0.05)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        val = np.cos(beta * t * t * t + phase)
        candidates.append(np.sign(val))

    # Two orthogonal chirps summed
    for trial in range(25):
        a1 = rng.uniform(0.05, 0.3)
        a2 = rng.uniform(0.05, 0.3)
        p1 = rng.uniform(0.0, 2.0 * np.pi)
        p2 = rng.uniform(0.0, 2.0 * np.pi)
        val = np.cos(a1 * t * t + p1) + np.cos(a2 * t * t + p2)
        candidates.append(np.sign(val))

    # Linear + quadratic (orthogonal chirp pair)
    for trial in range(25):
        lin_freq = rng.uniform(1.0, 10.0)
        quad_freq = rng.uniform(0.05, 0.3)
        p1 = rng.uniform(0.0, 2.0 * np.pi)
        p2 = rng.uniform(0.0, 2.0 * np.pi)
        val = (np.cos(2.0 * np.pi * lin_freq * t / N + p1)
               + np.cos(quad_freq * t * t + p2))
        candidates.append(np.sign(val))

    # Ensure no all-zeros
    for i, c in enumerate(candidates):
        if np.all(c == 0):
            candidates[i] = np.ones(N)

    return candidates

def _simulated_annealing_psl(h, time_budget=15.0, start_time=None, rng=None):
    """
    Simulated annealing on PSL using fast gradient evaluation.
    """
    if rng is None:
        rng = np.random.default_rng()

    h = h.copy().astype(np.float64)
    n = len(h)
    if start_time is None:
        start_time = time.time()

    acf = _compute_acf(h)
    current_psl = _psl_value(acf, n)
    best_h = h.copy()
    best_psl = current_psl

    T = 5.0
    T_min = 0.05
    cooling = 0.93

    while T > T_min:
        if time.time() - start_time > time_budget:
            break

        i = int(rng.integers(n))
        
        # Compute delta for this flip
        delta = np.zeros(2 * n - 1, dtype=np.float64)
        for k in range(1, n):
            term1 = 0.0
            term2 = 0.0
            if i + k < n:
                term1 = h[i + k]
            if i - k >= 0:
                term2 = h[i - k]
            delta[k] = -2.0 * h[i] * (term1 + term2)
            delta[-k] = delta[k]
        
        new_acf = acf + delta
        new_psl = _psl_value(new_acf, n)
        delta_psl = new_psl - current_psl

        if delta_psl < 0 or np.random.random() < np.exp(-delta_psl / T):
            h[i] = -h[i]
            acf = _compute_acf(h)
            current_psl = _psl_value(acf, n)
            if current_psl < best_psl:
                best_psl = current_psl
                best_h = h.copy()

        T *= cooling

    return best_h, best_psl

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    best_h = None
    best_score = float('inf')
    rng = np.random.default_rng(42)

    # Generate chirp-family candidates
    candidates = _chirp_family_candidates(rng)

    # Also add a few random and periodic candidates for diversity
    for i in range(5):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    for period in [3, 5, 7, 11]:
        pattern = np.sign(np.cos(2.0 * np.pi * np.arange(N) / period))
        candidates.append(pattern)

    # Score all candidates
    scored = []
    for h in candidates:
        s = _score(h)
        scored.append((s, h))
    scored.sort(key=lambda x: x[0])

    # Refine top candidates with fast PSL optimization
    top_k = 8
    for i, (s, h) in enumerate(scored[:top_k]):
        if time.time() > deadline - 15:
            break
        time_limit = deadline - 10
        h_refined, s_refined = _fast_psl_optimize(
            h, max_passes=400, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()

    # Random restarts with fast PSL optimization
    while time.time() < deadline - 10:
        h_init = rng.choice([-1.0, 1.0], size=N)
        time_limit = min(deadline - 5, time.time() + 4)
        h_refined, s_refined = _fast_psl_optimize(
            h_init, max_passes=80, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()

    # SA on best found so far
    if best_h is not None and time.time() < deadline - 5:
        sa_time = min(10.0, deadline - 5 - (time.time() - start_time))
        if sa_time > 2.0:
            h_sa, s_sa = _simulated_annealing_psl(
                best_h, time_budget=sa_time, start_time=start_time, rng=rng)
            if s_sa < best_score:
                best_score = s_sa
                best_h = h_sa.copy()

    if best_h is None:
        best_h = np.ones(N)

    return best_h
