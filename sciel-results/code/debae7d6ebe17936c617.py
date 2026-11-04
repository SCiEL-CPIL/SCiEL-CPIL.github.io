"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize

N = 400

# EVOLVE-BLOCK-START

def _autoconv_full(x):
    """Full linear autoconvolution of x (length 2N-1)."""
    n = len(x)
    size = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(x, n=size)
    conv = np.fft.irfft(F * F, n=size)
    return conv[:2 * n - 1]


def _score_offdiag(x):
    """
    Compute the metric: max(|autoconv|) for k != 0 / (sum(x))^2
    We want to minimize this.
    """
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e12
    conv = _autoconv_full(x)
    if len(conv) <= 1:
        return 0.0
    max_abs_offdiag = np.max(np.abs(conv[1:]))
    return max_abs_offdiag / (integral * integral)


def _score_std(x):
    """
    Compute the metric: max(|autoconv|) for all k / (sum(x))^2
    """
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e12
    conv = _autoconv_full(x)
    max_abs = np.max(np.abs(conv))
    return max_abs / (integral * integral)


def _param_to_signal(params):
    """
    Convert spectral coefficients to a time-domain signal.
    params: array of length 2*M+1, representing [c0, c1, d1, c2, d2, ..., cM, dM]
    x[n] = c0 + sum_{k=1}^{M} c_k * cos(2*pi*k*n/N) + d_k * sin(2*pi*k*n/N)
    """
    M = len(params) // 2
    c0 = params[0]
    x = np.full(N, c0)
    n = np.arange(N)
    for k in range(1, M + 1):
        c_k = params[2 * k - 1]
        d_k = params[2 * k]
        x += c_k * np.cos(2 * np.pi * k * n / N) + d_k * np.sin(2 * np.pi * k * n / N)
    return x


def _soft_max_offdiag(x, beta=50.0):
    """
    Smoothed objective for off-diagonal metric using log-sum-exp for soft max.
    """
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    conv = _autoconv_full(x)
    if len(conv) <= 1:
        return 0.0
    abs_c = np.abs(conv[1:])
    m = np.max(abs_c)
    if m < 1e-15:
        return 0.0
    soft_max = m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta
    return soft_max / (s * s)


def _m_sequence(n, k):
    """Generate an m-sequence of length n using LFSR with k stages."""
    polys = {
        4: [3, 0],
        5: [4, 3, 0],
        6: [5, 3, 0],
        7: [6, 5, 3, 0],
        8: [7, 5, 3, 2, 0],
    }
    if k not in polys:
        return np.sign(np.sin(np.arange(n) * 2 * np.pi / n * 3))[:n]
    
    taps = polys[k]
    state = [0] * k
    state[0] = 1
    seq = []
    for _ in range(n):
        fb = 0
        for t in taps:
            fb ^= state[t]
        for i in range(k - 1, 0, -1):
            state[i] = state[i - 1]
        state[0] = fb
        seq.append(1.0 if state[0] else -1.0)
    return np.array(seq, dtype=np.float64)


def _block_pattern(n, block_size, neg_frac):
    """Generate block pattern with alternating +1/-1 blocks."""
    h = np.zeros(n, dtype=np.float64)
    for i in range(n):
        block_idx = i // block_size
        if block_idx % 2 == 0:
            pos_in_block = i % block_size
            n_neg = int(neg_frac * block_size)
            h[i] = -1.0 if pos_in_block < n_neg else 1.0
        else:
            pos_in_block = i % block_size
            n_neg = int(neg_frac * block_size)
            h[i] = 1.0 if pos_in_block < n_neg else -1.0
    return h


def _chunk_flip_search(current, current_score, best_score, best_h, rng, deadline, n, use_offdiag=True):
    """
    Structural change: chunk-based local search with peak-targeting.
    Instead of single-bit flips, flip contiguous chunks of bits.
    Proposals are biased toward positions near autoconvolution peaks.
    """
    size = 1 << (2 * n - 1).bit_length()
    
    # Track peak location for targeting (off-diagonal)
    ac = _autoconv_full(current)
    if use_offdiag and len(ac) > 1:
        peak_idx = np.argmax(np.abs(ac[1:])) + 1
    else:
        peak_idx = np.argmax(np.abs(ac))
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            break
        
        # Choose chunk size
        r = rng.random()
        if r < 0.4:
            chunk_sz = 1
        elif r < 0.6:
            chunk_sz = rng.choice([2, 3, 4])
        elif r < 0.8:
            chunk_sz = rng.choice([5, 8, 12])
        else:
            chunk_sz = rng.choice([16, 24, 32])
        
        # Peak-targeted proposal
        if rng.random() < 0.6:
            target = peak_idx % n
            spread = max(chunk_sz, n // 8)
            start = int(rng.integers(max(0, target - spread), min(n - chunk_sz, target + spread) + 1))
        else:
            start = int(rng.integers(0, max(1, n - chunk_sz + 1)))
        
        start = max(0, min(start, n - chunk_sz))
        end = start + chunk_sz
        
        # Perform the chunk flip
        old_vals = current[start:end].copy()
        current[start:end] = -old_vals
        
        # Compute new score
        s = np.sum(current)
        if abs(s) < 1e-12:
            new_score = np.inf
        else:
            H = np.fft.rfft(current, size)
            ac_new = np.fft.irfft(H * np.conj(H), size)[:2 * n - 1]
            if use_offdiag and len(ac_new) > 1:
                max_abs = np.max(np.abs(ac_new[1:]))
            else:
                max_abs = np.max(np.abs(ac_new))
            new_score = max_abs / (s * s)
        
        # Accept or reject
        if new_score <= current_score:
            current_score = new_score
            no_improve_count = 0
            if new_score < best_score:
                best_score = new_score
                best_h = current.copy()
                if use_offdiag and len(ac_new) > 1:
                    peak_idx = np.argmax(np.abs(ac_new[1:])) + 1
                else:
                    peak_idx = np.argmax(np.abs(ac_new))
        else:
            current[start:end] = old_vals
            no_improve_count += 1
    
    return current, current_score, best_score, best_h


def _spectral_optimize(x_init, deadline, rng):
    """
    Optimize using spectral parameterization with differential evolution
    and a soft-max surrogate for the off-diagonal metric.
    """
    # Initialize parameters from the given signal
    n = len(x_init)
    # Use M frequency bins
    M = 20
    n_params = 2 * M + 1
    
    # Initialize params: try to fit the initial signal
    # Simple approach: start with random params and let DE optimize
    params_init = rng.uniform(-5, 5, size=n_params)
    
    bounds = [(-20, 20)] * n_params
    
    def objective(params):
        x = _param_to_signal(params)
        return _soft_max_offdiag(x, beta=30.0)
    
    try:
        result = scipy.optimize.differential_evolution(
            objective,
            bounds=bounds,
            maxiter=50,
            popsize=12,
            tol=1e-6,
            seed=42,
            polish=True,
            updating='immediate'
        )
        
        if result.fun < 1e10:
            x_opt = _param_to_signal(result.x)
            score_opt = _score_offdiag(x_opt)
            return x_opt, score_opt
    except Exception:
        pass
    
    return x_init, _score_offdiag(x_init)


def _multi_restart_search(best_h, best_score, rng, deadline, n):
    """
    Multiple restarts from perturbed versions of the best solution.
    """
    current_best = best_score
    current_best_h = best_h.copy()
    
    num_restarts = 15
    per_restart_time = (deadline - time.time()) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        h = current_best_h.copy()
        flip_frac = rng.choice([0.05, 0.1, 0.15, 0.2])
        n_flips = int(flip_frac * n)
        flip_indices = rng.choice(n, size=n_flips, replace=False)
        h[flip_indices] = -h[flip_indices]
        
        s = np.sum(h)
        if abs(s) < 1e-12:
            continue
        size = 1 << (2 * n - 1).bit_length()
        H = np.fft.rfft(h, size)
        ac = np.fft.irfft(H * np.conj(H), size)[:2 * n - 1]
        if len(ac) > 1:
            max_abs = np.max(np.abs(ac[1:]))
        else:
            max_abs = np.max(np.abs(ac))
        score = max_abs / (s * s)
        
        if score < current_best:
            h = current_best_h.copy()
            score = current_best
        
        restart_deadline = min(time.time() + per_restart_time, deadline)
        h, score, restart_best, restart_best_h = _chunk_flip_search(
            h, score, current_best, current_best_h.copy(), rng, restart_deadline, n, use_offdiag=True
        )
        
        if restart_best < current_best:
            current_best = restart_best
            current_best_h = restart_best_h
    
    return current_best_h, current_best


def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    rng = np.random.default_rng(42)
    
    # Diverse initializations
    candidates = []
    
    # 1. Hanning window (good off-diagonal behavior)
    candidates.append(np.hanning(N))
    
    # 2. Hamming window
    candidates.append(np.hamming(N))
    
    # 3. M-sequences
    for k in [6, 7, 8]:
        candidates.append(_m_sequence(N, k))
    
    # 4. Block patterns
    for bs in [8, 16, 32, 64]:
        for nf in [0.0, 0.2, 0.3, 0.5]:
            candidates.append(_block_pattern(N, bs, nf))
    
    # 5. Random with varying bias
    for bias in [0.0, 0.1, 0.2, 0.3]:
        p_pos = 0.5 + bias / 2
        h = rng.choice([-1.0, 1.0], size=N, p=[1 - p_pos, p_pos])
        candidates.append(h)
    
    # 6. Sinusoidal modulation
    for freq in [1, 2, 3, 5]:
        h = np.sign(np.sin(2 * np.pi * freq * np.arange(N) / N))
        candidates.append(h)
    
    # 7. Walsh-Hadamard-like patterns
    for level in range(4):
        h = np.ones(N)
        for i in range(N):
            val = 1.0
            for bit in range(level + 1):
                if (i >> bit) & 1:
                    val *= -1.0
            h[i] = val
        candidates.append(h)
    
    # 8. Gaussian window
    x = np.arange(N)
    sigma = N / 4
    candidates.append(np.exp(-0.5 * ((x - N/2) / sigma) ** 2))
    
    # Evaluate all candidates using off-diagonal metric
    best_score = np.inf
    best_h = np.ones(N)
    
    for h in candidates:
        if time.time() > deadline - 60:
            break
        s = _score_offdiag(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    n = N
    
    # Phase 1: Spectral optimization from best candidate
    if time.time() < deadline - 50:
        spectral_deadline = time.time() + 30
        best_h, best_score = _spectral_optimize(best_h, spectral_deadline, rng)
    
    # Phase 2: Chunk-flip local search
    phase2_deadline = min(t0 + 100, deadline - 20)
    best_h, best_score, best_score, best_h = _chunk_flip_search(
        best_h.copy(), best_score, best_score, best_h.copy(), rng, phase2_deadline, n, use_offdiag=True
    )
    
    # Phase 3: Multi-restart search
    if time.time() < deadline - 10:
        best_h, best_score = _multi_restart_search(
            best_h, best_score, rng, deadline - 5, n
        )
    
    # Phase 4: Final refinement
    if time.time() < deadline:
        best_h, best_score, best_score, best_h = _chunk_flip_search(
            best_h.copy(), best_score, best_score, best_h.copy(), rng, deadline, n, use_offdiag=True
        )
    
    return best_h

# EVOLVE-BLOCK-END
