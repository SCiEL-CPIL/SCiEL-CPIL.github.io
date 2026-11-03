"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _fft_score(h):
    """Compute max |autoconv| / (sum(h))^2 using FFT. Lower is better."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    n = len(h)
    size = 1 << (2 * n - 1).bit_length()
    H = np.fft.rfft(h, size)
    ac = np.fft.irfft(H * np.conj(H), size)[:2 * n - 1]
    max_abs = np.max(np.abs(ac))
    return max_abs / (s * s)


def _m_sequence(n, k):
    """Generate an m-sequence of length n using LFSR with k stages."""
    # Primitive polynomials for small k
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
            # Positive block
            pos_in_block = i % block_size
            n_neg = int(neg_frac * block_size)
            h[i] = -1.0 if pos_in_block < n_neg else 1.0
        else:
            # Negative block
            pos_in_block = i % block_size
            n_neg = int(neg_frac * block_size)
            h[i] = 1.0 if pos_in_block < n_neg else -1.0
    return h


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 10s safety margin from 150s limit

    rng = np.random.default_rng(42)
    
    # Diverse initializations
    candidates = []
    
    # 1. Constant ones (baseline)
    candidates.append(np.ones(N))
    
    # 2. M-sequences with different k
    for k in [6, 7, 8]:
        candidates.append(_m_sequence(N, k))
    
    # 3. Block patterns
    for bs in [4, 8, 16, 32, 64]:
        for nf in [0.0, 0.2, 0.3, 0.5]:
            candidates.append(_block_pattern(N, bs, nf))
    
    # 4. Random with varying bias
    for bias in [0.0, 0.1, 0.2, 0.3, 0.5]:
        p_pos = 0.5 + bias / 2
        h = rng.choice([-1.0, 1.0], size=N, p=[1 - p_pos, p_pos])
        candidates.append(h)
    
    # 5. Sinusoidal modulation
    for freq in [1, 2, 3, 5]:
        h = np.sign(np.sin(2 * np.pi * freq * np.arange(N) / N))
        candidates.append(h)
    
    # Evaluate all candidates
    best_score = np.inf
    best_h = np.ones(N)
    
    for h in candidates:
        if time.time() > deadline:
            break
        s = _fft_score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Simulated annealing with efficient evaluation
    current = best_h.copy()
    current_score = best_score
    T = 1.0
    T_min = 1e-8
    cooling_rate = 0.999
    
    # Precompute FFT size
    n = len(current)
    size = 1 << (2 * n - 1).bit_length()
    
    iter_count = 0
    max_iters = 200000
    
    while time.time() < deadline and iter_count < max_iters:
        iter_count += 1
        
        # Propose a flip
        idx = rng.integers(N)
        old_val = current[idx]
        current[idx] = -old_val
        
        # Compute new score
        s = np.sum(current)
        if abs(s) < 1e-12:
            new_score = np.inf
        else:
            H = np.fft.rfft(current, size)
            ac = np.fft.irfft(H * np.conj(H), size)[:2 * n - 1]
            max_abs = np.max(np.abs(ac))
            new_score = max_abs / (s * s)
        
        # Accept or reject
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = current.copy()
        else:
            # Probabilistic acceptance
            delta = new_score - current_score
            if delta > 0 and delta / T < 50:
                if rng.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    current[idx] = old_val
            else:
                current[idx] = old_val
        
        # Cool down
        T = max(T * cooling_rate, T_min)
    
    return best_h

# EVOLVE-BLOCK-END
