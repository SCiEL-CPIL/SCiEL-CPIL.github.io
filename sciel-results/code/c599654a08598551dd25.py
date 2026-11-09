import time
import numpy as np
from scipy import signal

N = 400
START_TIME = time.time()
DEADLINE = START_TIME + 140.0  # safety margin before 150s


def _c3_prime_val(h):
    """
    Compute max_k |autocorr_k| / (sum h)^2 for signed step function h of length N.
    autocorr_k = sum_i h[i] * h[i+k] for k in [-(N-1), N-1], with zero padding.
    """
    h = np.asarray(h, dtype=float)
    total = h.sum()
    if abs(total) < 1e-12:
        return np.inf
    denom = total * total
    ac = np.correlate(h, h, mode='full')
    return np.max(np.abs(ac)) / denom


def _walsh_hadamard_sequence(n, index):
    """Generate the n-th Walsh-Hadamard sequence of length 2^k >= n."""
    # Find smallest power of 2 >= n
    k = 1
    while (1 << k) < n:
        k += 1
    size = 1 << k
    
    # Compute the n-th Walsh function
    # Hadamard matrix construction
    def hadamard(m):
        if m == 1:
            return np.array([[1.0]])
        H = hadamard(m // 2)
        return np.block([[H, H], [H, -H]])
    
    H = hadamard(size)
    row = H[index % size]
    return row[:n]


def _fourier_basis_signal(n, freq_index, phase=0.0):
    """Generate a sign(sin) signal at a specific Fourier frequency."""
    t = np.arange(n)
    return np.sign(np.sin(2 * np.pi * freq_index * t / n + phase))


def _random_zero_mean(n, seed):
    """Generate a random ±1 sequence with exactly zero mean (or near-zero)."""
    rng = np.random.RandomState(seed)
    h = rng.choice([-1.0, 1.0], size=n)
    # Adjust to make sum closer to zero
    total = h.sum()
    if total > 0:
        # Flip some +1 to -1
        n_flip = int(total / 2)
        if n_flip > 0:
            plus_indices = np.where(h == 1.0)[0]
            flip_idx = rng.choice(plus_indices, size=min(n_flip, len(plus_indices)), replace=False)
            h[flip_idx] = -1.0
    elif total < 0:
        n_flip = int(-total / 2)
        if n_flip > 0:
            minus_indices = np.where(h == -1.0)[0]
            flip_idx = rng.choice(minus_indices, size=min(n_flip, len(minus_indices)), replace=False)
            h[flip_idx] = 1.0
    return h


def _m_sequence_like(n, seed):
    """Generate a sequence with good autocorrelation properties (m-sequence-like)."""
    rng = np.random.RandomState(seed)
    # Start with random, then apply local optimization
    h = rng.choice([-1.0, 1.0], size=n)
    # Quick local refinement
    for _ in range(50):
        i = rng.randint(0, n)
        trial = h.copy()
        trial[i] *= -1
        if _c3_prime_val(trial) < _c3_prime_val(h):
            h = trial
    return h


def construct():
    global START_TIME, DEADLINE
    
    # Start with the seed (all ones)
    best_h = np.ones(N)
    best_val = _c3_prime_val(best_h)
    
    # Generate diverse candidate patterns
    candidates = []
    
    # 1. Walsh-Hadamard sequences (orthogonal, good autocorrelation properties)
    for idx in range(0, 32):
        candidates.append(_walsh_hadamard_sequence(N, idx))
    
    # 2. Fourier basis signals at various frequencies
    for freq in [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]:
        for phase in [0, np.pi/4, np.pi/2, np.pi/3]:
            candidates.append(_fourier_basis_signal(N, freq, phase))
    
    # 3. Random zero-mean sequences (many different seeds)
    for seed in range(100):
        candidates.append(_random_zero_mean(N, seed))
    
    # 4. Alternating patterns with various block sizes
    for block_size in [2, 3, 4, 5, 7, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100]:
        h = np.zeros(N)
        sign = 1.0
        for i in range(N):
            h[i] = sign
            if (i + 1) % block_size == 0:
                sign = -sign
        candidates.append(h)
    
    # 5. Chirp patterns
    t = np.arange(N)
    for f0 in [1, 2, 5]:
        for f1 in [5, 10, 20, 50]:
            if f1 != f0:
                freqs = f0 + (f1 - f0) * t / N
                phase = 2 * np.pi * np.cumsum(freqs) / N
                candidates.append(np.sign(np.sin(phase)))
    
    # Evaluate all candidates
    for h in candidates:
        if time.time() > DEADLINE:
            break
        try:
            val = _c3_prime_val(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
        except Exception:
            continue
    
    # Local refinement: try flipping individual bits and small segments
    if time.time() < DEADLINE - 30:
        current_h = best_h.copy()
        current_val = best_val
        improved = True
        max_iters = 500
        iters = 0
        while improved and iters < max_iters and time.time() < DEADLINE - 15:
            improved = False
            iters += 1
            
            # Try flipping individual bits
            for i in range(N):
                if time.time() > DEADLINE - 15:
                    break
                trial = current_h.copy()
                trial[i] *= -1
                try:
                    val = _c3_prime_val(trial)
                    if val < current_val - 1e-12:
                        current_val = val
                        current_h = trial
                        improved = True
                except Exception:
                    continue
            if time.time() > DEADLINE - 15:
                break
            
            # Try flipping small segments
            for seg_len in [2, 3, 5, 8, 12]:
                for start in range(0, N - seg_len + 1, max(1, seg_len // 2)):
                    if time.time() > DEADLINE - 15:
                        break
                    trial = current_h.copy()
                    trial[start:start+seg_len] *= -1
                    try:
                        val = _c3_prime_val(trial)
                        if val < current_val - 1e-12:
                            current_val = val
                            current_h = trial
                            improved = True
                    except Exception:
                        continue
                if time.time() > DEADLINE - 15:
                    break
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h
    
    return best_h
