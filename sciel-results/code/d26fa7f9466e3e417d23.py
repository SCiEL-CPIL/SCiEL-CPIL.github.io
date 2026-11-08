"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.signal import fftconvolve

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_c3prime(f):
    """Compute max |autoconvolution| / sum(f^2)"""
    f = np.asarray(f, dtype=np.float64)
    # Full autoconvolution
    ac = fftconvolve(f, f[::-1], mode='full')
    # The peak is at the center
    n = len(f)
    max_abs = np.max(np.abs(ac))
    l2_sq = np.sum(f * f)
    if l2_sq == 0:
        return float('inf')
    return max_abs / l2_sq

def _try_binary_random(rng, n_trials=2000):
    """Try random +/-1 sequences"""
    best_score = float('inf')
    best_seq = None
    for _ in range(n_trials):
        seq = rng.choice([-1.0, 1.0], size=N)
        s = _compute_c3prime(seq)
        if s < best_score:
            best_score = s
            best_seq = seq.copy()
    return best_seq, best_score

def _try_barker_like():
    """Try known low-autocorrelation patterns"""
    # Barker sequence of length 13
    barker13 = np.array([1,1,1,-1,1,1,1,1,-1,-1,1,-1,-1], dtype=np.float64)
    # Tile and adjust to length N
    repeats = N // len(barker13) + 1
    seq = np.tile(barker13, repeats)[:N]
    return seq

def _try_m_sequence_like():
    """Try a pseudo-random sequence with good autocorrelation properties"""
    # Use a simple LFSR-based m-sequence
    # For length 400, we can use a random seed and generate
    rng = np.random.RandomState(42)
    # Try several random seeds
    best_seq = None
    best_score = float('inf')
    for seed in range(50):
        rng = np.random.RandomState(seed)
        seq = rng.choice([-1.0, 1.0], size=N)
        s = _compute_c3prime(seq)
        if s < best_score:
            best_score = s
            best_seq = seq.copy()
    return best_seq, best_score

def _try_gradient_descent(rng):
    """Try local optimization from a random start"""
    seq = rng.choice([-1.0, 1.0], size=N)
    best_score = _compute_c3prime(seq)
    best_seq = seq.copy()
    
    # Simple local search: flip one element at a time
    for iteration in range(5000):
        i = rng.randint(N)
        seq[i] = -seq[i]
        s = _compute_c3prime(seq)
        if s < best_score:
            best_score = s
            best_seq = seq.copy()
        else:
            seq[i] = -seq[i]  # revert
    
    return best_seq, best_score

def construct():
    deadline = time.time() + 110  # safety margin
    
    best_seq = np.ones(N)
    best_score = _compute_c3prime(best_seq)
    
    rng = np.random.RandomState(12345)
    
    # Try random binary sequences
    if time.time() < deadline:
        seq, s = _try_binary_random(rng, n_trials=500)
        if s < best_score:
            best_seq = seq
            best_score = s
    
    # Try m-sequence-like
    if time.time() < deadline:
        seq, s = _try_m_sequence_like()
        if s < best_score:
            best_seq = seq
            best_score = s
    
    # Try gradient descent / local search from best
    if time.time() < deadline:
        seq, s = _try_gradient_descent(rng)
        if s < best_score:
            best_seq = seq
            best_score = s
    
    # More random trials if time permits
    if time.time() < deadline - 10:
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            seq = rng.choice([-1.0, 1.0], size=N)
            s = _compute_c3prime(seq)
            if s < best_score:
                best_seq = seq
                best_score = s
    
    return best_seq
# EVOLVE-BLOCK-END
