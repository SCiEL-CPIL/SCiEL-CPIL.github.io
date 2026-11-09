"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_excl_lag0(f):
    """Compute autoconvolution excluding lag 0 using FFT for speed"""
    n = len(f)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    
    F = np.fft.fft(f, fft_size)
    ac = np.fft.ifft(F * F).real[:size]
    ac_no_lag0 = np.concatenate([ac[:n-1], ac[n:]])
    return np.max(np.abs(ac_no_lag0))


def _compute_objective(f):
    """Compute max |autoconv|_{k!=0} / (sum f)^2"""
    S = np.sum(f)
    if abs(S) < 1e-12:
        return np.inf
    max_ac = _autoconv_excl_lag0(f)
    return max_ac / (S * S)


def _generate_candidates(N):
    """Generate diverse initial candidates"""
    candidates = []
    
    # 1. All ones
    candidates.append(np.ones(N))
    
    # 2. Alternating pattern
    f_alt = np.ones(N)
    f_alt[1::2] = -1.0
    candidates.append(f_alt)
    
    # 3. Random ±1 sequences with different seeds
    for seed in range(8):
        np.random.seed(seed)
        f_rand = np.random.choice([-1.0, 1.0], size=N)
        if np.sum(f_rand) < 0:
            f_rand = -f_rand
        candidates.append(f_rand)
    
    # 4. Block-based patterns
    for block_size in [4, 8, 16, 32]:
        f_block = np.zeros(N)
        for i in range(N):
            block_idx = i // block_size
            f_block[i] = 1.0 if block_idx % 2 == 0 else -1.0
        candidates.append(f_block)
    
    # 5. Cosine-modulated sign pattern
    f_cos = np.zeros(N)
    for i in range(N):
        t = i / (N - 1)
        f_cos[i] = np.sign(np.cos(2 * np.pi * t * 2))
        if f_cos[i] == 0:
            f_cos[i] = 1.0
    if np.sum(f_cos) < 0:
        f_cos = -f_cos
    candidates.append(f_cos)
    
    return candidates


def _local_search(f, deadline, max_iters=100000):
    """Local search: try flipping signs to minimize objective"""
    best = f.copy()
    best_obj = _compute_objective(best)
    n = len(f)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        # Try flipping a random sign
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
    deadline = start_time + 145.0  # safety margin before 150s limit
    
    np.random.seed(42)
    
    # Try multiple initializations
    candidates = _generate_candidates(N)
    
    best_f = candidates[0].copy()
    best_obj = _compute_objective(best_f)
    
    # Allocate time budget among candidates
    time_per_candidate = (deadline - start_time - 10.0) / len(candidates)
    
    # Local search on each candidate
    for i, f_init in enumerate(candidates):
        if time.time() > deadline - 5.0:
            break
        
        f_opt, obj = _local_search(f_init, start_time + time_per_candidate)
        if obj < best_obj:
            best_f = f_opt
            best_obj = obj
    
    # Final polish with more iterations
    if time.time() < deadline - 3.0:
        f_polish, obj_polish = _local_search(best_f, deadline - 1.0, max_iters=50000)
        if obj_polish < best_obj:
            best_f = f_polish
            best_obj = obj_polish
    
    return best_f
# EVOLVE-BLOCK-END
