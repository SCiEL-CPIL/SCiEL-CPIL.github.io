"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy import signal


def _autoconv_excl_lag0(f):
    """Compute autoconvolution excluding lag 0"""
    ac = np.convolve(f, f, mode='full')
    # Exclude lag 0 (index N-1 in full convolution)
    n = len(f)
    ac_no_lag0 = np.concatenate([ac[:n-1], ac[n:]])
    return np.max(np.abs(ac_no_lag0))


def _compute_objective(f):
    """Compute max |autoconv|_{k!=0} / (sum f)^2"""
    S = np.sum(f)
    if abs(S) < 1e-12:
        return np.inf
    max_ac = _autoconv_excl_lag0(f)
    return max_ac / (S * S)


def _walsh_hadamard(n):
    """Generate Walsh-Hadamard sequence of length n (must be power of 2)"""
    if n <= 0 or (n & (n - 1)) != 0:
        return None
    f = np.array([1.0])
    while len(f) < n:
        f = np.concatenate([f, f, f, -f])
    return f[:n]


def _generate_wh_candidate(N):
    """Generate a candidate based on Walsh-Hadamard with padding"""
    # Find largest power of 2 <= N
    p = 1
    while p * 2 <= N:
        p *= 2
    
    wh = _walsh_hadamard(p)
    if wh is None:
        return np.ones(N)
    
    # Extend to N by repeating the pattern or padding
    f = np.zeros(N)
    for i in range(N):
        f[i] = wh[i % p]
    
    # Ensure sum is positive
    if np.sum(f) < 0:
        f = -f
    
    return f


def _local_search(f, deadline, max_iters=50000):
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
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    np.random.seed(42)
    
    # Try multiple initializations
    candidates = []
    
    # 1. Walsh-Hadamard based
    f_wh = _generate_wh_candidate(N)
    candidates.append(f_wh)
    
    # 2. All ones
    candidates.append(np.ones(N))
    
    # 3. Random ±1 sequences
    for _ in range(5):
        f_rand = np.random.choice([-1.0, 1.0], size=N)
        if np.sum(f_rand) < 0:
            f_rand = -f_rand
        candidates.append(f_rand)
    
    best_f = candidates[0].copy()
    best_obj = _compute_objective(best_f)
    
    # Local search on each candidate
    for i, f_init in enumerate(candidates):
        if time.time() > deadline - 5.0:
            break
        
        f_opt, obj = _local_search(f_init, deadline, max_iters=30000)
        if obj < best_obj:
            best_f = f_opt
            best_obj = obj
    
    # Final polish with more iterations
    if time.time() < deadline - 3.0:
        f_polish = best_f.copy()
        obj_polish = best_obj
        for _ in range(20000):
            if time.time() > deadline - 1.0:
                break
            idx = np.random.randint(N)
            trial = f_polish.copy()
            trial[idx] = -trial[idx]
            obj = _compute_objective(trial)
            if obj < obj_polish:
                f_polish = trial
                obj_polish = obj
        if obj_polish < best_obj:
            best_f = f_polish
            best_obj = obj_polish
    
    return best_f
# EVOLVE-BLOCK-END
