import time
import math
import numpy as np

_START = time.time()
_DEADLINE = _START + 125.0

def _elapsed():
    return time.time() - _START

def _score_set(U):
    U = np.asarray(sorted(set(U)), dtype=np.int64)
    n = len(U)
    if n == 0:
        return 0.0
    if U[0] != 0:
        U = np.concatenate(([0], U))
        n = len(U)
    M = int(U[-1])
    if M < 1:
        return 0.0
    
    # Use FFT for sumset and diffset sizes
    vec = np.zeros(M + 1, dtype=np.float64)
    vec[U] = 1.0
    
    # Sumset U+U
    f = np.fft.rfft(vec, n=2 * (M + 1))
    s = np.fft.irfft(f * f, n=2 * (M + 1))
    sumset = int(np.sum(s >= 0.5))
    
    # Diffset U-U
    g = np.fft.irfft(f * np.conj(f), n=2 * (M + 1))
    diffset = int(np.sum(g >= 0.5))
    
    ratio = diffset / sumset if sumset > 0 else 0.0
    base = 2 * M + 1
    if base <= 1:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(base)

def _greedy_golomb(n_points, max_len=500):
    """Greedy construction of a Golomb-like ruler to maximize |U-U|."""
    U = [0]
    for i in range(n_points - 1):
        best_pos = None
        best_min_diff = 0
        for pos in range(1, max_len + 1):
            # Check all pairwise differences
            valid = True
            min_d = max_len + 1
            for a in U:
                for b in U:
                    d = abs(pos - a + b) if b > 0 else abs(pos - a)
                    # Actually check: new differences are pos - u for u in U
                    # and existing differences u_i - u_j
                    pass
            # Simpler: check that pos - u is not equal to any existing difference
            existing_diffs = set()
            for a in U:
                for b in U:
                    if a != b:
                        existing_diffs.add(abs(a - b))
            new_diffs = set(abs(pos - u) for u in U)
            if new_diffs & existing_diffs:
                valid = False
            if valid:
                # Prefer positions that create large new differences
                max_new = max(abs(pos - u) for u in U)
                if best_pos is None or max_new > best_min_diff:
                    best_pos = pos
                    best_min_diff = max_new
        if best_pos is not None:
            U.append(best_pos)
        else:
            break
    return sorted(U)

def _local_search(U, rng, time_limit=20):
    """Local search with add, remove, and swap operations."""
    U = list(U)
    cur_score = _score_set(U)
    start_t = _elapsed()
    
    while _elapsed() - start_t < time_limit:
        improved = False
        max_val = max(U) if U else 1
        
        # Try adding elements
        for _ in range(30):
            if _elapsed() - start_t > time_limit:
                break
            c = rng.randint(1, max_val * 3 + 1)
            if c not in U:
                trial = sorted(U + [c])
                s = _score_set(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    improved = True
        
        # Try removing elements
        if len(U) > 2:
            for idx in range(1, len(U)):
                if _elapsed() - start_t > time_limit:
                    break
                trial = U[:idx] + U[idx+1:]
                s = _score_set(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    improved = True
        
        # Try swap: replace one element with another
        if len(U) > 2:
            for _ in range(20):
                if _elapsed() - start_t > time_limit:
                    break
                idx = rng.randint(1, len(U))
                new_val = rng.randint(1, max_val * 2 + 1)
                if new_val not in U:
                    trial = sorted(U[:idx] + [new_val] + U[idx+1:])
                    s = _score_set(trial)
                    if s > cur_score + 1e-12:
                        cur_score = s
                        U = trial
                        improved = True
        
        if not improved:
            # Try more aggressive moves
            for _ in range(10):
                if _elapsed() - start_t > time_limit:
                    break
                # Random restart around current best
                U_new = list(U)
                for _ in range(3):
                    idx = rng.randint(1, len(U_new))
                    U_new[idx] = rng.randint(1, max_val * 2 + 1)
                U_new = sorted(set(U_new))
                if len(U_new) >= 2:
                    s = _score_set(U_new)
                    if s > cur_score + 1e-12:
                        cur_score = s
                        U = U_new
                        improved = True
            if not improved:
                break
    
    return U, cur_score

def construct():
    best_U = [0]
    best_score = 0.0
    rng = np.random.RandomState(42)
    
    # Candidate 1: Golomb ruler based constructions
    for n_pts in range(3, 12):
        if _elapsed() > 15:
            break
        U = _greedy_golomb(n_pts, max_len=300)
        if len(U) >= 2:
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U
    
    # Candidate 2: Known good Golomb rulers (from literature)
    known_golomb = [
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 9, 11, 17],
        [0, 1, 4, 9, 11, 17, 22],
        [0, 1, 4, 9, 11, 17, 22, 26],
        [0, 1, 4, 9, 11, 17, 22, 26, 32],
        [0, 1, 4, 9, 11, 17, 22, 26, 32, 37],
    ]
    for U in known_golomb:
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U
    
    # Candidate 3: Powers of 2 (baseline)
    for k in range(1, 15):
        U = [0] + [2**i for i in range(k)]
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U
    
    # Candidate 4: Scaled Golomb rulers
    for base_U in known_golomb[:5]:
        for a in [1, 2, 3, 5, 7, 10]:
            U = [a * u for u in base_U]
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U
    
    # Candidate 5: Random sparse sets with bias toward sparse structure
    for max_val in [20, 40, 80, 160, 320]:
        if _elapsed() > 40:
            break
        for size in range(3, min(15, max_val//5 + 1)):
            for _ in range(20):
                if _elapsed() > 45:
                    break
                # Generate sparse random set
                candidates = sorted(rng.choice(range(1, max_val+1), size=size, replace=False))
                U = sorted([0] + list(candidates))
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_U = U
    
    # Local search from best candidate
    U, cur_score = _local_search(list(best_U), rng, time_limit=25)
    if cur_score > best_score:
        best_score = cur_score
        best_U = U
    
    # Multiple restarts with local search
    for restart in range(5):
        if _elapsed() > 105:
            break
        
        # Generate diverse initial candidates
        max_val = max(best_U) if best_U else 50
        size = len(best_U)
        
        # Perturb best solution
        U_init = list(best_U)
        for _ in range(rng.randint(1, 4)):
            idx = rng.randint(1, len(U_init))
            U_init[idx] = rng.randint(1, max_val * 2 + 1)
        U_init = sorted(set(U_init))
        if len(U_init) < 2:
            U_init = list(best_U)
        
        U, cur_score = _local_search(U_init, rng, time_limit=12)
        if cur_score > best_score:
            best_score = cur_score
            best_U = U
    
    # Final polish: intensive local search on best
    U, cur_score = _local_search(list(best_U), rng, time_limit=15)
    if cur_score > best_score:
        best_score = cur_score
        best_U = U
    
    return best_U
