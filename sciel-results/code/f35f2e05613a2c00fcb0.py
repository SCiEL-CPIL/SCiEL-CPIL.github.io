import time
import numpy as np
import random
from itertools import combinations

def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U or len(U) < 2:
        return 0.0
    
    U_arr = np.array(U, dtype=np.int64)
    U_arr = U_arr[U_arr >= 0]
    if 0 not in U_arr:
        U_arr = np.concatenate(([0], U_arr))
    U_arr = np.unique(U_arr)
    
    n = len(U_arr)
    if n < 2:
        return 0.0
        
    mx = int(U_arr.max())
    if mx == 0:
        return 0.0
    
    N = 2 * mx + 1
    
    # Efficient counting via FFT
    sig = np.zeros(N, dtype=np.float64)
    sig[U_arr] = 1.0
    
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
        
    F = np.fft.rfft(sig, n=fft_size)
    
    # Sumset U+U
    S_sum = np.fft.irfft(F * F, n=fft_size)
    sumset_size = np.sum(S_sum[:N] > 1e-6)
    
    # Diffset U-U
    flip_sig = np.zeros(N, dtype=np.float64)
    flip_sig[:] = sig[::-1]
    
    F_flip = np.fft.rfft(flip_sig, n=fft_size)
    S_diff = np.fft.irfft(F * F_flip, n=fft_size)
    
    diffset_size = np.sum(S_diff > 1e-6)
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
        
    denom = np.log(2 * mx + 1)
    if denom == 0:
        return 0.0
        
    return 1.0 + np.log(ratio) / denom

def _score_set_small(U):
    """Fast scoring for small sets using direct computation."""
    U_set = set(U)
    n = len(U_set)
    if n < 2:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 0.0
    
    sumset = set()
    diffset = set()
    for a in U_set:
        for b in U_set:
            sumset.add(a + b)
            diffset.add(a - b)
            
    if not sumset:
        return 0.0
        
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
        
    import math
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 0.0
        
    return 1.0 + log_ratio / log_denom

def _greedy_mian_chowla(max_elements, max_value=10**6):
    """
    Generates a Mian-Chowla sequence.
    U[0] = 0.
    U[i] is the smallest integer > U[i-1] such that all pairwise sums 
    U[j] + U[i] (for j < i) are distinct from all previous pairwise sums.
    """
    U = [0]
    used_sums = {0}
    
    while len(U) < max_elements:
        candidate = U[-1] + 1
        while True:
            valid = True
            for u in U:
                s = candidate + u
                if s in used_sums:
                    valid = False
                    break
                if s > max_value:
                    valid = False
                    break
            if valid:
                U.append(candidate)
                for u in U[:-1]:
                    used_sums.add(candidate + u)
                break
            candidate += 1
            if candidate > max_value:
                return U
    return U

def _known_good_structures():
    """Return a list of known good candidate sets."""
    candidates = []
    
    # Known optimal Golomb rulers
    known_gr = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 22],
        [0, 1, 4, 10, 12, 17, 22, 29],
        [0, 1, 4, 10, 12, 17, 22, 29, 36],
        [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
    ]
    candidates.extend(known_gr)
    
    # Mian-Chowla sequences of various lengths
    for L in [3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50]:
        try:
            seq = _greedy_mian_chowla(L, max_value=10**5)
            if len(seq) > 1:
                candidates.append(seq)
        except Exception:
            pass
    
    # Sets of form {0, 1, 3, 7, 15, ...} (2^k - 1)
    for k in range(2, 12):
        U = [2**i - 1 for i in range(k)]
        candidates.append(U)
    
    # "Almost AP" structures: {0, 1, ..., k-2, M}
    for k in range(3, 10):
        for M in range(k, 60):
            U = list(range(k - 1)) + [M]
            if len(set(U)) == len(U):
                candidates.append(U)
    
    # Randomized perturbations of AP
    random.seed(42)
    for _ in range(100):
        n = random.randint(3, 8)
        base_ap = list(range(n))
        i = random.randint(1, n - 1)
        new_val = random.randint(1, 60)
        U = base_ap[:]
        U[i] = new_val
        if len(set(U)) == len(U) and 0 in U:
            candidates.append(sorted(set(U)))
    
    return candidates

def _exhaustive_small_search(max_n=7, max_val=50, deadline=None):
    """Exhaustive search for small sets U containing 0."""
    best_score = 0.0
    best_U = [0, 1, 3]
    
    start_time = time.time()
    
    for n in range(3, max_n + 1):
        for combo in combinations(range(1, max_val + 1), n - 1):
            if deadline and time.time() > deadline:
                return best_U, best_score
                
            U = [0] + list(combo)
            U.sort()
            
            s = _score_set_small(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
                
    return best_U, best_score

def _local_search(U, time_limit=5.0):
    """
    Attempts to improve the set U via local moves using FFT-based scoring.
    Moves:
    1. Add a new element x (x > max(U)) if it improves score.
    2. Remove an element x (x > 0) if it improves score.
    3. Swap: Replace element x with y (y not in U).
    """
    if max(U) > 50000:
        return U, _score_set(U)
        
    U = sorted(list(set(U)))
    current_score = _score_set(U)
    start_ls = time.time()
    steps = 0
    max_steps = 200
    
    while steps < max_steps and time.time() - start_ls < time_limit:
        improved = False
        steps += 1
        
        # Move 1: Add a new element
        mx = max(U)
        candidates_add = []
        for delta in range(1, 15):
            candidates_add.append(mx + delta)
        for _ in range(5):
            candidates_add.append(mx + random.randint(1, 50))
        
        for cand in candidates_add:
            if cand in U:
                continue
            new_U = U + [cand]
            s = _score_set(new_U)
            if s > current_score + 1e-9:
                U = new_U
                current_score = s
                improved = True
                break
        
        # Move 2: Remove an element
        if len(U) > 2:
            candidates_remove = [max(U)]
            for _ in range(3):
                candidates_remove.append(random.choice(U[1:]))
            
            for cand in candidates_remove:
                if cand == 0:
                    continue
                new_U = [x for x in U if x != cand]
                if len(new_U) < 2:
                    continue
                s = _score_set(new_U)
                if s > current_score + 1e-9:
                    U = new_U
                    current_score = s
                    improved = True
                    break
        
        # Move 3: Swap an element
        if len(U) > 2 and not improved:
            for i in range(1, len(U)):
                old_val = U[i]
                # Try replacing with values near old_val or near max
                for new_val in range(max(1, old_val - 5), min(max(U) + 10, old_val + 10)):
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U[:]
                    new_U[i] = new_val
                    new_U = sorted(set(new_U))
                    if len(new_U) != len(U):
                        continue
                    s = _score_set(new_U)
                    if s > current_score + 1e-9:
                        U = new_U
                        current_score = s
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
                
    return U, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Phase 1: Exhaustive search for small sets
    try:
        U_sol, s_sol = _exhaustive_small_search(max_n=6, max_val=45, deadline=deadline)
        if s_sol > best_score:
            best_score = s_sol
            best_set = U_sol
    except Exception:
        pass
    
    # Phase 2: Evaluate known good structures
    candidates = _known_good_structures()
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    scored_candidates = []
    for seq in unique_candidates:
        if time.time() > deadline:
            break
        try:
            s = _score_set(seq)
            scored_candidates.append((s, seq))
        except Exception:
            pass
    
    # Sort by score descending
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Update best from known structures
    for s_init, seq in scored_candidates[:20]:
        if s_init > best_score:
            best_score = s_init
            best_set = seq
    
    # Phase 3: Local search on top candidates
    for s_init, seq in scored_candidates[:15]:
        if time.time() > deadline - 15.0:
            break
            
        mx = max(seq)
        if mx > 100000:
            ls_time = 1.0
        elif mx > 20000:
            ls_time = 2.0
        else:
            ls_time = 4.0
            
        try:
            refined_seq, s_refined = _local_search(seq, time_limit=ls_time)
            if s_refined > best_score:
                best_score = s_refined
                best_set = refined_seq
        except Exception:
            pass
            
        if s_init > best_score:
            best_score = s_init
            best_set = seq

    # Phase 4: If time remains, do more local search iterations on best set
    if time.time() < deadline - 5.0:
        current_best = list(best_set)
        current_best_score = best_score
        
        for _ in range(30):
            if time.time() > deadline - 5.0:
                break
            if len(current_best) < 2:
                break
            
            # Random perturbation: swap a random element
            if len(current_best) > 1:
                idx = random.randint(1, len(current_best)-1)
                removed = current_best[idx]
                
                mx = max(current_best) if current_best else 0
                new_val = mx + random.randint(-5, 15)
                if new_val < 1:
                    new_val = 1
                if new_val not in current_best:
                    current_best[idx] = new_val
                    current_best = sorted(current_best)
                    
                    s = _score_set(current_best)
                    if s > current_best_score:
                        current_best_score = s
                        best_score = s
                        best_set = list(current_best)
                    else:
                        current_best[idx] = removed
                        current_best = sorted(current_best)

    return best_set

if __name__ == "__main__":
    s = construct()
    print(f"Result: {s[:15]}... Size: {len(s)}, Max: {max(s)}, Score: {_score_set(s)}")
