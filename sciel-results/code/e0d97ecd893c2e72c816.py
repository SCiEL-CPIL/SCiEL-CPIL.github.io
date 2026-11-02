import time
import numpy as np
from itertools import combinations, product
import sys
import math

def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U or len(U) < 2:
        return 0.0
    arr = np.array(U, dtype=np.int64)
    arr = np.unique(arr)
    n = len(arr)
    if n < 2:
        return 0.0
    mx = int(arr.max())
    if mx == 0:
        return 0.0
    
    N = 2 * mx + 1
    
    sig = np.zeros(N, dtype=np.float64)
    sig[arr] = 1.0
    
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
        
    F = np.fft.rfft(sig, n=fft_size)
    
    S_sum = np.fft.irfft(F * F, n=fft_size)
    sumset_size = np.sum(S_sum[:N] > 1e-6)
    
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

def _atom_efficiency(D):
    """Compute efficiency log(|D-D|/|D+D|) / log(2*max(D)+1) for atom D."""
    if not D or len(D) < 2:
        return 0.0, 0.0
    D = sorted(D)
    if D[0] != 0:
        return 0.0, 0.0
    s = max(D)
    if s == 0:
        return 0.0, 0.0
    sumset = set()
    diffset = set()
    for a in D:
        for b in D:
            sumset.add(a + b)
            diffset.add(a - b)
    r = len(diffset) / len(sumset)
    if r <= 1.0:
        return 0.0, r
    eff = np.log(r) / np.log(2 * s + 1)
    return eff, r

def _calc_max_u(K, D):
    """Compute max(U_K) for mixed-radix construction with K copies of atom D."""
    max_u = 0
    m = 1
    s_D = max(D)
    for i in range(K):
        max_u += s_D * m
        m = 2 * max_u + 1
    return max_u

def _search_small_sets(max_n=6, max_val=60, deadline=None):
    """Exhaustive search for small sets U to find optimal atoms."""
    best_score = 0.0
    best_U = [0, 1, 3]
    
    start_time = time.time()
    
    for n in range(3, max_n + 1):
        for combo in combinations(range(1, max_val + 1), n - 1):
            if deadline and time.time() > deadline:
                return best_U, best_score
                
            U = [0] + list(combo)
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
                
    return best_U, best_score

def _local_search(U, max_iters=100, deadline=None):
    """Improve set U via local search by moving individual elements."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 30
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        for i in range(len(U)):
            old_val = U[i]
            if old_val == 0:
                continue
            for new_val in range(1, upper_bound + 1):
                if new_val == old_val:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
            
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Step 1: Build a pool of candidate atoms
    atom_pool = []
    
    # Structured family 1: powers of 2
    for k in range(2, 8):
        D = [0] + [2**i for i in range(k)]
        eff, r = _atom_efficiency(D)
        if r > 1.0:
            atom_pool.append((eff, r, D))
    
    # Structured family 2: {0} ∪ {2^i - 1}
    for k in range(2, 8):
        D = [0] + [2**i - 1 for i in range(1, k+1)]
        eff, r = _atom_efficiency(D)
        if r > 1.0:
            atom_pool.append((eff, r, D))
    
    # Structured family 3: arithmetic progressions
    for a in range(2, 20):
        for k in range(2, 7):
            D = [0] + [a * i for i in range(1, k)]
            eff, r = _atom_efficiency(D)
            if r > 1.0:
                atom_pool.append((eff, r, D))
    
    # Structured family 4: small structured sets
    for a in range(1, 15):
        for b in range(a+1, 30):
            D = sorted(set([0, a, b, a+b, 2*a, 2*b]))
            if len(D) >= 2 and D[0] == 0:
                eff, r = _atom_efficiency(D)
                if r > 1.0:
                    atom_pool.append((eff, r, D))
    
    # Known good Golomb rulers
    known_gr = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 22],
        [0, 1, 4, 10, 12, 17, 22, 29],
    ]
    for D in known_gr:
        eff, r = _atom_efficiency(D)
        if r > 1.0:
            atom_pool.append((eff, r, D))
    
    # Exhaustive search for small atoms (size 2-4)
    for size in range(2, 5):
        span_limit = 40 if size <= 3 else 25
        for comb in combinations(range(1, span_limit + 1), size - 1):
            D = [0] + list(comb)
            eff, r = _atom_efficiency(D)
            if r > 1.0:
                atom_pool.append((eff, r, D))
    
    # Random search for additional atoms
    np.random.seed(42)
    for _ in range(8000):
        size = np.random.randint(2, 8)
        span_limit = 15 if size > 4 else 30
        D = [0] + list(np.random.choice(range(1, span_limit + 1), size=size-1, replace=False))
        D = sorted(D)
        eff, r = _atom_efficiency(D)
        if r > 1.0:
            atom_pool.append((eff, r, D))
    
    # Deduplicate and sort by efficiency
    seen = set()
    unique_pool = []
    for eff, r, D in sorted(atom_pool, key=lambda x: -x[0]):
        key = tuple(D)
        if key not in seen:
            seen.add(key)
            unique_pool.append((eff, r, D))
    
    if not unique_pool:
        return [0, 1]
    
    # Step 2: For each top atom, find optimal K and compute analytic score
    best_score = 0.0
    best_config = None
    
    top_atoms = unique_pool[:20]
    
    for eff, r_D, D in top_atoms:
        if time.time() > deadline - 25:
            break
        s_D = max(D)
        log_r_D = np.log(r_D)
        
        K_max = 1
        max_u = 0
        m = 1
        for K in range(1, 20):
            max_u_new = max_u + s_D * m
            if max_u_new > 10**7:
                K_max = K - 1
                break
            max_u = max_u_new
            m = 2 * max_u + 1
        else:
            K_max = 19
        
        for K in range(1, K_max + 1):
            if time.time() > deadline - 20:
                break
            max_u = _calc_max_u(K, D)
            if max_u > 10**7:
                continue
            denom = np.log(2 * max_u + 1)
            score = 1.0 + (K * log_r_D) / denom
            if score > best_score:
                best_score = score
                best_config = (D, K, score)
    
    if best_config is None:
        D, r_D, eff = unique_pool[0]
        return D
    
    best_D, best_K, best_analytic = best_config
    
    # Step 3: Construct U using best_K copies of best_D
    U = [0]
    M = 1
    for i in range(best_K):
        new_U = []
        for u in U:
            for d in best_D:
                new_U.append(u + d * M)
        U = new_U
        M = 2 * max(U) + 1
        
        if time.time() > deadline - 15:
            break
    
    # Step 4: Local search refinement on the constructed set
    if time.time() < deadline - 30 and len(U) <= 200:
        # Try local search on the full set if it's small enough
        U_refined, score_refined = _local_search(U, max_iters=50, deadline=deadline)
        if score_refined > best_analytic:
            U = U_refined
    
    # Step 5: Try replacing the last atom with other good atoms
    if time.time() < deadline - 25 and best_K >= 2:
        U_base = [0]
        M_base = 1
        for i in range(best_K - 1):
            new_U = []
            for u in U_base:
                for d in best_D:
                    new_U.append(u + d * M_base)
            U_base = new_U
            M_base = 2 * max(U_base) + 1
        
        best_U = U
        best_real_score = _score_set(U)
        
        for _, _, D_alt in top_atoms[:10]:
            if time.time() > deadline - 15:
                break
            U_trial = []
            for u in U_base:
                for d in D_alt:
                    U_trial.append(u + d * M_base)
            U_trial = sorted(set(U_trial))
            if len(U_trial) >= 2 and 0 in U_trial:
                s_trial = _score_set(U_trial)
                if s_trial > best_real_score:
                    best_real_score = s_trial
                    best_U = U_trial
        
        U = best_U
    
    # Step 6: Final local search polish if time permits
    if time.time() < deadline - 10 and len(U) <= 100:
        U, _ = _local_search(U, max_iters=30, deadline=deadline)
    
    return U

if __name__ == "__main__":
    s = construct()
    print(f"Result: {s[:10]}... Size: {len(s)}, Score: {_score_set(s)}")
