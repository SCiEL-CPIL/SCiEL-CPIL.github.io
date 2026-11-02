"""Ternary covering code via Simulated Annealing with in-place coverage tracking."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius `radius`.
    Uses greedy init + simulated annealing with fast in-place coverage updates.
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    N = q ** n
    if N <= 0:
        return []
    
    all_words = np.zeros((N, n), dtype=np.int32)
    for idx in range(N):
        rem = idx
        for j in range(n - 1, -1, -1):
            all_words[idx, j] = rem % q
            rem //= q
            
    # Adjacency matrix
    dist = np.zeros((N, N), dtype=np.int8)
    for p in range(n):
        col = all_words[:, p]
        diff = col[:, np.newaxis] != col[np.newaxis, :]
        dist += diff.astype(np.int8)
    A = dist <= radius
    
    # Precompute A as list of boolean arrays for faster access
    A_list = [A[i] for i in range(N)]
    
    def greedy_init(seed):
        rng = np.random.RandomState(seed)
        uncovered = np.ones(N, dtype=bool)
        S = []
        S_set = set()
        
        while True:
            if time.time() > deadline - 30:
                break
            if not np.any(uncovered):
                break
            gain = A @ uncovered.astype(np.float32)
            gain += rng.uniform(0, 0.01, size=N)
            for c in S_set:
                gain[c] = -1.0
            best_c = int(np.argmax(gain))
            if gain[best_c] < 0.005:
                break
            S.append(best_c)
            S_set.add(best_c)
            uncovered &= ~A_list[best_c]
        return S
    
    def prune(S):
        S = list(S)
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            for i in range(len(S) - 1, -1, -1):
                if time.time() > deadline - 10:
                    break
                c = S[i]
                ok = True
                for v in np.where(A_list[c])[0]:
                    if not any(A_list[S[j]][v] for j in range(len(S)) if j != i):
                        ok = False
                        break
                if ok:
                    S.pop(i)
                    changed = True
        return S
    
    S = greedy_init(42)
    S = prune(S)
    best_S = S.copy()
    best_size = len(S)
    
    # SA with in-place updates
    def sa_run(S_init, max_time):
        rng = random.Random(999)
        S = list(S_init)
        cov = np.zeros(N, dtype=np.int32)
        for c in S:
            cov += A_list[c].astype(np.int32)
        
        T = 3.0
        T_min = 0.05
        cool = 0.997
        iters = 0
        max_iters = 100000
        
        while T > T_min and iters < max_iters and time.time() < deadline - 15:
            iters += 1
            if not S:
                break
            
            r = rng.random()
            accepted = False
            new_size = len(S)
            
            if r < 0.45:
                # Try remove
                i = rng.randrange(len(S))
                c = S[i]
                # Check: any vertex with cov==1 that c covers?
                if np.any((cov == 1) & A_list[c]):
                    continue
                # Safe removal
                cov -= A_list[c].astype(np.int32)
                S.pop(i)
                new_size -= 1
                accepted = True
                
            elif r < 0.7:
                # Try add
                S_set = set(S)
                # Pick random candidate
                c = rng.randrange(N)
                if c in S_set:
                    continue
                cov += A_list[c].astype(np.int32)
                S.append(c)
                new_size += 1
                # Accept with probability based on T (worse move)
                if rng.random() > np.exp(-1.0 / T):
                    # Reject
                    cov -= A_list[c].astype(np.int32)
                    S.pop()
                    accepted = False
                else:
                    accepted = True
                    
            else:
                # Swap
                if not S:
                    continue
                i = rng.randrange(len(S))
                c_rm = S[i]
                S_set = set(S)
                S_set.discard(c_rm)
                c_add = rng.randrange(N)
                if c_add in S_set:
                    continue
                # Test swap
                new_cov_test = cov - A_list[c_rm].astype(np.int32) + A_list[c_add].astype(np.int32)
                if np.any(new_cov_test < 1):
                    continue
                # Size unchanged, accept with moderate probability
                if rng.random() < 0.4:
                    cov = new_cov_test
                    S[i] = c_add
                    accepted = True
            
            if accepted and new_size < best_size:
                best_size = new_size
                best_S = S.copy()
            
            T *= cool
            
        return S, best_S, best_size
    
    S, best_S, best_size = sa_run(S, deadline - time.time() - 10)
    
    # Additional restarts
    for seed in [7, 13, 21]:
        if time.time() > deadline - 25:
            break
        S_try = greedy_init(seed)
        S_try = prune(S_try)
        if len(S_try) < best_size:
            best_S = S_try
            best_size = len(S_try)
        S_try, best_S, best_size = sa_run(S_try, deadline - time.time() - 10)
    
    best_S = prune(best_S)
    return [list(all_words[i]) for i in best_S]

# EVOLVE-BLOCK-END
