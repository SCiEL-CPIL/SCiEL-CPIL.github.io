import numpy as np
import time
import random

NMAX = 256
UNIVERSE = np.array(list(range(2, NMAX + 1)), dtype=np.int64)

def _can_add(x, S):
    """
    Check if x can be added to S (numpy array of ints).
    S must not contain x.
    """
    if len(S) == 0:
        return True
    if len(S) == 1:
        return True
    
    # 1. Check x | b*c for all b,c in S, b != c
    # prod = np.outer(S, S)
    # mask = (prod % x == 0)
    # We need to check off-diagonals.
    # Instead of full outer product, we can do:
    # For each b in S, check if there exists c in S, c != b such that x | b*c.
    # This is equivalent to: for each b, (S % (x // gcd(b,x))) == 0 ?
    # No, x | b*c  <=>  (x / gcd(b,x)) | c.
    
    # Let's use the vectorized approach for simplicity and speed, 
    # but be careful with memory if S is large. S is small (~50).
    
    S_arr = S
    # Check x | b*c
    # We can compute b*c % x == 0
    # To avoid large products, we can use modular arithmetic.
    # But b,c <= 256, b*c <= 65536. Fits in int32.
    
    # Create meshgrid
    # B, C = np.meshgrid(S_arr, S_arr)
    # This creates N x N arrays.
    
    # Alternative:
    # For each i, j with i != j:
    # if (S_arr[i] * S_arr[j]) % x == 0: return False
    
    # Vectorized:
    # prod = np.outer(S_arr, S_arr)
    # mask = (prod % x == 0)
    # np.fill_diagonal(mask, False)
    # if mask.any(): return False
    
    # Check b | x*c
    # For each i, j with i != j:
    # if (x * S_arr[j]) % S_arr[i] == 0: return False
    
    # Vectorized:
    # term = x * S_arr[None, :]  # 1 x N
    # mask2 = (term % S_arr[:, None]) == 0  # N x N
    # np.fill_diagonal(mask2, False)
    # if mask2.any(): return False
    
    # Check c | x*b
    # This is symmetric to b | x*c, so mask2 covers it?
    # b | x*c  <=>  S[i] | x*S[j]
    # c | x*b  <=>  S[j] | x*S[i]
    # These are different.
    # mask2[i,j] is S[i] | x*S[j].
    # We also need S[j] | x*S[i].
    # This is mask2.T.
    # So if mask2.any() or mask2.T.any(), but we already filled diagonal.
    # Actually, if mask2[i,j] is true, it's a violation.
    # If mask2[j,i] is true, it's a violation.
    # So checking mask2.any() is sufficient?
    # No, mask2.any() checks if ANY [i,j] is true.
    # If [j,i] is true, it is also in mask2 (at position j,i).
    # So yes, mask2.any() covers both.
    
    # Let's implement carefully.
    
    if len(S_arr) < 2:
        return True
        
    # 1. x | b*c
    # We can optimize this.
    # x | b*c  <=>  (x / gcd(b,x)) | c
    # Let's just use the product method for simplicity, it's fast enough for N=50.
    
    B, C = np.meshgrid(S_arr, S_arr, indexing='ij')
    # B[i,j] = S[i], C[i,j] = S[j]
    
    # Check x | B*C
    # (B * C) % x == 0
    # To avoid overflow (though not an issue here), we can do:
    # But B*C is small.
    mask1 = (B * C) % x == 0
    np.fill_diagonal(mask1, False)
    if mask1.any():
        return False
        
    # Check B | x * C
    # (x * C) % B == 0
    mask2 = (x * C) % B == 0
    np.fill_diagonal(mask2, False)
    if mask2.any():
        return False
        
    # Check C | x * B
    # (x * B) % C == 0
    mask3 = (x * B) % C == 0
    np.fill_diagonal(mask3, False)
    if mask3.any():
        return False
        
    return True

def _greedy_numpy(deadline):
    S = []
    S_set = set()
    candidates = list(UNIVERSE)
    
    # Sort candidates: Primes first?
    # Let's just shuffle and pick best?
    # Or greedy by "least conflict".
    
    # For now, random order
    random.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline:
            break
        if x in S_set:
            continue
        S_arr = np.array(S, dtype=np.int64)
        if _can_add(x, S_arr):
            S.append(x)
            S_set.add(x)
            
    return S

def _local_search_numpy(S, deadline, max_iters=10000, rng_seed=42):
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(rng_seed)
    
    candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # 1. Try to add
        if candidates:
            x = rng.choice(candidates)
            S_arr = np.array(S, dtype=np.int64)
            if _can_add(x, S_arr):
                S.append(x)
                S_set.add(x)
                candidates.remove(x)
                if len(S) > best_size:
                    best = S[:]
                    best_size = len(S)
                continue # Restart loop? Or just continue?
        
        # 2. Try to swap
        if S:
            idx_remove = rng.randrange(len(S))
            removed = S[idx_remove]
            
            S_temp = S[:]
            S_temp.pop(idx_remove)
            S_temp_set = S_set - {removed}
            
            if candidates:
                x = rng.choice(candidates)
                S_temp_arr = np.array(S_temp, dtype=np.int64)
                if _can_add(x, S_temp_arr):
                    # Accept?
                    # New size is same.
                    # In SA, we might accept same size to explore.
                    # But we want to maximize size.
                    # If we accept same size, we might get stuck.
                    # Let's only accept if we can then add?
                    # Or use SA on size.
                    
                    # Let's try to add after swap.
                    # If we swap, size is same.
                    # We should prefer moves that increase size.
                    # If we do a swap, we should check if we can add another element.
                    
                    # For now, let's just do swaps to escape local optima.
                    # Accept with probability based on T?
                    # Since size is same, delta = 0. Accept always?
                    # This leads to random walk.
                    
                    S = S_temp + [x]
                    S_set = S_temp_set | {x}
                    if removed not in candidates:
                        candidates.append(removed)
                    if x in candidates:
                        candidates.remove(x)
                    
                    # Try to add after swap
                    if candidates:
                        y = rng.choice(candidates)
                        S_arr = np.array(S, dtype=np.int64)
                        if _can_add(y, S_arr):
                            S.append(y)
                            S_set.add(y)
                            candidates.remove(y)
                            if len(S) > best_size:
                                best = S[:]
                                best_size = len(S)
                    continue
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_S = []
    best_size = 0
    
    # Multiple restarts
    for seed in range(20):
        if time.time() > deadline - 5:
            break
            
        rng = random.Random(seed)
        
        # Greedy Init
        S_init = _greedy_numpy(deadline)
        
        if len(S_init) > best_size:
            best_S = S_init
            best_size = len(S_init)
            
        # Local Search
        S_improved = _local_search_numpy(S_init, deadline - 5, max_iters=5000, rng_seed=seed)
        
        if len(S_improved) > best_size:
            best_S = S_improved
            best_size = len(S_improved)
            
    if not best_S:
        best_S = [2, 3]
        
    return [int(x) for x in best_S]
