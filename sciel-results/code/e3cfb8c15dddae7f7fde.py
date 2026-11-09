"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import math
import time

# EVOLVE-BLOCK-START
def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _score(A, B):
    """Return distinct count if valid, else 0."""
    sums = [a + b for a in A for b in B]
    distinct = list(set(sums))
    n = len(distinct)
    for i in range(n):
        for j in range(i+1, n):
            if _gcd(distinct[i], distinct[j]) != 1:
                return 0
    return n

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin
    
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    s = _score(best_A, best_B)
    if s > 0:
        best_score = s
    
    rng = random.Random(12345)
    
    # Precompute gcd table for 1..256 (max sum is 128+128=256)
    max_val = 256
    gcd_table = [[0]*(max_val+1) for _ in range(max_val+1)]
    for i in range(1, max_val+1):
        for j in range(i, max_val+1):
            g = _gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def _score_fast(A, B):
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        n = len(distinct)
        for i in range(n):
            for j in range(i+1, n):
                if gcd_table[distinct[i]][distinct[j]] != 1:
                    return 0
        return n
    
    # Local search with random restarts
    for restart in range(50000):
        if time.time() > deadline:
            break
        
        # Random initialization
        A = list(rng.sample(range(1, 129), 3))
        B = list(rng.sample(range(1, 129), 3))
        
        local_best = _score_fast(A, B)
        
        # Local search: try single-element changes
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try modifying each element
            for set_idx in range(2):
                arr = A if set_idx == 0 else B
                for i in range(3):
                    old_val = arr[i]
                    # Try a subset of candidate values for speed
                    # Use random candidates + neighbors
                    candidates = set()
                    candidates.add(old_val)
                    for _ in range(10):
                        candidates.add(rng.randint(1, 128))
                    for dv in [-3,-2,-1,1,2,3]:
                        nv = old_val + dv
                        if 1 <= nv <= 128:
                            candidates.add(nv)
                    
                    for new_val in candidates:
                        if new_val == old_val:
                            continue
                        if new_val in arr:
                            continue
                        arr[i] = new_val
                        s = _score_fast(A, B)
                        if s > local_best:
                            local_best = s
                            improved = True
                        else:
                            arr[i] = old_val
                    arr[i] = old_val
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        
        if local_best > best_score:
            best_score = local_best
            best_A = list(A)
            best_B = list(B)
            if best_score >= 9:
                break
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
