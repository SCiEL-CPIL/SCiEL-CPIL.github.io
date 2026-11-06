import time
import random
import math

MAX_MOD = 128

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    moduli = list(range(2, MAX_MOD + 1, 2))
    n = len(moduli)
    
    # Precompute gcds
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            g = math.gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g
            
    def _is_compatible(r, m_idx, assigned):
        # assigned is dict m_idx -> r
        for m2_idx, r2 in assigned.items():
            g = gcds[m_idx][m2_idx]
            if (r - r2) % g == 0:
                return False
        return True

    def _greedy_full(seed):
        random.seed(seed)
        indices = list(range(n))
        random.shuffle(indices)
        assigned = {}
        for idx in indices:
            m = moduli[idx]
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                if _is_compatible(r, idx, assigned):
                    assigned[idx] = r
                    break
            else:
                return None
        return [[assigned[i], moduli[i]] for i in range(n)]

    def _greedy_subset(seed):
        random.seed(seed)
        indices = list(range(n))
        random.shuffle(indices)
        assigned = {}
        for idx in indices:
            m = moduli[idx]
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                if _is_compatible(r, idx, assigned):
                    assigned[idx] = r
                    break
        return [[assigned[i], moduli[i]] for i in assigned]

    best_result = None
    best_size = 0
    attempts = 0
    
    # Phase 1: Full
    while time.time() < deadline:
        attempts += 1
        res = _greedy_full(attempts)
        if res is not None:
            return res
        if attempts >= 200:
            break

    # Phase 2: Subset
    while time.time() < deadline:
        attempts += 1
        res = _greedy_subset(attempts)
        size = len(res)
        if size > best_size:
            best_size = size
            best_result = res
            if best_size == n:
                return best_result

    if best_result is None:
        return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
        
    return best_result
