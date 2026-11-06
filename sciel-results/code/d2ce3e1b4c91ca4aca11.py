import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Generate all 4-subsets of [1, 16] as bitmasks and store original tuples
    candidates = []
    candidate_tuples = []
    for comb in combinations(range(1, 17), 4):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        candidates.append(mask)
        candidate_tuples.append(comb)
    
    N = len(candidates)  # 1820
    
    # Precompute pairwise intersections
    pair_int = [[0] * N for _ in range(N)]
    for i in range(N):
        ci = candidates[i]
        for j in range(i + 1, N):
            val = ci & candidates[j]
            pair_int[i][j] = val
            pair_int[j][i] = val

    def can_add(family, k):
        """
        Check if candidate k can be added to family without forming a sunflower.
        family: list of indices
        k: index
        """
        n_fam = len(family)
        ck = candidates[k]
        
        for i in range(n_fam):
            idx_a = family[i]
            ca = candidates[idx_a]
            aik = ca & ck
            
            for j in range(i + 1, n_fam):
                idx_b = family[j]
                cb = candidates[idx_b]
                
                ab = pair_int[idx_a][idx_b]
                if ab != aik:
                    continue
                
                bk = cb & ck
                if ab == bk:
                    return False
        return True

    best_family = []
    best_size = 0
    
    attempts = 0
    while time.time() < deadline - 2.0:
        attempts += 1
        random.seed(attempts)
        order = list(range(N))
        random.shuffle(order)
        
        family = []
        fam_set = set()
        
        for idx in order:
            if idx in fam_set:
                continue
            
            if can_add(family, idx):
                family.append(idx)
                fam_set.add(idx)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family.copy()
            
        # Local search / Second pass to fill gaps
        if best_size > 0 and time.time() < deadline - 5.0:
            current_best = best_family.copy()
            current_best_set = set(current_best)
            remaining = [i for i in range(N) if i not in current_best_set]
            random.shuffle(remaining)
            
            improved = False
            for idx in remaining:
                if can_add(current_best, idx):
                    current_best.append(idx)
                    current_best_set.add(idx)
                    improved = True
            
            if improved and len(current_best) > best_size:
                best_size = len(current_best)
                best_family = current_best

    # Convert best_family (list of indices) to a list of tuples
    # The evaluator expects a list or tuple of the subsets.
    result = []
    for idx in best_family:
        result.append(candidate_tuples[idx])
        
    return result
