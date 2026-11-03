import itertools
import math

def construct():
    def calc_score(u):
        if not u or len(u) < 2:
            return 0
        max_u = u[-1]
        
        sum_mask = 0
        diff_mask = 0
        
        for i in u:
            for j in u:
                sum_mask |= (1 << (i + j))
                diff_mask |= (1 << (i - j + max_u))
                
        num_sums = bin(sum_mask).count('1')
        num_diffs = bin(diff_mask).count('1')
        
        if num_sums == 0:
            return 0
            
        ratio = num_diffs / num_sums
        if ratio <= 1:
            return 0
            
        denom = math.log2(2 * max_u + 1)
        if denom == 0:
            return 0
            
        return 1 + math.log2(ratio) / denom

    best_score = 0
    best_set = [0, 1, 3, 7, 15]

    # Strategy: Try structured constructions that create difference collisions
    # 1. Two-block construction: {0..a} U {b, b+1, ..., b+c}
    # 2. Geometric-like: {0, 1, 2, 4, 8, ...}
    # 3. Sparse arithmetic: {0, d, 2d, ...} with perturbations
    # 4. Local search from good seeds

    # Test some structured candidates
    candidates = []
    
    # Two-block: first block [0..a], second block [b..b+c]
    for a in range(0, 8):
        for b in range(a+1, 30):
            for c in range(0, 8):
                if b + c <= a:
                    continue
                u = list(range(0, a+1)) + list(range(b, b+c+1))
                u = sorted(set(u))
                if 0 in u:
                    candidates.append(u)
    
    # Geometric-like
    for base in [2, 3, 4]:
        u = [0]
        val = 1
        while val < 50:
            u.append(val)
            val *= base
        candidates.append(u)
    
    # Sparse with collisions: {0, 1, 3, 9, 27, ...} (base 3)
    u = [0, 1, 3, 9, 27]
    candidates.append(u)
    u = [0, 1, 3, 9, 27, 81]
    candidates.append(u)
    
    # Try sets with many small differences
    # {0, 1, 2, 4, 8, 16}
    candidates.append([0, 1, 2, 4, 8, 16])
    candidates.append([0, 1, 2, 4, 8, 16, 32])
    
    # Difference-sparse: large gaps with small clusters
    for gap in [3, 4, 5, 6, 7, 8, 10, 12, 15]:
        for cluster_size in [2, 3, 4]:
            for num_clusters in [2, 3, 4, 5]:
                u = [0]
                for i in range(1, num_clusters):
                    start = i * gap
                    for j in range(cluster_size):
                        if start + j < 100:
                            u.append(start + j)
                u = sorted(set(u))
                if 0 in u:
                    candidates.append(u)
    
    # Evaluate all candidates
    for u in candidates:
        s = calc_score(u)
        if s > best_score:
            best_score = s
            best_set = u
    
    # Local search: try adding/removing elements from best set
    def local_search(start_set, max_iters=200):
        current = list(start_set)
        current_score = calc_score(current)
        for _ in range(max_iters):
            improved = False
            max_u = max(current)
            # Try adding elements
            for new_el in range(1, max_u + 10):
                if new_el not in current:
                    trial = sorted(current + [new_el])
                    s = calc_score(trial)
                    if s > current_score:
                        current = trial
                        current_score = s
                        improved = True
                        break
            # Try removing elements (except 0)
            if not improved:
                for el in current[1:]:
                    trial = [x for x in current if x != el]
                    if len(trial) >= 2:
                        s = calc_score(trial)
                        if s > current_score:
                            current = trial
                            current_score = s
                            improved = True
                            break
            if not improved:
                break
        return current, current_score
    
    # Run local search from best and some other good candidates
    for seed in [best_set] + candidates[:10]:
        ls_set, ls_score = local_search(seed, max_iters=100)
        if ls_score > best_score:
            best_score = ls_score
            best_set = ls_set
    
    return best_set
