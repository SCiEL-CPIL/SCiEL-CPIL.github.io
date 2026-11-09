import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 20
    max_val = 10000
    
    # Precompute all pairs (i, j) with i <= j for n=20
    # There are n*(n+1)/2 = 210 pairs
    pairs_i = np.array([i for i in range(n) for j in range(i, n)], dtype=np.int32)
    pairs_j = np.array([j for i in range(n) for j in range(i, n)], dtype=np.int32)
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A using NumPy vectorization."""
        A_arr = np.array(A, dtype=np.int64)
        sums = A_arr[pairs_i] + A_arr[pairs_j]
        prods = A_arr[pairs_i] * A_arr[pairs_j]
        sumset_size = len(np.unique(sums))
        prodset_size = len(np.unique(prods))
        return max(sumset_size, prodset_size)

    def is_valid_set(A):
        if len(A) != n:
            return False
        if len(set(A)) != n:
            return False
        if min(A) < 1 or max(A) > max_val:
            return False
        return True

    # Strategy: Model A as union of two structured sub-sets.
    # Sub-set 1: Geometric-like progression (good for products)
    # Sub-set 2: Arithmetic-like progression (good for sums)
    # We will enumerate coarse parameters for these structures and then refine.
    
    best_score = float('inf')
    best_set = None
    
    # Helper to generate a candidate set from parameters
    # params: (geo_start, geo_ratio, geo_len, arith_start, arith_step, arith_len)
    # We need to ensure the union has exactly n distinct elements.
    # We'll generate a superset and pick the best n? No, the task requires exactly 20 distinct.
    # So we must construct exactly 20.
    
    # Let's try a mixed approach:
    # 1. Enumerate geometric progressions (ratio r, start s, length k)
    # 2. For the remaining n-k elements, try to fill with arithmetic progression or smooth numbers
    # 3. Or, simply generate candidates from these families and use a fast local search to refine.
    
    # Family 1: Pure Geometric Progression (integer approximations)
    # Family 2: Pure Arithmetic Progression
    # Family 3: Hybrid - First half geometric, second half arithmetic
    # Family 4: Smooth numbers (from inspiration 1, but we'll integrate it better)
    
    def get_smooth_pool():
        pool = set()
        for a in range(0, 14):
            v2 = 1 << a
            if v2 > max_val: break
            for b in range(0, 9):
                v3 = v2 * (3 ** b)
                if v3 > max_val: break
                for c in range(0, 6):
                    v5 = v3 * (5 ** c)
                    if v5 > max_val: break
                    for d in range(0, 5):
                        v7 = v5 * (7 ** d)
                        if v7 > max_val: break
                        pool.add(v7)
        return sorted(pool)
    
    smooth_pool = get_smooth_pool()
    
    # Generate initial candidates from structured families
    candidates = []
    
    # 1. Smallest smooth numbers
    if len(smooth_pool) >= n:
        candidates.append(smooth_pool[:n])
        
    # 2. Geometric progressions with integer steps
    # Try ratios like 1.1, 1.2, 1.5, 2.0
    for r in [1.1, 1.2, 1.5, 2.0, 3.0]:
        for start in [1, 2, 5, 10]:
            seq = []
            curr = float(start)
            for _ in range(n):
                val = int(round(curr))
                if val < 1: val = 1
                if val > max_val: break
                if val not in seq:
                    seq.append(val)
                curr *= r
            if len(seq) == n:
                candidates.append(seq)
                
    # 3. Arithmetic progressions
    for start in [1, 2, 5, 10, 50]:
        for step in [1, 2, 5, 10, 50]:
            seq = [start + i * step for i in range(n)]
            if max(seq) <= max_val:
                candidates.append(seq)
                
    # 4. Hybrid: First 10 geometric, next 10 arithmetic
    # This is tricky to ensure distinctness and coverage. 
    # Instead, let's use a "seeded" local search on these candidates.
    
    # Evaluate initial candidates
    for cand in candidates:
        if not is_valid_set(cand):
            continue
        score = evaluate(cand)
        if score < best_score:
            best_score = score
            best_set = cand[:]
            
    # If no valid candidate was found (unlikely), fallback to random
    if best_set is None:
        best_set = random.sample(range(1, max_val + 1), n)
        best_score = evaluate(best_set)
        
    current_set = best_set[:]
    current_score = best_score
    
    # Local Search / Simulated Annealing
    # We will use a more targeted mutation: 
    # - Swap with a nearby number
    # - Adjust by a small delta
    # - Replace with a smooth number if not already
    
    T = 50.0
    T_min = 0.1
    cooling = 0.9995
    
    best_set_sa = current_set[:]
    best_score_sa = current_score
    
    max_evals = 500000  # Higher budget allowed due to simpler evaluation logic? No, eval is same. 
                        # But we start from a much better point.
    
    eval_count = 0
    
    while T > T_min and time.time() < deadline - 5 and eval_count < max_evals:
        # Pick a random element to change
        i = random.randint(0, n - 1)
        old_val = current_set[i]
        
        # Generate a new value
        # Strategy: 
        # 1. 20% chance: pick a random smooth number
        # 2. 30% chance: local perturbation +/- small delta
        # 3. 50% chance: random uniform in [1, max_val]
        
        r = random.random()
        if r < 0.2 and smooth_pool:
            new_val = random.choice(smooth_pool)
        elif r < 0.5:
            delta = random.randint(1, 20)
            sign = random.choice([-1, 1])
            new_val = old_val + sign * delta
            if new_val < 1: new_val = 1
            if new_val > max_val: new_val = max_val
        else:
            new_val = random.randint(1, max_val)
            
        if new_val == old_val:
            continue
            
        # Check distinctness
        if new_val in current_set:
            continue
            
        # Apply change
        current_set[i] = new_val
        
        # Evaluate
        new_score = evaluate(current_set)
        eval_count += 1
        
        delta_score = new_score - current_score
        
        if delta_score < 0 or random.random() < math.exp(-delta_score / T):
            # Accept
            current_score = new_score
            if new_score < best_score_sa:
                best_score_sa = new_score
                best_set_sa = current_set[:]
        else:
            # Reject
            current_set[i] = old_val
            
        T *= cooling
        
    return sorted(best_set_sa)
