import time
import random
from itertools import combinations
from collections import Counter

def max_sidon_size(S):
    """
    Computes the size of the largest subset of S that is a 'strong Sidon' set.
    A strong Sidon set has all unordered pair sums (including diagonals) unique.
    """
    S = sorted(S)
    n = len(S)
    if n <= 1:
        return n

    # Precompute pair sums
    # pair_sum[i][j] = S[i] + S[j] for i <= j
    # We need to check if a subset is valid.
    # A subset is valid if all sums S[i]+S[j] for i,j in subset, i<=j are distinct.
    
    # Branch and Bound approach
    # State: current subset (list of indices), last index added
    # We want to maximize the size of the subset.
    
    best = 0
    # Sort S to help with pruning? Not strictly necessary for B&B, but good for consistency.
    
    # Precompute all pair sums for the full set S
    # sums[i][j] = S[i] + S[j]
    sums = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            sums[i][j] = S[i] + S[j]
            sums[j][i] = sums[i][j] # Symmetry, though we only use i<=j

    # We will use DFS to find the maximum size.
    # To speed up, we can check if adding a new element x to current set C is valid.
    # x is valid if:
    # 1. 2*x is not in {S[i]+S[j] for i,j in C}
    # 2. x+S[k] is not in {S[i]+S[j] for i,j in C} for all k in C
    
    # Let's maintain a set of used sums.
    
    def dfs(start_idx, current_size, used_sums):
        nonlocal best
        if current_size > best:
            best = current_size
        
        # Pruning: if we can't beat best, return
        # Max possible size is current_size + (n - start_idx)
        if current_size + (n - start_idx) <= best:
            return

        for i in range(start_idx, n):
            x = S[i]
            # Check if x can be added to the current subset
            # We need to check against all pairs in the current subset?
            # No, we just need to check if x creates a duplicate sum with any existing pair.
            # The existing pairs are those formed by elements in the current subset.
            # But we don't have the current subset elements explicitly in this state, only used_sums.
            # This is a problem: used_sums contains sums of pairs from the current subset.
            # When we add x, we form new pairs: (x,x) and (x, y) for all y in current subset.
            # We need to know the elements in the current subset to check (x,y).
            
            # So we must pass the current subset elements or indices.
            # This makes the state space larger.
            # For n=12, depth is at most 12.
            # We can pass a tuple of indices.
            
            # Revised DFS:
            # dfs(start_idx, current_indices, used_sums)
            
            # This will be implemented below.
            pass

    # Let's rewrite DFS properly
    def dfs(start_idx, current_indices, used_sums):
        nonlocal best
        current_size = len(current_indices)
        if current_size > best:
            best = current_size
            
        # Pruning
        if current_size + (n - start_idx) <= best:
            return

        for i in range(start_idx, n):
            x = S[i]
            # Check if x is compatible with current_indices
            valid = True
            
            # Check diagonal: 2*x
            if 2*x in used_sums:
                valid = False
            
            # Check off-diagonals: x + S[j] for j in current_indices
            if valid:
                for j_idx in current_indices:
                    s = x + S[j_idx]
                    if s in used_sums:
                        valid = False
                        break
            
            if valid:
                # Add x
                new_indices = current_indices + (i,)
                new_used_sums = set(used_sums)
                new_used_sums.add(2*x)
                for j_idx in current_indices:
                    new_used_sums.add(x + S[j_idx])
                
                dfs(i + 1, new_indices, new_used_sums)

    dfs(0, (), set())
    return best

def _valid(c):
    if len(c) != 12 or len(set(c)) != 12:
        return False
    return all(0 <= x <= 128 for x in c)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_set = None
    best_val = 99
    
    # Initialize with a known good candidate (consecutive)
    init_set = list(range(12))
    best_val = max_sidon_size(init_set)
    best_set = init_set.copy()
    
    # Local search to improve
    # We want to minimize max_sidon_size.
    # Strategy: Try to swap elements to create more sum collisions.
    # A good proxy for "more collisions" is minimizing the number of distinct pair sums.
    # Or directly minimizing the max Sidon size, but that's expensive.
    
    # Let's use a hybrid: 
    # 1. Generate many random sets with small diameter.
    # 2. Evaluate them.
    # 3. Take the best few and do local search.
    
    rng = random.Random(42)
    
    # Phase 1: Random search in small windows
    # Sets with small diameter are more likely to have small max Sidon subsets.
    # Diameter D means max-min = D.
    # Number of pairs = 78.
    # If D is small, collisions are frequent.
    
    candidates = []
    
    # Add the initial consecutive set
    candidates.append(init_set)
    
    # Generate random sets with diameter between 12 and 30
    for _ in range(5000):
        if time.time() > deadline - 10:
            break
        lo = rng.randint(0, 128 - 30)
        hi = lo + rng.randint(12, 30)
        if hi > 128:
            hi = 128
        if hi - lo < 11:
            continue
        c = sorted(rng.sample(range(lo, hi + 1), 12))
        if _valid(c):
            candidates.append(c)
    
    # Evaluate candidates
    # To speed up, we can first check a cheap proxy: number of distinct sums.
    # Sets with fewer distinct sums are more likely to have smaller max Sidon subsets.
    
    scored_candidates = []
    for c in candidates:
        if time.time() > deadline - 5:
            break
        # Cheap proxy
        sums = set()
        for i in range(12):
            for j in range(i, 12):
                sums.add(c[i] + c[j])
        num_distinct_sums = len(sums)
        scored_candidates.append((num_distinct_sums, c))
        
    # Sort by proxy (fewer distinct sums is better)
    scored_candidates.sort(key=lambda x: x[0])
    
    # Evaluate the top 100 candidates exactly
    for num_sums, c in scored_candidates[:100]:
        if time.time() > deadline - 5:
            break
        val = max_sidon_size(c)
        if val < best_val:
            best_val = val
            best_set = c.copy()
            # If we reach the theoretical minimum (maybe 4?), we can stop early?
            # The minimum possible size for 12 elements is likely 4 or 5.
            # If we find 4, it's very good.
            if best_val <= 4:
                break

    # Phase 2: Local Search on the best candidate found
    if best_set is not None:
        cur = best_set.copy()
        cur_val = best_val
        
        # Try to improve by swapping
        for iteration in range(500):
            if time.time() > deadline - 2:
                break
            
            improved = False
            # Try swapping each element with a random one or a nearby one
            for pos in range(12):
                if time.time() > deadline - 2:
                    break
                
                # Generate a few candidates for replacement
                old_val = cur[pos]
                # Try values near existing elements to maintain density
                candidates_repl = set()
                for other in cur:
                    for delta in range(-5, 6):
                        nv = other + delta
                        if 0 <= nv <= 128 and nv != old_val:
                            candidates_repl.add(nv)
                
                # Also try some random
                for _ in range(5):
                    nv = rng.randint(0, 128)
                    if nv != old_val:
                        candidates_repl.add(nv)
                        
                for nv in candidates_repl:
                    if nv in cur:
                        continue
                    new_cur = cur.copy()
                    new_cur[pos] = nv
                    new_cur.sort()
                    
                    # Check validity
                    if not _valid(new_cur):
                        continue
                    
                    # Evaluate
                    new_val = max_sidon_size(new_cur)
                    if new_val < cur_val:
                        cur = new_cur
                        cur_val = new_val
                        best_val = cur_val
                        best_set = cur.copy()
                        improved = True
                        break # Restart pos loop with new cur
                if improved:
                    break
            
            if not improved:
                # No single swap improved it. Try a random restart or break.
                # Let's try a few random perturbations
                found_better = False
                for _ in range(10):
                    new_cur = cur.copy()
                    # Swap two random elements with random values
                    p1, p2 = rng.sample(range(12), 2)
                    v1 = rng.randint(0, 128)
                    v2 = rng.randint(0, 128)
                    new_cur[p1] = v1
                    new_cur[p2] = v2
                    new_cur.sort()
                    if _valid(new_cur):
                        new_val = max_sidon_size(new_cur)
                        if new_val < cur_val:
                            cur = new_cur
                            cur_val = new_val
                            best_val = cur_val
                            best_set = cur.copy()
                            found_better = True
                            break
                if not found_better:
                    break # Local minimum reached

    return best_set
