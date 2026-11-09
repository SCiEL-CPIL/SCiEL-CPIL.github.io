import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s guard
    
    # 1. Generate smooth numbers up to 128
    smooth = set()
    a = 1
    while a <= 128:
        b = a
        while b <= 128:
            c = b
            while c <= 128:
                smooth.add(c)
                c *= 5
            b *= 3
        a *= 2
    smooth_list = sorted(smooth)
    n_smooth = len(smooth_list)
    
    # Map value to index for quick lookup
    val_to_idx = {v: i for i, v in enumerate(smooth_list)}
    
    # 2. Precompute compatibility masks
    # compatible_mask[i] has bit j set if smooth_list[i] and smooth_list[j] are compatible
    # i.e., smooth_list[i] % smooth_list[j] != 0 and smooth_list[j] % smooth_list[i] != 0
    # We also need to know which indices are available given a set of chosen indices.
    # It's easier to track forbidden indices.
    # forbidden_mask[i] has bit j set if smooth_list[i] and smooth_list[j] are INcompatible
    # (i.e., one divides the other) or i == j.
    incompatible_mask = [0] * n_smooth
    for i in range(n_smooth):
        vi = smooth_list[i]
        for j in range(n_smooth):
            vj = smooth_list[j]
            if i == j:
                incompatible_mask[i] |= (1 << j)
                continue
            if vi % vj == 0 or vj % vi == 0:
                incompatible_mask[i] |= (1 << j)

    # Precompute answers for 1..128
    answers = [None] * 129
    
    # Memoization for DFS: (rem, k, forb_mask, end_idx) -> list of values or None
    # Note: forb_mask can be large, but we only go up to 128 targets.
    # We will clear memo for each k or each target? 
    # The state space is small enough that we can just run DFS for each target.
    # To speed up, we can reuse memo across targets if we structure it well, 
    # but the 'rem' and 'forb_mask' depend on the specific path.
    # However, 'rem' is specific to the target.
    # Let's just run a fresh search for each target.
    
    def find_min_antichain(n):
        # Check k=1
        if n in val_to_idx:
            return [n]
            
        # We search for k=2, 3, ...
        # To make it faster, we can precompute all valid antichains of size 2 and 3?
        # Size 2: pairs (a,b) with a+b = n.
        # Size 3: triples.
        
        # Let's use a generic DFS with pruning.
        # To avoid deep recursion limits and overhead, we implement it carefully.
        
        # Heuristic: Larger numbers first.
        # We iterate k from 2 upwards.
        
        # Optimization: If n is small, max k is small.
        # Max k is bounded by n // min_smooth (which is 1). But distinct and antichain.
        # The smallest antichain numbers are 2,3,5,4,6,10,8,9,12,15... 
        # Actually 1 is not compatible with anything. So 1 cannot be part of a sum > 1 unless it's the only term.
        # If n=1, rep is [1].
        # If n>1, 1 cannot be used because 1 divides everything.
        # So for n>1, we only use smooth numbers >= 2.
        
        # Let's refine the list for n>1
        # If n==1, return [1]
        
        if n == 1:
            return [1]
            
        # For n > 1, we cannot use 1.
        # Let's filter smooth_list to exclude 1 for the search.
        # But wait, if we pick 1, we can't pick anything else. So for k>1, 1 is forbidden.
        
        memo = {}
        
        def dfs(rem, k, forb_mask, end_idx):
            if k == 0:
                return [] if rem == 0 else None
                
            if rem < 0:
                return None
                
            key = (rem, k, forb_mask, end_idx)
            if key in memo:
                return memo[key]
                
            # Pruning: Check if enough numbers available
            # We need to pick k numbers from indices <= end_idx that are not in forb_mask
            # And their sum must be rem.
            
            # Quick check: min sum of k smallest available compatible numbers
            # And max sum of k largest available compatible numbers
            
            # To speed up, we can precompute suffix sums of min/max?
            # No, forb_mask changes.
            
            # Let's just iterate.
            # We try picking the largest possible number first to reduce rem.
            found = None
            for i in range(end_idx, -1, -1):
                if (forb_mask & (1 << i)):
                    continue
                val = smooth_list[i]
                if val > rem:
                    continue # Since list is sorted, smaller values are to the left. 
                             # But we are iterating downwards. 
                             # If val > rem, we can continue to smaller i (smaller val).
                
                # Check if this choice allows a solution
                # New forbidden mask includes incompatible with i
                new_forb = forb_mask | incompatible_mask[i]
                
                # Recursive call
                res = dfs(rem - val, k - 1, new_forb, i - 1)
                if res is not None:
                    res.append(val)
                    found = res
                    break # We found a solution, return it. 
                          # We want ANY solution with min k.
                          
            memo[key] = found
            return found

        for k in range(2, 31):
            if time.time() > deadline:
                return None # Fallback handled outside
            # Optimization: If k is large, min sum might exceed n.
            # Min sum of k distinct smooth numbers (excluding 1) 
            # The smallest smooths are 2,3,4,5,6,8,9,10...
            # We can precompute prefix sums of smallest smooths to prune k.
            
            res = dfs(n, k, 0, n_smooth - 1)
            if res is not None:
                return res
                
        return None

    # Precompute all answers
    for n in range(1, 129):
        if time.time() > deadline:
            # Fallback: fill rest with [1] or best guess
            # This shouldn't happen if algorithm is fast
            while len(answers) < 129:
                answers.append([1])
            break
            
        rep = find_min_antichain(n)
        if rep is None:
            # Should not happen, but fallback
            rep = [n] if n in val_to_idx else [1]
        answers[n] = rep
        
    # If we broke out early, fill remaining
    for i in range(1, 129):
        if answers[i] is None:
            answers[i] = [1]
            
    return answers[1:129]
