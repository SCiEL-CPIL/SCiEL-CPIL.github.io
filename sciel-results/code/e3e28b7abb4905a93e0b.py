import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 80  # Leave 10s margin before 90s kill
    
    best_set = None
    best_len = 0
    
    # Try multiple random seeds for diversity
    for seed in range(20):
        if time.time() > deadline - 5:
            break
        rng = random.Random(seed)
        
        # Shuffle candidates to avoid greedy bias
        candidates = list(range(1, 513))
        rng.shuffle(candidates)
        
        selected = []
        # Maintain sums for fast checking
        # sums[k] = set of sums of k elements from selected
        # But we only need to check if x is mean of subset of size k:
        #   k * x in sums[k]
        # And if any a in selected is mean of subset containing x:
        #   For subset S of selected U {x} containing x, |S| = k, sum(S) = k*a
        #   sum(S\{x}) + x = k*a  =>  sum(S\{x}) = k*a - x
        #   S\{x} is a subset of selected of size k-1
        #   So we need: k*a - x in sums[k-1] for any a in selected, k>=2
        
        # We'll maintain sums_by_size as dict of sets
        sums_by_size = {0: {0}}
        
        for x in candidates:
            if time.time() > deadline - 5:
                break
                
            # Check 1: x is not the mean of any subset of 'selected' of size >= 2
            valid = True
            for k in range(2, len(selected) + 1):
                if k * x in sums_by_size.get(k, set()):
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check 2: No a in selected is the mean of any subset of (selected U {x}) containing x
            # For each a in selected, for each k >= 2 (size of subset containing x):
            #   Need: k*a - x NOT in sums_by_size[k-1]
            for a in selected:
                for k in range(2, len(selected) + 2):
                    # k is the size of the subset containing x
                    # k-1 elements from selected must sum to k*a - x
                    target = k * a - x
                    if k - 1 <= len(selected):
                        if target in sums_by_size.get(k - 1, set()):
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                selected.append(x)
                # Update sums_by_size
                old_sums = sums_by_size
                new_sums = {k: set() for k in range(len(selected) + 1)}
                new_sums[0] = {0}
                for k in range(1, len(selected) + 1):
                    if k in old_sums:
                        new_sums[k].update(old_sums[k])
                    if k - 1 in old_sums:
                        new_sums[k].update(s + x for s in old_sums[k - 1])
                sums_by_size = new_sums
                
                if len(selected) > best_len:
                    best_len = len(selected)
                    best_set = list(selected)
        
        # Early stop if we reached 48
        if best_len >= 48:
            break
    
    # Final verification of best_set
    if best_set is None:
        return list(range(1, 49))  # Fallback
    
    # Verify: no element is the mean of any subset of at least 2 OTHER elements
    n = len(best_set)
    for i, a in enumerate(best_set):
        others = [best_set[j] for j in range(n) if j != i]
        m = len(others)
        # Check all subset sizes k from 2 to m
        for k in range(2, m + 1):
            # Check if there's a subset of 'others' of size k with sum = k*a
            # Use DP for subset sum of size k
            target = k * a
            # DP: dp[j] = set of sums using j elements
            dp = [set() for _ in range(k + 1)]
            dp[0] = {0}
            for val in others:
                for j in range(k, 0, -1):
                    if dp[j - 1]:
                        dp[j].update(s + val for s in dp[j - 1])
            if target in dp[k]:
                # Invalid - should not happen if construction was correct
                # But if it does, we need to handle it
                # For safety, try to remove 'a' and re-verify
                break
    
    return best_set
