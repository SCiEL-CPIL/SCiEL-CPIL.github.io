"""Rewritten program: maximize pairwise-noncoprime subset of [1,255] containing 255.

255 = 3 * 5 * 17. The seed uses multiples of 5 (size 51).
Using multiples of 3 gives size 85. We can potentially do better by combining
multiples of 3 and 5, but they may not be pairwise non-coprime with each other.

Key insight: If we take all multiples of 3, we get 85 elements.
If we try to add some multiples of 5 that are NOT multiples of 3, those added elements
must share a gcd > 1 with ALL existing multiples of 3. But a number like 5 has gcd(5, 3)=1,
gcd(5, 6)=1, etc. So we can't easily mix.

However, we could try a different approach: use a local search to find the maximum
clique in the graph where edges connect numbers with gcd > 1, constrained to include 255.

But for n=255, a full clique search is expensive. Let's think about structure.

255 = 3 * 5 * 17. Any number that shares a factor with 255 must be divisible by 3, 5, or 17.

The set of all numbers in [1,255] divisible by 3 has size 85.
The set divisible by 5 has size 51.
The set divisible by 17 has size 15.

Can we combine these? If we take all multiples of 3, can we add some multiples of 5 or 17?
A multiple of 5 that is not a multiple of 3, say 5, has gcd(5, 3) = 1. So it can't be added.

What if we take a different strategy? Consider the set of all numbers that are divisible by
at least two of {3, 5, 17}? No, that's smaller.

Actually, the largest pairwise non-coprime set containing 255 is likely just the multiples
of the smallest prime factor, which is 3, giving 85 elements.

But wait - could we do better with a more clever construction? Let's use a greedy/local
search approach to see if we can exceed 85.

Given the 90s deadline, let's implement a heuristic search that tries to build a large
pairwise non-coprime set containing 255.
"""

import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Candidates: numbers in [1, 255] with gcd(x, 255) > 1
    # These are the only numbers that can be in a valid set with 255
    candidates = [i for i in range(1, 256) if math.gcd(i, 255) > 1]
    
    # Initial solution: all multiples of 3 (includes 255)
    # 255 is divisible by 3, so this is valid
    initial = set(range(3, 256, 3))
    
    # Verify initial is valid
    def is_valid(s):
        lst = list(s)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if math.gcd(lst[i], lst[j]) == 1:
                    return False
        return True
    
    # Precompute gcd matrix for candidates
    n = len(candidates)
    idx = {v: i for i, v in enumerate(candidates)}
    gcd_mat = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = math.gcd(candidates[i], candidates[j])
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g
    
    # Check if two candidates are compatible
    def compatible(i, j):
        return gcd_mat[i][j] > 1
    
    # Local search: try to add elements from candidates not in current set
    # and remove elements that cause conflicts
    
    best_set = set(initial)
    best_size = len(best_set)
    
    # Convert to index-based for efficiency
    best_idx = set(idx[v] for v in best_set)
    
    # Greedy improvement: try adding each candidate not in set
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        improved = False
        # Try to add elements
        for v in candidates:
            if time.time() > deadline:
                break
            if v in best_set:
                continue
            v_idx = idx[v]
            # Check if v is compatible with all in best_set
            can_add = True
            for u in best_set:
                if not compatible(v_idx, idx[u]):
                    can_add = False
                    break
            if can_add:
                best_set.add(v)
                best_idx.add(v_idx)
                improved = True
                best_size += 1
        
        if not improved:
            break
    
    # Try local search: swap operations
    for it in range(200):
        if time.time() > deadline:
            break
        
        # Try removing one element and adding two
        random.shuffle(candidates)
        found_swap = False
        for remove_v in candidates:
            if time.time() > deadline:
                break
            if remove_v not in best_set:
                continue
            # Try to find two elements to add after removing remove_v
            temp_set = best_set - {remove_v}
            # Find all candidates compatible with temp_set
            addable = []
            for v in candidates:
                if v in best_set:
                    continue
                v_idx = idx[v]
                can_add = True
                for u in temp_set:
                    if not compatible(v_idx, idx[u]):
                        can_add = False
                        break
                if can_add:
                    addable.append(v)
            
            # Try to find two addable elements that are compatible with each other
            for i in range(len(addable)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(addable)):
                    a, b = addable[i], addable[j]
                    if compatible(idx[a], idx[b]):
                        # Do the swap
                        best_set.remove(remove_v)
                        best_set.add(a)
                        best_set.add(b)
                        found_swap = True
                        best_size += 1
                        break
                if found_swap:
                    break
            if found_swap:
                break
    
    # Final greedy pass
    for v in candidates:
        if time.time() > deadline:
            break
        if v in best_set:
            continue
        v_idx = idx[v]
        can_add = True
        for u in best_set:
            if not compatible(v_idx, idx[u]):
                can_add = False
                break
        if can_add:
            best_set.add(v)
            best_size += 1
    
    # Ensure 255 is in the set
    if 255 not in best_set:
        best_set.add(255)
    
    # Final validation
    lst = sorted(best_set)
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if math.gcd(lst[i], lst[j]) == 1:
                # Fallback to initial solution
                return list(initial)
    
    return list(best_set)
