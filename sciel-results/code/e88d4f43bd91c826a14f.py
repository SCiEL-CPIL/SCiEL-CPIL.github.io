import time
import sys
import math
from itertools import permutations

def construct(n):
    # Deadline guard
    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    # 1. Generate recursive superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Optimize overlap: only check up to m-1 or len(new)
            max_overlap = min(len(new), len(block) - 1)
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    # If n is small, return directly
    if n < 5:
        return "".join(map(str, word))

    # 2. Optimized Greedy Reduction
    # Precompute permutation to index mapping
    # Map permutations of 1..n to 0..n!-1
    perm_map = {}
    idx = 0
    for p in permutations(range(1, n + 1)):
        perm_map[p] = idx
        idx += 1
    
    num_perms = math.factorial(n)
    counts = [0] * num_perms
    
    # Initialize counts
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        counts[perm_map[p]] += 1
        
    # Reduction loop
    # We will do multiple passes. In each pass, we scan left to right.
    # If removing word[i] is valid, we remove it and continue.
    # This avoids the O(L^2) behavior of the previous version if many deletions are possible.
    
    while time.time() < deadline:
        changed = False
        i = 0
        L = len(word)
        while i < L:
            # Check if removing word[i] is valid
            # Affected windows: start j from max(0, i-n+1) to min(i, L-n)
            j_min = max(0, i - n + 1)
            j_max = min(i, L - n)
            
            valid = True
            affected_indices = []
            
            for j in range(j_min, j_max + 1):
                p = tuple(word[j:j+n])
                p_idx = perm_map[p]
                counts[p_idx] -= 1
                affected_indices.append(p_idx)
                if counts[p_idx] == 0:
                    valid = False
                    break
            
            if valid:
                word.pop(i)
                changed = True
                L -= 1
                # Do not increment i, because the next element shifted into position i
            else:
                # Restore counts
                for p_idx in affected_indices:
                    counts[p_idx] += 1
                i += 1
        
        if not changed:
            break

    return "".join(map(str, word))
