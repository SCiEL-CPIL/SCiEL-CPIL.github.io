"""Two Sidon sets with disjoint difference sets, n=256."""
import time
import random
import numpy as np
from itertools import combinations, combinations_with_replacement

def _pair_sums_unique(arr):
    """Check if all pair sums a+b (a<=b) are unique."""
    n = len(arr)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _nonzero_diffs(arr):
    """Return set of nonzero differences |b-a| for a != b in arr."""
    n = len(arr)
    d = set()
    for i in range(n):
        for j in range(i + 1, n):
            d.add(abs(arr[j] - arr[i]))
    return d

def _is_valid_pair(A, B):
    """Check if A and B are valid: each is Sidon, diff sets disjoint."""
    if not _pair_sums_unique(A):
        return False
    if not _pair_sums_unique(B):
        return False
    dA = _nonzero_diffs(A)
    dB = _nonzero_diffs(B)
    if dA & dB:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin within 90s limit
    
    # Start with the seed program's greedy construction as baseline
    sets = [[], []]
    for x in range(1, 129):
        for side in (0, 1):
            candidate = sets[side] + [x]
            sums = [a + b for a, b in combinations_with_replacement(candidate, 2)]
            differences = {b - a for a, b in combinations(candidate, 2)}
            other = {b - a for a, b in combinations(sets[1 - side], 2)}
            if len(sums) == len(set(sums)) and not differences & other:
                sets[side] = candidate
    
    best_A = list(sets[0])
    best_B = list(sets[1])
    best_score = len(best_A) + len(best_B)
    
    # Now try to improve via local search / hill climbing
    # Strategy: try adding elements from 1..256 that aren't in either set,
    # or swapping elements, to increase total size while maintaining validity.
    
    def try_add(A, B, x):
        """Try to add x to A or B, return new pair if valid."""
        # Try adding to A
        if x not in A:
            new_A = A + [x]
            if _pair_sums_unique(new_A):
                dA_new = _nonzero_diffs(new_A)
                dB = _nonzero_diffs(B)
                if not (dA_new & dB):
                    return new_A, B
        # Try adding to B
        if x not in B:
            new_B = B + [x]
            if _pair_sums_unique(new_B):
                dA = _nonzero_diffs(A)
                dB_new = _nonzero_diffs(new_B)
                if not (dA & dB_new):
                    return A, new_B
        return None
    
    # Greedy improvement: try all elements 1..256 in random order
    random.seed(42)
    elements = list(range(1, 257))
    random.shuffle(elements)
    
    A = list(best_A)
    B = list(best_B)
    
    for x in elements:
        if time.time() > deadline:
            break
        result = try_add(A, B, x)
        if result is not None:
            A, B = result
    
    if len(A) + len(B) > best_score:
        best_A, best_B = list(A), list(B)
        best_score = len(A) + len(B)
    
    # Try more aggressive local search: multiple restarts with random initial orders
    for restart in range(50):
        if time.time() > deadline:
            break
        
        random.seed(1000 + restart)
        A = list(best_A)
        B = list(best_B)
        
        elements = list(range(1, 257))
        random.shuffle(elements)
        
        for x in elements:
            if time.time() > deadline:
                break
            result = try_add(A, B, x)
            if result is not None:
                A, B = result
        
        if len(A) + len(B) > best_score:
            best_A, best_B = list(A), list(B)
            best_score = len(A) + len(B)
    
    # Try removing elements from one set and adding to the other (swap strategy)
    for restart in range(20):
        if time.time() > deadline:
            break
        
        random.seed(2000 + restart)
        A = list(best_A)
        B = list(best_B)
        
        # Try swapping: remove an element from A, add a different one
        for _ in range(100):
            if time.time() > deadline:
                break
            
            # Try adding random element
            x = random.randint(1, 256)
            result = try_add(A, B, x)
            if result is not None:
                A, B = result
        
        if len(A) + len(B) > best_score:
            best_A, best_B = list(A), list(B)
            best_score = len(A) + len(B)
    
    # Final polish: try to add any remaining elements in sorted order
    A = list(best_A)
    B = list(best_B)
    for x in range(1, 257):
        if time.time() > deadline:
            break
        result = try_add(A, B, x)
        if result is not None:
            A, B = result
    
    if len(A) + len(B) > best_score:
        best_A, best_B = list(A), list(B)
        best_score = len(A) + len(B)
    
    # Ensure both sets are nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    # Final validation
    if not _is_valid_pair(best_A, best_B):
        # Fallback to seed result
        best_A = list(sets[0])
        best_B = list(sets[1])
    
    return [best_A, best_B]
