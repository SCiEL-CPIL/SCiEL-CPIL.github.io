import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums differ.
    
    Uses systematic search with optimized validity checking to maximize set size.
    """
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    # Known valid 31-element solution
    fallback = list(range(226, 257))
    
    def is_valid(s):
        """Check if a set satisfies the cardinality-separated property.
        
        The condition is equivalent to: for all k from 1 to n-1:
        sum of k smallest < sum of (k+1) largest
        """
        n = len(s)
        if n == 0:
            return True
        s_sorted = sorted(s)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + s_sorted[i]
        
        for k in range(1, n):
            if prefix[k] >= prefix[n] - prefix[n - (k + 1)]:
                return False
        return True
    
    best_size = 31
    best_set = fallback
    
    # Systematic search for larger sets
    
    # Pattern 1: [1..m] + [L..256]
    for n_small in range(1, 25):
        if time.time() > deadline - 3:
            break
        for n_large in range(1, 40):
            n_total = n_small + n_large
            if n_total <= best_size:
                continue
            
            small = list(range(1, n_small + 1))
            large = list(range(257 - n_large, 257))
            
            candidate = small + large
            if len(set(candidate)) != n_total:
                continue
            
            if is_valid(candidate):
                if n_total > best_size:
                    best_size = n_total
                    best_set = candidate
    
    # Pattern 2: Try different large starting points
    for n_small in range(1, 20):
        if time.time() > deadline - 3:
            break
        for n_large in range(1, 40):
            n_total = n_small + n_large
            if n_total <= best_size:
                continue
            
            small = list(range(1, n_small + 1))
            for large_start in range(100, 257 - n_large + 2):
                if time.time() > deadline - 3:
                    break
                large = list(range(large_start, large_start + n_large))
                if large[-1] > 256:
                    continue
                
                candidate = small + large
                if len(set(candidate)) != n_total:
                    continue
                
                if is_valid(candidate):
                    if n_total > best_size:
                        best_size = n_total
                        best_set = candidate
    
    # Pattern 3: Geometric sequences
    for base in [2, 3, 4, 5, 6, 7, 8]:
        if time.time() > deadline - 3:
            break
        elems = []
        v = 1
        while v <= 256:
            elems.append(v)
            v = v * base + 1
        if len(elems) > best_size and is_valid(elems):
            best_size = len(elems)
            best_set = elems
    
    # Pattern 4: AP with various parameters
    for n in range(best_size + 1, 50):
        if time.time() > deadline - 3:
            break
        for d in range(1, 10):
            if time.time() > deadline - 3:
                break
            max_a = 256 - (n - 1) * d
            if max_a < 1:
                continue
            for a in range(1, min(max_a + 1, 50)):
                s = [a + i * d for i in range(n)]
                if s[-1] > 256:
                    continue
                if is_valid(s):
                    if n > best_size:
                        best_size = n
                        best_set = s
                        break
            if best_size >= n:
                break
    
    # Final validation
    if not is_valid(best_set):
        return fallback
    
    result = [int(x) for x in best_set]
    return result
# EVOLVE-BLOCK-END
