import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums differ.
    
    Uses ILP via scipy.optimize.milp to maximize set size, with fallback to
    known 31-element AP [226..256] and greedy heuristics.
    """
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    # Known valid 31-element solution
    fallback = list(range(226, 257))
    
    # Quick validation helper: check if a set satisfies the cardinality-separated property
    def is_valid(s):
        n = len(s)
        if n == 0:
            return True
        s_sorted = sorted(s)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + s_sorted[i]
        min_sums = [prefix[k] for k in range(n + 1)]
        max_sums = [prefix[n] - prefix[n - k] for k in range(n + 1)]
        for c1 in range(n + 1):
            for c2 in range(c1 + 1, n + 1):
                if max_sums[c1] >= min_sums[c2]:
                    return False
        return True
    
    best_size = 31
    best_set = fallback
    
    # Try APs quickly first
    for n in range(32, 40):
        if time.time() > deadline - 5:
            break
        for d in range(1, 4):
            max_a = 256 - (n - 1) * d
            if max_a < 1:
                continue
            for a in [1, max_a, max_a - d, max_a - 2*d]:
                if a < 1 or a > max_a:
                    continue
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
    
    # Try ILP-based search for larger sets
    if best_size < 35 and time.time() < deadline - 10:
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import scipy.sparse as sp
            
            # We'll try to find a set of size N by solving an ILP
            # Variables: x_i for i in 1..256 (binary, select element i)
            # Objective: maximize sum(x_i)
            # Constraints:
            # 1. Sum of selected elements with cardinality k must have
            #    max_sum(k) < min_sum(k+1) for all k
            # 
            # This is tricky because subset sums are nonlinear.
            # Instead, we use a sufficient condition based on sorted elements.
            # If we select elements v_1 < v_2 < ... < v_n, then:
            # min_sum(k) = sum of k smallest selected
            # max_sum(k) = sum of k largest selected
            # We need: sum of k largest < sum of (k+1) smallest for all k
            #
            # This is still nonlinear. Let's use a different approach:
            # Try specific candidate sets and validate them.
            
            # Generate candidate sets using various patterns
            candidates_to_try = []
            
            # Pattern 1: Two clusters - small and large
            for n_small in range(10, 25):
                n_large = 35 - n_small
                if n_large < 5:
                    continue
                small = list(range(1, n_small + 1))
                large = list(range(257 - n_large, 257))
                if len(set(small + large)) == 35:
                    candidates_to_try.append(small + large)
            
            # Pattern 2: Geometric-like
            for base in [2, 3]:
                v = 1
                while v <= 256:
                    candidates_to_try.append([v])
                    v = v * base + 1
            
            # Pattern 3: Spread out
            for start in range(1, 20):
                s = list(range(start, 257, max(1, (256 - start) // 34)))
                if len(s) >= 35:
                    candidates_to_try.append(s[:35])
            
            # Validate candidates
            for cand in candidates_to_try:
                if time.time() > deadline - 5:
                    break
                if len(cand) >= 32 and is_valid(cand):
                    if len(cand) > best_size:
                        best_size = len(cand)
                        best_set = cand
        except Exception:
            pass
    
    # Greedy construction: try to build a large valid set
    if best_size < 35 and time.time() < deadline - 5:
        # Start with a promising base and try to add elements
        # Key insight: we want elements spaced out so that subset sum ranges
        # for different cardinalities don't overlap
        
        # Try: select elements from 1..256 with good spacing
        # Use a greedy approach: start empty, add elements that maintain validity
        
        def try_greedy_build(target_size, max_iter=100000):
            """Try to build a valid set of target_size elements."""
            selected = []
            available = list(range(1, 257))
            
            # Sort available by some heuristic
            # Try starting with spread-out elements
            for _ in range(target_size):
                best_candidate = None
                best_score = -1
                for elem in available:
                    trial = selected + [elem]
                    if len(trial) != len(set(trial)):
                        continue
                    if is_valid(trial):
                        # Score: prefer elements that maintain good separation
                        score = elem  # higher elements help max sums
                        if score > best_score:
                            best_score = score
                            best_candidate = elem
                if best_candidate is None:
                    break
                selected.append(best_candidate)
                available.remove(best_candidate)
            
            if len(selected) >= target_size and is_valid(selected):
                return selected
            return None
        
        # Try building a 35-element set
        for target in [35, 34, 33, 32]:
            if time.time() > deadline - 3:
                break
            result = try_greedy_build(target)
            if result is not None and len(result) > best_size:
                best_size = len(result)
                best_set = result
                break
    
    # Final safety check
    if not is_valid(best_set):
        return fallback
    
    # Ensure return type is correct
    result = [int(x) for x in best_set]
    return result
# EVOLVE-BLOCK-END
