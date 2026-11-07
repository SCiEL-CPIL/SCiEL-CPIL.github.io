import time
import itertools

def construct():
    deadline = time.time() + 55  # Safety margin before 60s limit
    
    def is_valid(S):
        """Check if set S satisfies the cardinality-separated subset sum property."""
        n = len(S)
        if n == 0:
            return True
        # For each cardinality k, compute all possible subset sums
        # and check that ranges for different k don't overlap
        sums_by_k = {}
        for k in range(1, n + 1):
            k_sums = set()
            for subset in itertools.combinations(S, k):
                s = sum(subset)
                k_sums.add(s)
            sums_by_k[k] = k_sums
        
        # Check that no sum appears in two different cardinality classes
        seen_sums = {}
        for k, k_sums in sums_by_k.items():
            for s in k_sums:
                if s in seen_sums:
                    if seen_sums[s] != k:
                        return False
                else:
                    seen_sums[s] = k
        return True
    
    def is_valid_fast(S):
        """Faster validity check using range separation."""
        n = len(S)
        if n == 0:
            return True
        S_sorted = sorted(S)
        # Compute min and max sum for each cardinality
        min_sums = [0] * (n + 1)
        max_sums = [0] * (n + 1)
        for k in range(1, n + 1):
            min_sums[k] = sum(S_sorted[:k])  # smallest k elements
            max_sums[k] = sum(S_sorted[-k:])  # largest k elements
        # Check that for all k < k', max_sums[k] < min_sums[k']
        for k in range(1, n):
            for k2 in range(k + 1, n + 1):
                if max_sums[k] >= min_sums[k2]:
                    return False
        return True
    
    def can_add(S, x):
        """Check if adding x to S maintains validity."""
        new_S = S + [x]
        return is_valid_fast(new_S)
    
    # Start with empty set, greedily add values
    best_set = []
    
    # Try to build a large set using backtracking
    def backtrack(S, start_val):
        nonlocal best_set
        if time.time() > deadline:
            return
        if len(S) > len(best_set):
            best_set = list(S)
        
        if len(S) >= 60:
            return
        
        # Try adding values from start_val to 256
        for x in range(start_val, 257):
            if time.time() > deadline:
                return
            if can_add(S, x):
                S.append(x)
                backtrack(S, x + 1)
                S.pop()
                # If we've found a set of size 60, we can stop
                if len(best_set) >= 60:
                    return
    
    backtrack([], 1)
    
    # Convert to proper Python ints and list
    result = [int(x) for x in best_set]
    return result
