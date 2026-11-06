import time
import sys

def construct():
    """
    Constructs a set of 12 distinct positive integers such that all subset sums are distinct,
    minimizing the largest integer.
    
    Uses backtracking search with pruning to find a better solution than powers of 2.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 12
    
    # Powers of 2 as fallback
    fallback = [1 << i for i in range(n)]
    best_max = fallback[-1]
    best_solution = fallback[:]
    
    # We'll search for solutions with max < best_max
    # Use backtracking: build the sequence one element at a time
    # Key insight: sort elements, and at each step try the smallest valid next element
    
    # For efficiency, we use the fact that if we have k elements with all subset sums distinct,
    # and the current max subset sum is S, then the next element x must satisfy:
    # The new subset sums are {s, s+x for s in current_sums}
    # These are all distinct iff x is not equal to the difference of any two existing subset sums
    # i.e., x not in {a - b for a, b in current_sums, a > b}
    
    # But that's expensive to check. Instead, we can use a bitset approach.
    
    # Let's use a different approach: try to find a solution with a given max M
    # using backtracking, and binary search on M
    
    def can_construct(max_val, nums, sums_set, depth):
        """Try to complete the construction with remaining elements <= max_val."""
        if time.time() > deadline:
            return False, None
        
        if depth == n:
            return True, nums[:]
        
        # Determine the range of valid next values
        # Must be > 0, <= max_val, and not already used
        # And must not create duplicate subset sums
        
        # For efficiency, try values in increasing order
        # Lower bound: 1, but we can do better
        # Upper bound: max_val
        
        # To make this tractable, we'll try a limited set of candidates
        # The key constraint: new_sums = sums_set | {s + x for s in sums_set}
        # must have size 2 * len(sums_set)
        
        # This is equivalent to: x is not in {a - b : a, b in sums_set, a != b}
        # i.e., x is not a difference of two subset sums
        
        # For small depth, this is feasible. For larger depth, we need pruning.
        
        # Try candidates from 1 to max_val, but skip used values
        used = set(nums)
        
        # Optimization: the next element must be at least 1
        # And at most max_val
        
        # For very large depth, this search space is too big
        # We need a smarter approach
        
        # Alternative: use the fact that for distinct subset sums,
        # a sufficient condition is that each element > sum of all previous
        # But that gives powers of 2, which is what we want to improve upon
        
        # Let's try a different strategy: fix the first few elements to be small,
        # and search for the rest
        
        # Actually, let's just do a simple backtracking with the constraint
        # that we try values in a smart order
        
        # To keep this tractable within time limits, we'll limit the search
        if depth > 6:
            # For deeper levels, use a greedy-like approach with backtracking
            # Try a limited number of candidates
            candidates = []
            for x in range(1, min(max_val + 1, 500)):
                if x not in used:
                    candidates.append(x)
            # Try a subset of candidates for speed
            if len(candidates) > 200:
                # Sample candidates
                import random
                candidates = candidates[:100] + candidates[-100:]
        else:
            candidates = [x for x in range(1, min(max_val + 1, 2000)) if x not in used]
        
        for x in candidates:
            if time.time() > deadline:
                return False, None
            
            # Check if adding x creates duplicate subset sums
            new_sums = set()
            valid = True
            for s in sums_set:
                if s in new_sums or (s + x) in new_sums:
                    valid = False
                    break
                new_sums.add(s)
                new_sums.add(s + x)
            
            if valid and len(new_sums) == 2 * len(sums_set):
                nums.append(x)
                result, sol = can_construct(max_val, nums, new_sums, depth + 1)
                nums.pop()
                if result:
                    return True, sol
        
        return False, None
    
    # Binary search on the maximum value
    # Lower bound: we need at least n distinct positive integers, so max >= n
    # But actually, for distinct subset sums, the sum must be >= 2^n - 1
    # So max >= (2^n - 1) / n approximately
    
    low = 1
    high = best_max  # We know powers of 2 works
    
    # First, try to improve upon powers of 2
    # Try values from high-1 down to some lower bound
    
    # Actually, let's just do a linear search from high-1 downward
    # and stop at the first feasible value
    
    for M in range(high - 1, 0, -1):
        if time.time() > deadline:
            break
        
        # Try to construct with max element = M
        # Start with empty set
        nums = []
        sums_set = {0}
        
        result, sol = can_construct(M, nums, sums_set, 0)
        
        if result:
            # Verify the solution
            all_sums = set()
            for mask in range(1 << n):
                s = 0
                for i in range(n):
                    if mask & (1 << i):
                        s += sol[i]
                if s in all_sums:
                    break
                all_sums.add(s)
            else:
                if len(all_sums) == 1 << n:
                    # Valid solution
                    if max(sol) < best_max:
                        best_max = max(sol)
                        best_solution = sol[:]
            break  # Found the best with this M, stop
    
    # If we found an improvement, return it
    if best_solution != fallback:
        return best_solution
    else:
        return fallback
