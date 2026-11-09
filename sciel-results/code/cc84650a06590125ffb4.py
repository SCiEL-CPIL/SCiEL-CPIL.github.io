"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

_DEADLINE = 85.0

def _check_distinct_subset_sums(arr):
    """Check if all subset sums are distinct. Returns True if valid."""
    n = len(arr)
    sums = set()
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += arr[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def _check_prefix(arr, prefix_len):
    """Check if the first prefix_len elements have distinct subset sums."""
    n = prefix_len
    sums = set()
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += arr[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def _greedy_construct(start_val, max_val, n=12, deadline=None):
    """
    Greedy: pick smallest possible value > start_val that keeps all subset sums distinct.
    Returns the constructed list or None if we can't complete.
    """
    arr = []
    current_max = start_val
    sums_set = {0}
    
    for i in range(n):
        if deadline is not None and time.time() > deadline:
            return None
        found = False
        candidate = current_max + 1
        while candidate <= max_val:
            # Check if adding candidate keeps all subset sums distinct
            # New sums = old sums + {s + candidate for s in old sums}
            # Need: no collision between old sums and new sums, and new sums all distinct
            new_sums = set()
            valid = True
            for s in sums_set:
                ns = s + candidate
                if ns in sums_set or ns in new_sums:
                    valid = False
                    break
                new_sums.add(ns)
            if valid:
                arr.append(candidate)
                sums_set.update(new_sums)
                current_max = candidate
                found = True
                break
            candidate += 1
        if not found:
            return None
    return arr

def construct():
    start_time = time.time()
    n = 12
    max_allowed = 1048576
    
    # First, verify the powers of 2 solution is valid (it is)
    baseline = [1 << i for i in range(n)]
    best_max = max(baseline)  # 2048
    
    # Try to find a better solution via greedy with different starting points
    # The greedy from 1 gives powers of 2. Let's try other strategies.
    
    # Strategy: Try to find a set where the max is less than 2048.
    # Use backtracking with pruning.
    
    def backtrack(arr, sums_set, current_max, target_max, deadline):
        """
        Try to complete the set from current state.
        arr: current list of elements
        sums_set: set of all subset sums so far
        current_max: last element added
        target_max: maximum allowed value for the next element
        """
        if time.time() > deadline:
            return None
        if len(arr) == n:
            return arr[:]
        
        # Try candidates from current_max+1 up to target_max
        for candidate in range(current_max + 1, target_max + 1):
            if time.time() > deadline:
                return None
            # Check validity
            new_sums = set()
            valid = True
            for s in sums_set:
                ns = s + candidate
                if ns in sums_set or ns in new_sums:
                    valid = False
                    break
                new_sums.add(ns)
            if valid:
                arr.append(candidate)
                result = backtrack(arr, sums_set | new_sums, candidate, target_max, deadline)
                if result is not None:
                    return result
                arr.pop()
        
        return None
    
    # Try to find a solution with max < 2048
    # Start with a_1 = 1 (we can assume WLOG that the smallest is 1 for minimal max)
    # Actually, for minimal max, we should try all possible first elements.
    # But a_1=1 is best for minimizing the total sum.
    
    deadline = start_time + _DEADLINE - 5  # safety margin
    
    # Try target_max from 2047 down, but that's too many.
    # Instead, use a smarter approach: binary search on max value.
    
    # Actually, let's try a different greedy: allow non-superincreasing sequences.
    # The key constraint is just distinct subset sums.
    
    # Let me try: fix a_1=1, then greedily pick the smallest valid a_2, a_3, etc.
    # but with a twist: instead of just smallest, try a few options.
    
    # First, confirm greedy gives powers of 2
    greedy_result = _greedy_construct(1, 2047, n, deadline)
    if greedy_result is not None:
        if max(greedy_result) < best_max:
            best_max = max(greedy_result)
            baseline = greedy_result
    
    # Try a more sophisticated search: 
    # For each possible value of a_1 (1 to some limit), do a constrained backtracking
    # that tries to keep the max as small as possible.
    
    # Actually, the most promising approach: 
    # We know the answer is at most 2048. Let's try to prove or disprove that 2047 works.
    # 
    # With max element <= 2047, total sum <= 12*2047 = 24564.
    # We need 4096 distinct subset sums in [0, 24564]. That's fine (24564 >= 4095).
    #
    # The real constraint is structural. Let me try a targeted backtracking.
    
    # Try: a_1 = 1, and search for a valid completion with all elements <= 2047
    # Use iterative deepening on the maximum element.
    
    for target_max in range(2047, 1900, -1):
        if time.time() > deadline:
            break
        
        # Backtracking with a_1 fixed to 1
        arr = [1]
        sums_set = {0, 1}
        result = backtrack(arr, sums_set, 1, target_max, deadline)
        if result is not None:
            best_max = target_max
            baseline = result
            break
    
    # If we found something better, use it; otherwise use powers of 2
    if max(baseline) <= max_allowed and len(baseline) == n:
        return baseline
    return [1 << i for i in range(n)]

# EVOLVE-BLOCK-END
