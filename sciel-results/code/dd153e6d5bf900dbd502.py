"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _is_valid_set(s):
    """Check if a set of integers has the required properties."""
    n = len(s)
    # Generate all nonempty subset sums
    subset_sums = set()
    for r in range(1, n + 1):
        for comb in combinations(s, r):
            total = sum(comb)
            subset_sums.add(total)
    
    sums_list = sorted(subset_sums)
    
    # Check divisibility constraint: no two distinct sums divide each other
    for i in range(len(sums_list)):
        for j in range(i + 1, len(sums_list)):
            if sums_list[j] % sums_list[i] == 0:
                return False
    
    return True

def _try_construct_greedy():
    """Try to construct a valid set using a greedy approach."""
    # Start with a base that has unique subset sums
    # Powers of 2 guarantee unique subset sums
    # But we need to also satisfy the divisibility constraint
    
    # Let's try a different approach: start with numbers that are pairwise
    # non-dividing and check the full constraint
    
    # For small sets, we can be more systematic
    # Start with single elements and grow
    
    best_set = []
    
    # Try starting with different initial elements
    for start in range(2, 257):
        current = [start]
        for candidate in range(start + 1, 257):
            new_set = current + [candidate]
            if _is_valid_set(new_set):
                current = new_set
                if len(current) > len(best_set):
                    best_set = current[:]
        if len(best_set) >= 5:
            break
    
    return best_set

def _backtrack_search(deadline):
    """Backtracking search for a valid set of maximum size."""
    best = []
    
    def backtrack(current, start_val, remaining_time):
        nonlocal best
        if time.time() > deadline:
            return
        
        if len(current) > len(best):
            best = current[:]
        
        # Try to add more elements
        for val in range(start_val, 257):
            if time.time() > deadline:
                return
            new_set = current + [val]
            if _is_valid_set(new_set):
                backtrack(new_set, val + 1, deadline)
    
    backtrack([], 2, deadline)
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    # Try greedy first
    greedy_result = _try_construct_greedy()
    
    # If greedy didn't find anything good, try backtracking
    if len(greedy_result) < 4:
        result = _backtrack_search(deadline)
    else:
        result = greedy_result
    
    # Ensure we return a list of Python ints
    return list(result)
# EVOLVE-BLOCK-END
