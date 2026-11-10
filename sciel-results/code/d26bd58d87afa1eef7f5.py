"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    n = 512
    deadline = time.time() + 45  # safety margin before 60s limit
    start_time = time.time()
    
    # Greedy construction with base-3 digit sum ordering
    # Numbers with no digit > 1 in base 3 are 3-free (no 3-term AP)
    # For 4-term AP free, we use a different strategy
    
    # Strategy: Start with numbers whose base-3 representation has no consecutive 2s
    # or use a greedy algorithm with good ordering
    
    # Better approach: Use the known construction based on base 3
    # Numbers with base-3 digits in {0,1} give a 3-AP free set of size ~ n^(log_3 2)
    # For 4-AP free, we can use a denser set
    
    # Let's use a greedy algorithm with multiple passes and different orderings
    
    def digit_sum_base3(x):
        s = 0
        while x > 0:
            s += x % 3
            x //= 3
        return s
    
    def has_consecutive_2s_base3(x):
        prev = -1
        while x > 0:
            d = x % 3
            if d == 2 and prev == 2:
                return True
            prev = d
            x //= 3
        return False
    
    # Start with a good initial set
    # Use numbers that have at most one 2 in base 3, or no 2s
    initial_set = set()
    for x in range(1, n + 1):
        if not has_consecutive_2s_base3(x):
            initial_set.add(x)
    
    # Greedy add: try to add elements in order of increasing "conflict potential"
    selected = set(initial_set)
    
    # Compute conflicts for each candidate
    def count_conflicts(x, s):
        """Count how many 4-APs would be created by adding x to s"""
        conflicts = 0
        # x as first term: x, x+d, x+2d, x+3d
        for d in range(1, (n - x) // 3 + 1):
            if x + d in s and x + 2*d in s and x + 3*d in s:
                conflicts += 1
        # x as second term: x-3d, x-2d, x-d, x
        for d in range(1, x // 3 + 1):
            if x - 3*d in s and x - 2*d in s and x - d in s:
                conflicts += 1
        # x as third term: x-3d, x-2d, x-d, x -> wait, x is third: x-3d, x-2d, x, x+d
        # Actually 4-AP: a, a+d, a+2d, a+3d
        # x as third: a+2d = x, so a = x-2d, terms: x-2d, x-d, x, x+d
        for d in range(1, min(x, n - x) // 2 + 1):
            if x - 2*d in s and x - d in s and x + d in s:
                conflicts += 1
        # x as fourth: a+3d = x, so a = x-3d, terms: x-3d, x-2d, x-d, x
        for d in range(1, x // 3 + 1):
            if x - 3*d in s and x - 2*d in s and x - d in s:
                conflicts += 1
        return conflicts
    
    # Try to add elements with 0 conflicts first, then minimize conflicts
    remaining = [x for x in range(1, n + 1) if x not in selected]
    remaining.sort(key=lambda x: (digit_sum_base3(x), x))
    
    # Pass 1: add elements with 0 conflicts
    for x in remaining:
        if time.time() > deadline:
            break
        if count_conflicts(x, selected) == 0:
            selected.add(x)
    
    # Pass 2: try to add elements by removing 1 element to add 2
    # This is a local search improvement
    improved = True
    while improved and time.time() < deadline:
        improved = False
        remaining = [x for x in range(1, n + 1) if x not in selected]
        for x in remaining:
            if time.time() > deadline:
                break
            # Find all conflicts
            conflict_elements = set()
            for d in range(1, (n - x) // 3 + 1):
                if x + d in selected and x + 2*d in selected and x + 3*d in selected:
                    conflict_elements.update([x + d, x + 2*d, x + 3*d])
            for d in range(1, x // 3 + 1):
                if x - 3*d in selected and x - 2*d in selected and x - d in selected:
                    conflict_elements.update([x - 3*d, x - 2*d, x - d])
            for d in range(1, min(x, n - x) // 2 + 1):
                if x - 2*d in selected and x - d in selected and x + d in selected:
                    conflict_elements.update([x - 2*d, x - d, x + d])
            for d in range(1, x // 3 + 1):
                if x - 3*d in selected and x - 2*d in selected and x - d in selected:
                    conflict_elements.update([x - 3*d, x - 2*d, x - d])
            
            if len(conflict_elements) == 0:
                selected.add(x)
                improved = True
            elif len(conflict_elements) == 1:
                # Try removing one element to add x (net 0, but might help later)
                pass
    
    # Final greedy pass in different order
    if time.time() < deadline:
        remaining = [x for x in range(1, n + 1) if x not in selected]
        remaining.sort(key=lambda x: -digit_sum_base3(x))  # reverse order
        for x in remaining:
            if time.time() > deadline:
                break
            if count_conflicts(x, selected) == 0:
                selected.add(x)
    
    return sorted(selected)
# EVOLVE-BLOCK-END
