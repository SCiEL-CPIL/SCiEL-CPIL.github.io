from fractions import Fraction
import time
import functools

def _lcm_all(nums):
    """Compute LCM of all numbers in the list."""
    if not nums:
        return 1
    result = nums[0]
    for n in nums[1:]:
        g = _gcd(result, n)
        result = result * n // g
    return result

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _is_dissociated(selected):
    """Check if all reciprocal subset sums are distinct."""
    if not selected:
        return True
    
    # Use integer arithmetic: multiply by LCM
    lcm = _lcm_all(selected)
    weights = [lcm // x for x in selected]
    
    # All subset sums of weights must be distinct
    n = len(selected)
    sums = set()
    sums.add(0)
    
    for i in range(n):
        w = weights[i]
        current = set(sums)
        for s in current:
            ns = s + w
            if ns in sums:
                return False
            sums.add(ns)
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin
    
    # Precompute LCM for all possible subsets is too expensive
    # Instead, use branch and bound with a good heuristic
    
    # Start with a greedy solution as baseline
    best_size = 0
    best_set = None
    
    # Greedy: try to build a large dissociated set
    # Order elements to maximize chances: larger denominators first
    # but also consider that small denominators create more conflicts
    
    # Try multiple greedy orderings
    orderings = [
        list(range(22, 0, -1)),  # Largest to smallest
        list(range(1, 23)),      # Smallest to largest
        # Interleaved
        [x for x in range(22, 0, -2)] + [x for x in range(21, 0, -2)],
    ]
    
    for order in orderings:
        selected = []
        current_sums = {Fraction(0, 1)}
        
        for x in order:
            inv = Fraction(1, x)
            new_sums = set(current_sums)
            is_valid = True
            
            for s in current_sums:
                ns = s + inv
                if ns in new_sums:
                    is_valid = False
                    break
                new_sums.add(ns)
            
            if is_valid:
                selected.append(x)
                current_sums = new_sums
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
    
    # Now try to improve via local search / branch and bound
    # We'll use a simple improvement strategy: try to add excluded elements
    # and remove conflicting ones
    
    if best_set is None:
        return list(range(1, 23))[:best_size] if best_size > 0 else [1]
    
    current_set = set(best_set)
    
    # Try to improve by swapping elements
    for _ in range(100):  # Limited iterations
        if time.time() > deadline:
            break
        
        improved = False
        # Try to add each non-selected element
        for x in range(1, 23):
            if x in current_set:
                continue
            
            # Check if adding x is valid
            test_set = current_set | {x}
            if _is_dissociated(list(test_set)):
                current_set = test_set
                best_size = len(current_set)
                improved = True
                break
        
        if not improved:
            # Try removing one element to allow adding two others
            for remove_x in list(current_set):
                if time.time() > deadline:
                    break
                
                test_set = current_set - {remove_x}
                # Try to add two elements from the complement
                added = 0
                for add_x in range(1, 23):
                    if add_x in test_set or add_x == remove_x:
                        continue
                    if _is_dissociated(list(test_set | {add_x})):
                        test_set.add(add_x)
                        added += 1
                        if added >= 2:
                            break
                
                if len(test_set) > len(current_set):
                    current_set = test_set
                    best_size = len(current_set)
                    improved = True
                    break
        
        if not improved and time.time() > start_time + 40:
            break
    
    return list(current_set)
