import time
from fractions import Fraction

def construct():
    """
    Constructs a subset of [1, 256] with no non-constant 3-term arithmetic progression
    that maximizes the sum of reciprocals.
    """
    n = 256
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Precompute conflicts: for each x, list of pairs (a, b) such that {a, b, x} forms a 3-AP
    # Actually, let's use a different approach: for each x, store the set of elements that conflict with it
    # Two elements a, b conflict with x if they form a 3-AP with x
    # For a 3-AP (a, b, c) with b in the middle: a + c = 2b
    # So x conflicts with pair (a, c) if a + c = 2x
    # And x conflicts with pair (a, b) if a + b = 2x (x is the middle)
    # Or (b, c) if b + c = 2x... wait, let me think more carefully.
    
    # A 3-AP is (a, a+d, a+2d) for d > 0.
    # If we add x to a set S, it creates a 3-AP if:
    # 1. x is the middle: exists a, c in S such that a + c = 2x and a < x < c
    # 2. x is the end: exists a, b in S such that a + b = 2x (x is the far end, a is start, b is middle)
    #    i.e., b = (a+x)/2, so a and (a+x)/2 are in S
    # 3. x is the start: exists b, c in S such that c - b = b - x, i.e., c = 2b - x
    #    i.e., b and (2b-x) are in S
    
    # Let's use a bitset approach for efficiency
    # Represent the set as a Python int (bitset) where bit i is set if i is in the set
    
    def can_add_bitset(x, bitset):
        """Check if x can be added to the set represented by bitset."""
        # Check if x is already in the set
        if (bitset >> x) & 1:
            return False
        
        # Check all three cases
        # Case 1: x is middle, need a + c = 2x with a < x < c
        for a in range(1, x):
            c = 2 * x - a
            if c > x and c <= n and (bitset >> c) & 1:
                return False
        
        # Case 2: x is the end (largest), need a + b = 2x with a < b < x
        # b = (a + x) / 2, so a + x must be even
        for a in range(1, x):
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if b > a and b < x and (bitset >> b) & 1:
                    return False
        
        # Case 3: x is the start (smallest), need b + c = 2b... 
        # Actually: x, b, c with b = (x+c)/2, so c = 2b - x
        for b in range(x + 1, n + 1):
            c = 2 * b - x
            if c > b and c <= n and (bitset >> c) & 1:
                return False
        
        return True
    
    def reciprocal_sum_from_bitset(bitset):
        total = Fraction(0, 1)
        for x in range(1, n + 1):
            if (bitset >> x) & 1:
                total += Fraction(1, x)
        return total
    
    def bitset_to_list(bitset):
        return [x for x in range(1, n + 1) if (bitset >> x) & 1]
    
    def is_3ap_free(lst):
        s = set(lst)
        for i, a in enumerate(lst):
            for b in lst[i+1:]:
                if (a + b) % 2 == 0:
                    mid = (a + b) // 2
                    if mid != a and mid != b and mid in s:
                        return False
        return True
    
    # Greedy initialization using bitset
    bitset = 0
    for x in range(1, n + 1):
        if can_add_bitset(x, bitset):
            bitset |= (1 << x)
    
    current_sum = reciprocal_sum_from_bitset(bitset)
    
    # Local improvement: try removing one element and re-optimizing
    improved = True
    max_iterations = 200
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        if time.time() > deadline:
            break
        
        current_bitset = bitset
        current_sum = reciprocal_sum_from_bitset(current_bitset)
        
        # Try removing each element
        for rem in range(1, n + 1):
            if time.time() > deadline:
                break
            
            if not (current_bitset >> rem) & 1:
                continue
            
            trial_bitset = current_bitset & ~(1 << rem)
            new_bitset = trial_bitset
            
            # Try to add back elements in order of decreasing 1/x (increasing x)
            for cand in range(1, n + 1):
                if time.time() > deadline:
                    break
                if (new_bitset >> cand) & 1:
                    continue
                if can_add_bitset(cand, new_bitset):
                    new_bitset |= (1 << cand)
            
            new_sum = reciprocal_sum_from_bitset(new_bitset)
            if new_sum > current_sum:
                bitset = new_bitset
                improved = True
                break
    
    # Swap-based improvements
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_bitset = bitset
            current_sum = reciprocal_sum_from_bitset(current_bitset)
            
            for a in range(1, n + 1):
                if time.time() > deadline:
                    break
                if not (current_bitset >> a) & 1:
                    continue
                for b in range(a + 1, n + 1):
                    if time.time() > deadline:
                        break
                    if (current_bitset >> b) & 1:
                        continue
                    
                    # Try swapping: remove a, add b
                    trial_bitset = current_bitset & ~(1 << a)
                    if can_add_bitset(b, trial_bitset):
                        trial_bitset |= (1 << b)
                        new_bitset = trial_bitset
                        
                        # Now try to add more elements
                        for cand in range(1, n + 1):
                            if time.time() > deadline:
                                break
                            if (new_bitset >> cand) & 1:
                                continue
                            if can_add_bitset(cand, new_bitset):
                                new_bitset |= (1 << cand)
                        
                        new_sum = reciprocal_sum_from_bitset(new_bitset)
                        if new_sum > current_sum:
                            bitset = new_bitset
                            improved = True
                            break
                if improved:
                    break
    
    # Final local search: try adding any missing element
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for cand in range(1, n + 1):
                if time.time() > deadline:
                    break
                if (bitset >> cand) & 1:
                    continue
                if can_add_bitset(cand, bitset):
                    bitset |= (1 << cand)
                    improved = True
    
    result = bitset_to_list(bitset)
    assert len(result) > 0, "Set must be nonempty"
    assert is_3ap_free(result), "Set must be 3-AP-free"
    
    return result
