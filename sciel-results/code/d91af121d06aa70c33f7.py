"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

def _subset_sums_distinct_fast(arr):
    """Faster check using incremental sums."""
    n = len(arr)
    sums = [0]
    for x in arr:
        new_sums = [s + x for s in sums]
        all_sums = sums + new_sums
        if len(set(all_sums)) != len(all_sums):
            return False
        sums = all_sums
    return True

def _validate(arr):
    """Validate: 12 distinct positive ints <= max, all subset sums distinct."""
    if len(arr) != 12:
        return False
    if len(set(arr)) != 12:
        return False
    if any(x <= 0 for x in arr):
        return False
    if any(x > 1048576 for x in arr):
        return False
    return _subset_sums_distinct_fast(arr)

def _score(arr):
    """Score is the largest element (lower is better)."""
    return max(arr)

def construct():
    deadline = time.time() + 85  # 85s out of 90s budget, 5s safety margin
    
    n = 12
    # Power of 2 solution as baseline
    best = [1 << i for i in range(n)]
    best_val = max(best)  # 2048
    
    # Beam search with difference set tracking
    # State: (elements_list, difference_set, max_element_so_far)
    # We build elements in sorted order (each new element > all previous)
    # The difference set D contains all |a-b| for a,b in subset_sums
    
    # Start: first element must be 1 (smallest positive)
    # Actually, we can start with any positive integer, but for minimization,
    # starting small is better. Let's start with element 1.
    
    # Initial state: elements = [1], subset_sums = {0, 1}, diffs = {1}
    # For beam search, we maintain multiple states
    
    beam_width = 20
    candidate_range = 80  # How many candidates above greedy to try
    
    # State representation: (elements_tuple, diffs_set, max_elem)
    # We sort elements, so elements_tuple is sorted
    
    # Initialize beam with a few starting points
    # Start with element 1 (optimal for first element)
    initial_beam = []
    for first_elem in [1, 2, 3]:
        elements = [first_elem]
        # subset sums: {0, first_elem}
        # diffs: {first_elem}
        diffs = {first_elem}
        initial_beam.append((tuple(elements), diffs, first_elem))
    
    current_beam = initial_beam
    
    for step in range(1, n):
        next_beam = []
        deadline_check = deadline - 5
        
        for elements, diffs, max_elem in current_beam:
            if time.time() > deadline_check:
                break
            
            # Find greedy minimum: smallest positive int not in diffs and not in elements
            elem_set = set(elements)
            greedy = 1
            while greedy in diffs or greedy in elem_set:
                greedy += 1
            
            # Generate candidates: from greedy to greedy + candidate_range
            # Also include some random larger values for diversity
            candidates = list(range(greedy, greedy + candidate_range + 1))
            
            # Add a few random larger candidates for exploration
            for _ in range(5):
                rand_val = random.randint(greedy, greedy + candidate_range * 3)
                if rand_val not in elem_set and rand_val not in diffs:
                    candidates.append(rand_val)
            
            # Shuffle to add randomness
            random.shuffle(candidates)
            
            for x in candidates:
                if x in elem_set or x in diffs:
                    continue
                
                # Check if x is valid: x not in diffs ensures S and S+x are disjoint
                # New diffs: old_diffs ∪ {x} ∪ {|d - x| : d in old_diffs ∪ {0}}
                new_diffs = set(diffs)
                new_diffs.add(x)
                for d in diffs:
                    new_diffs.add(abs(d - x))
                # d=0 case: |0 - x| = x, already added
                
                new_elements = elements + (x,)
                new_max = max(max_elem, x)
                
                next_beam.append((new_elements, new_diffs, new_max))
                
                if time.time() > deadline_check:
                    break
            
            if time.time() > deadline_check:
                break
        
        if not next_beam:
            break
        
        # Sort by max element, keep top beam_width
        next_beam.sort(key=lambda item: item[2])
        current_beam = next_beam[:beam_width]
        
        # Update best if we have a complete solution
        for elements, diffs, max_elem in current_beam:
            if len(elements) == n:
                if max_elem < best_val:
                    best_val = max_elem
                    best = list(elements)
        
        if time.time() > deadline - 10:
            break
    
    # Final validation
    if _validate(best):
        return best
    
    # Fallback: return power of 2 solution
    return [1 << i for i in range(n)]
