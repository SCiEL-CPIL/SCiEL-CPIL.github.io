"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce

def construct():
    """
    Find a nonempty subset of [1,22] with maximum cardinality such that
    all reciprocal subset sums (including empty sum 0) are distinct.
    """
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Compute LCM of 1..22 to convert reciprocal sums to integer sums
    L = reduce(lcm, range(1, 23))
    
    # For each element a in [1,22], its contribution to the integer sum is L // a
    # We need to find a subset S of [1,22] such that all subset sums of {L//a : a in S} are distinct
    
    # A set of integers {v1, ..., vk} has all distinct subset sums iff it's a "dissociated" set
    # i.e., no two distinct subsets have the same sum
    
    # Strategy: try to build the largest possible set using backtracking with pruning
    # Start from the full set [1..22] and try to remove minimal elements to make it dissociated
    
    # First, let's try a greedy approach: start with empty set, add elements one by one
    # if the set remains dissociated
    
    # Better approach: use the fact that for a set to have all distinct subset sums,
    # the elements should grow faster than the sum of previous elements (like superincreasing)
    # But we're constrained to L//a values
    
    # Let's compute the values
    vals = {a: L // a for a in range(1, 23)}
    
    # Try to find maximum cardinality subset
    # We'll use a systematic search: try all subsets of size 22, 21, 20, ...
    # but that's too many. Instead, use backtracking with pruning.
    
    # Key insight: if we sort the values in descending order, we can check incrementally
    # A set has all distinct subset sums if and only if for every element, it's greater than
    # the sum of all smaller elements (sufficient condition, not necessary)
    
    # Actually, the necessary and sufficient condition is harder. Let's use a different approach:
    # For a set of k elements, there are 2^k subset sums. They must all be distinct.
    # We can check this by generating all subset sums and checking for duplicates.
    
    # For efficiency, let's try to build the set greedily from largest to smallest values,
    # and check at each step whether the set remains dissociated.
    
    # Sort elements by their value (L//a) in descending order
    elements = sorted(range(1, 23), key=lambda a: vals[a], reverse=True)
    
    def is_dissociated(subset):
        """Check if all subset sums of the given subset (using integer values) are distinct."""
        if not subset:
            return True
        k = len(subset)
        if k > 20:  # 2^20 = 1M, feasible; 2^21 = 2M, still feasible
            pass
        
        # Generate all subset sums
        sums = {0}
        for a in subset:
            v = vals[a]
            new_sums = set()
            for s in sums:
                ns = s + v
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return True
    
    # Greedy: start with all 22 elements, try to remove as few as possible
    # Actually, let's try to add elements one by one from largest value to smallest
    
    # Try different orderings and strategies
    
    best_subset = []
    
    # Strategy 1: Greedy - add elements in order of decreasing value
    current = []
    for a in elements:
        if time.time() > deadline - 5:
            break
        test = current + [a]
        if is_dissociated(test):
            current = test
            best_subset = list(current)
    
    # Strategy 2: Try to start with a large subset and prune
    # Start with all elements, remove ones that cause conflicts
    # This is harder, let's try a different approach
    
    # Strategy 3: Systematic search for maximum cardinality
    # Try sizes from 22 down to 1
    # For each size, try combinations (but this is too slow for large sizes)
    
    # Better: use the greedy result as a baseline, then try to improve
    # by swapping elements or adding more
    
    # Let's try a more systematic backtracking approach
    # Try to build a set of size k for k from high to low
    
    # First, let's see what size the greedy approach gives us
    greedy_size = len(best_subset)
    
    # Try to find a larger set using backtracking
    # We'll try to find a set of size greedy_size + 1, then greedy_size + 2, etc.
    
    # For backtracking, we need to be smart about ordering
    # Sort elements by value descending for pruning
    
    def backtrack(start_idx, current_subset, target_size):
        """Try to build a dissociated set of target_size starting from start_idx."""
        if time.time() > deadline - 3:
            return None
        
        if len(current_subset) == target_size:
            return list(current_subset)
        
        remaining_needed = target_size - len(current_subset)
        available = 22 - start_idx
        
        if available < remaining_needed:
            return None
        
        for i in range(start_idx, 22):
            if time.time() > deadline - 3:
                return None
            
            a = elements[i]
            test = current_subset + [a]
            
            # Quick check: if the new element's value is too small relative to others,
            # it might cause conflicts. But we need to check properly.
            
            if is_dissociated(test):
                result = backtrack(i + 1, test, target_size)
                if result is not None:
                    return result
        
        return None
    
    # Try to find larger sets
    for target_size in range(22, greedy_size, -1):
        if time.time() > deadline - 5:
            break
        
        result = backtrack(0, [], target_size)
        if result is not None:
            best_subset = result
            break
    
    # If we still have time, try to improve further
    if time.time() < deadline - 10 and len(best_subset) < 22:
        # Try random restarts with different orderings
        import random
        random.seed(42)
        
        for _ in range(100):
            if time.time() > deadline - 3:
                break
            
            # Shuffle elements
            shuffled = elements[:]
            random.shuffle(shuffled)
            
            current = []
            for a in shuffled:
                test = current + [a]
                if is_dissociated(test):
                    current = test
            
            if len(current) > len(best_subset):
                best_subset = current
    
    # Ensure we have at least one element
    if not best_subset:
        best_subset = [1]
    
    return best_subset

# EVOLVE-BLOCK-END
