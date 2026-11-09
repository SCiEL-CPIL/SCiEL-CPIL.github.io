"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds safety margin
    
    def is_valid_subset_sums(sums_set):
        """Check if a set of subset sums is valid (no divisibility, no duplicates)."""
        if len(sums_set) == 0:
            return True
        if len(sums_set) != len(list(sums_set)):
            return False
            
        # Check divisibility
        # For each a in sums_set, check if any multiple k*a (k>=2) is in sums_set
        max_sum = max(sums_set)
        for a in sums_set:
            for k in range(2, max_sum // a + 1):
                if (a * k) in sums_set:
                    return False
        return True

    def get_all_subset_sums(elements):
        """Compute all non-empty subset sums for a list of elements."""
        if not elements:
            return set()
        sums = set()
        for x in elements:
            # New sums are old_sums + x, plus x itself
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            new_sums.add(x)
            sums.update(new_sums)
        return sums

    def can_add_element(x, current_sums):
        """Check if x can be added to the current set of elements (represented by current_sums)."""
        # New sums will be current_sums ∪ {s+x for s in current_sums} ∪ {x}
        new_sums = set(current_sums)
        new_sums.add(x)
        for s in current_sums:
            new_sums.add(s + x)
        
        # Check for duplicates (if len(new_sums) < len(current_sums) + 1 + len(current_sums), there's a duplicate)
        # Actually, just check if the set size is as expected
        expected_size = len(current_sums) + 1 + len(current_sums)
        if len(new_sums) < expected_size:
            return False, None
            
        # Check divisibility
        if not is_valid_subset_sums(new_sums):
            return False, None
            
        return True, new_sums

    def greedy_search(order, time_limit):
        """Greedy search trying elements in the given order."""
        current_sums = set()
        chosen = []
        
        for x in order:
            if time.time() > time_limit:
                break
            ok, new_sums = can_add_element(x, current_sums)
            if ok:
                chosen.append(x)
                current_sums = new_sums
                
        return chosen

    # Try multiple random orders and keep the best result
    best_result = []
    
    # First, try the descending order (which is what the original did)
    desc_order = list(range(256, 0, -1))
    result = greedy_search(desc_order, deadline)
    if len(result) > len(best_result):
        best_result = result
    
    # Try several random permutations
    random.seed(42)  # Fixed seed for reproducibility
    for i in range(100):
        if time.time() > deadline - 5:
            break
            
        # Create a random order, but bias towards larger numbers
        # We can do this by sorting with a random key
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        # Try this order
        result = greedy_search(candidates, deadline)
        if len(result) > len(best_result):
            best_result = result
            # If we found a significantly better result, we can continue
            # but be careful with time
            
    # Also try some heuristic orders
    # e.g., prefer numbers that are not multiples of small primes
    # or numbers that are "large primes"
    
    # One more strategy: start with large numbers, then fill in smaller ones
    # This is similar to descending but with some randomness
    
    for i in range(20):
        if time.time() > deadline - 2:
            break
        
        # Create an order that starts with large numbers but has some randomness
        large_numbers = list(range(128, 257))
        small_numbers = list(range(1, 128))
        random.shuffle(large_numbers)
        random.shuffle(small_numbers)
        
        # Interleave or just concatenate
        order = large_numbers + small_numbers
        result = greedy_search(order, deadline)
        if len(result) > len(best_result):
            best_result = result

    # Ensure we return a list of Python ints
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
