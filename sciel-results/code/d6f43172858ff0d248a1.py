"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations

def construct():
    """
    Construct a set of distinct integers in 1..256 such that:
    1. All nonempty subset sums are distinct (no two subsets have the same sum).
    2. No two distinct subset sums divide one another.
    
    Strategy:
    - Use a superincreasing sequence to guarantee unique subset sums.
    - Greedily select elements from 1..256 in descending order.
    - For each candidate, check if adding it maintains:
      a) Unique subset sums (superincreasing property)
      b) No two subset sums divide each other
    """
    
    def all_subset_sums(elements):
        """Compute all nonempty subset sums."""
        sums = {0}
        for e in elements:
            new_sums = set()
            for s in sums:
                new_sums.add(s + e)
            sums = sums | new_sums
        sums.discard(0)
        return sums
    
    def check_divisibility(sums):
        """Check that no two distinct sums divide each other."""
        sum_list = sorted(sums)
        for i in range(len(sum_list)):
            for j in range(i + 1, len(sum_list)):
                if sum_list[j] % sum_list[i] == 0:
                    return False
        return True
    
    def try_greedy(order):
        """Try greedy construction with given candidate order."""
        selected = []
        current_sums = set()
        
        for candidate in order:
            if candidate in selected:
                continue
            
            # Compute new subset sums
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + candidate)
            new_sums.add(candidate)
            
            # Check uniqueness: with superincreasing, new_sums should have 2*len(current_sums)+1 elements
            if len(new_sums) != 2 * len(current_sums) + 1:
                continue
            
            # Check divisibility
            if not check_divisibility(new_sums):
                continue
            
            selected.append(candidate)
            current_sums = new_sums
        
        return selected
    
    # Try multiple orders to find the best set
    orders = [
        list(range(256, 0, -1)),  # Descending
        list(range(1, 257)),      # Ascending
    ]
    
    # Also try some heuristic orders
    # Prefer odd numbers, then even
    odds = list(range(255, 0, -2))
    evens = list(range(256, 0, -2))
    orders.append(odds + evens)
    orders.append(evens + odds)
    
    # Randomized orders
    for _ in range(10):
        random_order = list(range(1, 257))
        random.shuffle(random_order)
        orders.append(random_order)
    
    best = []
    for order in orders:
        result = try_greedy(order)
        if len(result) > len(best):
            best = result
    
    # Verify the best set
    if best:
        sums = all_subset_sums(best)
        if not check_divisibility(sums):
            # Fallback to a known valid set
            best = [31, 32, 34]
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best]

# EVOLVE-BLOCK-END
