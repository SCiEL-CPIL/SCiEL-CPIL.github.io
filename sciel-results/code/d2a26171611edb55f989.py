"""Search-based approach to maximize set size under divisibility constraint."""
import time
import random

DEADLINE_SECONDS = 85.0

def _subset_sums(elements):
    """Compute all nonempty subset sums for a list of elements."""
    n = len(elements)
    sums = set()
    # Use iterative approach: for each element, add it to all existing sums
    current = set()
    for e in elements:
        new_sums = set()
        for s in current:
            new_sums.add(s + e)
        new_sums.add(e)
        current = current | new_sums
        sums = current
    return sums

def _check_no_divisibility(sums):
    """Check that no two distinct sums in the set divide each other."""
    s_list = sorted(sums)
    n = len(s_list)
    for i in range(n):
        a = s_list[i]
        for j in range(i + 1, n):
            b = s_list[j]
            if b % a == 0:
                return False
    return True

def _is_valid(elements):
    """Check if the set of elements satisfies the divisibility constraint."""
    if len(elements) == 0:
        return True
    sums = _subset_sums(elements)
    return _check_no_divisibility(sums)

def _can_add(elements, new_elem, existing_sums):
    """Check if adding new_elem to elements maintains validity.
    existing_sums is the set of all nonempty subset sums of elements."""
    # New sums after adding new_elem:
    # - new_elem itself
    # - s + new_elem for each s in existing_sums
    new_sums_set = set(existing_sums)
    new_sums_set.add(new_elem)
    for s in existing_sums:
        new_sums_set.add(s + new_elem)
    
    # Check: no two distinct sums in new_sums_set divide each other
    # We only need to check pairs where at least one is new
    # New elements in the sum set: new_elem and {s + new_elem for s in existing_sums}
    new_only = {new_elem} | {s + new_elem for s in existing_sums}
    
    # For each new sum, check against all sums in new_sums_set
    for ns in new_only:
        for other in new_sums_set:
            if ns == other:
                continue
            if other % ns == 0:
                return False
            if ns % other == 0:
                return False
    return True

def _greedy_from_order(order, deadline):
    """Greedily build a valid set by trying elements in the given order."""
    elements = []
    existing_sums = set()
    
    for elem in order:
        if time.time() > deadline:
            break
        if elem in elements:
            continue
        if _can_add(elements, elem, existing_sums):
            elements.append(elem)
            # Update existing_sums
            new_sums = set(existing_sums)
            new_sums.add(elem)
            for s in existing_sums:
                new_sums.add(s + elem)
            existing_sums = new_sums
    
    return elements

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    best = []
    best_size = 0
    
    # Multiple random restarts with different orderings
    random.seed(42)
    
    # First, try a few structured approaches
    # Approach 1: Try elements in various sorted orders
    candidates = list(range(1, 257))
    
    # Try ascending, descending, and random orders
    orders = [
        candidates,
        list(reversed(candidates)),
    ]
    
    # Add several random permutations
    for _ in range(20):
        if time.time() > deadline:
            break
        order = candidates[:]
        random.shuffle(order)
        orders.append(order)
    
    for order in orders:
        if time.time() > deadline:
            break
        result = _greedy_from_order(order, deadline)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Try local improvement: try to add more elements to the best set found
    if best_size > 0:
        # Try adding remaining elements in random order
        remaining = [x for x in range(1, 257) if x not in best]
        random.shuffle(remaining)
        
        current = list(best)
        existing_sums = _subset_sums(current)
        
        for elem in remaining:
            if time.time() > deadline:
                break
            if _can_add(current, elem, existing_sums):
                current.append(elem)
                new_sums = set(existing_sums)
                new_sums.add(elem)
                for s in existing_sums:
                    new_sums.add(s + elem)
                existing_sums = new_sums
        
        if len(current) > best_size:
            best = current
            best_size = len(current)
    
    # Final verification
    if best:
        if not _is_valid(best):
            # Fallback to a known valid small set
            best = [31, 32, 34]
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best]
