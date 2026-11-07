"""Search-based maximization of a divisibility-incomparable subset-sum set."""
import time
import random
from math import gcd

DEADLINE_SECONDS = 85.0

def _has_divisibility_conflict(sums):
    """
    Given a set of positive ints (subset sums), return True if there exist
    two distinct values a < b with a | b.
    Optimized: Iterate through sums, check if any multiple is in the set.
    """
    if len(sums) < 2:
        return False
    
    # Get max sum to bound the multiple checks
    max_sum = max(sums)
    
    # To optimize, we can iterate through the sums.
    # If 'a' is in sums, we check 2a, 3a, ... in sums.
    # We don't need to sort for this logic, but iterating smaller 'a' first
    # might find conflicts faster if they exist?
    # Actually, if a large 'a' is in sums, its multiples are larger and less likely to be in sums
    # if sums are dense? Not necessarily.
    # Just iterate.
    
    for a in sums:
        # Check multiples 2a, 3a, ...
        # We can break if a*2 > max_sum
        if a * 2 > max_sum:
            continue
        multiple = a * 2
        while multiple <= max_sum:
            if multiple in sums:
                return True
            multiple += a
    return False

def _can_add(current_sums, x):
    """
    Check if adding element x to a set with nonempty-subset-sums `current_sums`
    keeps the divisibility-incomparability property.
    New sums = current_sums ∪ {x} ∪ {s + x for s in current_sums}.
    """
    new_sums = set(current_sums)
    new_sums.add(x)
    for s in current_sums:
        new_sums.add(s + x)
    if _has_divisibility_conflict(new_sums):
        return None
    return new_sums

def _greedy_construct(candidates, start_time, deadline):
    """
    Greedy: try to add candidates in given order, keep the set that grows largest.
    Returns (set_of_elements, set_of_sums).
    """
    elements = []
    sums = set()
    for x in candidates:
        if time.time() > deadline:
            break
        new_sums = _can_add(sums, x)
        if new_sums is not None:
            elements.append(x)
            sums = new_sums
    return elements, sums

def _random_order(pool):
    """Return a random permutation of pool."""
    lst = list(pool)
    random.shuffle(lst)
    return lst

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    # Candidate pool: integers in 1..256.
    # To maximize size, we want many elements. Larger elements reduce subset-sum
    # collisions and divisibility conflicts? Actually smaller elements create more
    # subset sums (more chances for divisibility). Larger elements create sums that
    # are large and less likely to divide each other. But we need distinct ints in 1..256.
    # Strategy: try multiple random orderings of a subset of candidates.
    # Use a pool of candidates. We'll use all 1..256 but random orderings.

    pool = list(range(1, 257))

    best_elements = []
    best_sums = set()

    # First: a deterministic greedy with descending order (larger elements first)
    # Larger elements first may allow more elements because sums grow faster and
    # divisibility conflicts are less likely among large numbers.
    order = sorted(pool, reverse=True)
    elems, sums = _greedy_construct(order, start_time, deadline)
    if len(elems) > len(best_elements):
        best_elements = elems
        best_sums = sums

    # Also try ascending order
    if time.time() < deadline:
        order = sorted(pool)
        elems, sums = _greedy_construct(order, start_time, deadline)
        if len(elems) > len(best_elements):
            best_elements = elems
            best_sums = sums

    # Random restarts
    # With faster conflict check, we can do more restarts.
    # We also try different strategies:
    # 1. Pure random
    # 2. Random with bias towards larger numbers (they are "safer")
    
    max_restarts = 5000
    for r in range(max_restarts):
        if time.time() > deadline - 0.5:
            break
        
        # Mix of random and biased orders
        if r % 3 == 0:
            # Descending bias: shuffle but keep larger numbers more likely to be early?
            # Or just sort descending and shuffle a prefix?
            # Let's just use random for now, but maybe weight by value?
            # Simple random is robust.
            order = _random_order(pool)
        elif r % 3 == 1:
            # Descending order with slight noise
            order = sorted(pool, reverse=True)
            # Shuffle first 20 elements to break ties/blocks
            if len(order) > 20:
                random.shuffle(order[:20])
        else:
            # Ascending order with slight noise
            order = sorted(pool)
            if len(order) > 20:
                random.shuffle(order[:20])
                
        elems, sums = _greedy_construct(order, start_time, deadline)
        if len(elems) > len(best_elements):
            best_elements = elems
            best_sums = sums

    # Final: try local improvement on best set by adding remaining elements in random order
    # Try multiple times with different shuffles of remaining elements
    if time.time() < deadline - 0.5:
        used = set(best_elements)
        remaining = [x for x in pool if x not in used]
        
        # Try a few random orderings of the remaining elements
        for _ in range(5):
            if time.time() > deadline - 0.2:
                break
            random.shuffle(remaining)
            elems = list(best_elements)
            sums = set(best_sums)
            for x in remaining:
                if time.time() > deadline - 0.2:
                    break
                new_sums = _can_add(sums, x)
                if new_sums is not None:
                    elems.append(x)
                    sums = new_sums
            if len(elems) > len(best_elements):
                best_elements = elems
                best_sums = sums
                # Update remaining list for next iteration?
                # The set of "used" changes, so "remaining" should be updated.
                # But best_elements is updated, so let's recompute remaining if we improved.
                if len(elems) > len(best_elements): # This is always true if we updated
                    used = set(best_elements)
                    remaining = [x for x in pool if x not in used]

    # Ensure return type is list of Python ints
    return list(best_elements)
