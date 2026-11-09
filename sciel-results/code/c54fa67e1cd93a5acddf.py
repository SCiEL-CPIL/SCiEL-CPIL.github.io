"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

_TARGET = 200
_UNIVERSE = list(range(1, 101))
_MASK = (1 << (_TARGET + 1)) - 1  # Mask to keep bits 0..200


def _reachable_sums(selected):
    """Compute all reachable subset sums using bitset."""
    reachable = 1
    for x in selected:
        reachable |= (reachable << x)
        reachable &= _MASK
    return reachable


def _has_sum_200(selected):
    """Check if any subset of selected sums to 200."""
    return bool(_reachable_sums(selected) & (1 << _TARGET))


def _greedy_blocking_set():
    """
    Greedy set-cover to find a minimum-size blocking set B such that
    [1,100] \\ B has no subset summing to 200.
    
    Strategy: iteratively find a 200-summing subset S in the current universe,
    then pick the element in S that appears in the most 200-summing subsets
    (or heuristically, the element that "blocks" the most potential sums).
    Remove it from the universe and repeat until no 200-sum remains.
    """
    universe = set(_UNIVERSE)
    blocking_set = []
    
    # Precompute: for each element, count how many 200-sums it participates in
    # This is expensive, so we'll use a heuristic instead
    
    while universe:
        # Find a 200-summing subset in the current universe
        # Use greedy: try to build a sum of 200
        found_subset = _find_200_subset(universe)
        if found_subset is None:
            break
        
        # Pick the element in found_subset to remove
        # Heuristic: pick the element that appears in the most potential 200-sums
        # Approximation: pick the largest element (removes more "room" for sums)
        # Or: pick the element that is in the most minimal 200-sums
        # For now, pick the largest element in the found subset
        candidate = max(found_subset)
        
        # Alternative: try each element and see which removal "destroys" the most sums
        # This is expensive, so we'll use a simpler heuristic
        blocking_set.append(candidate)
        universe.discard(candidate)
    
    return blocking_set


def _find_200_subset(universe):
    """
    Find a subset of universe that sums to 200.
    Returns the subset or None if no such subset exists.
    """
    if not universe:
        return None
    
    # Use meet-in-the-middle or greedy
    # For small sets, we can use a simpler approach
    
    # Sort elements
    elements = sorted(universe, reverse=True)
    
    # Greedy: try to build sum 200
    # This is NP-hard in general, but for our specific case, we can use heuristics
    
    # Try all subsets of size 1, 2, 3, ... up to some limit
    # But that's exponential. Instead, use a bitset DP to find if 200 is reachable,
    # and if so, backtrack to find the subset.
    
    # First, check if 200 is reachable
    reachable = 1
    for x in elements:
        reachable |= (reachable << x)
        reachable &= _MASK
    
    if not (reachable & (1 << _TARGET)):
        return None
    
    # Backtrack to find a subset
    # This is tricky with bitset. Let's use a different approach.
    
    # Use a recursive search with memoization, but that might be slow.
    # Instead, use a greedy approach: try to build sum 200 greedily.
    
    # Greedy: start from largest, add if doesn't exceed 200, then adjust
    # This won't always find a solution, but it's fast.
    
    # Better: use a DP table to track which elements are used
    # But that's 100 * 200 = 20000 states, which is fine.
    
    # Let's use a different approach: enumerate subsets of small size first
    # Most 200-sums will have small cardinality
    
    # Try size 1
    for x in elements:
        if x == _TARGET:
            return [x]
    
    # Try size 2
    n = len(elements)
    for i in range(n):
        for j in range(i + 1, n):
            if elements[i] + elements[j] == _TARGET:
                return [elements[i], elements[j]]
    
    # Try size 3
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                s = elements[i] + elements[j] + elements[k]
                if s == _TARGET:
                    return [elements[i], elements[j], elements[k]]
                if s > _TARGET:
                    break
    
    # Try size 4
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                s3 = elements[i] + elements[j] + elements[k]
                if s3 >= _TARGET:
                    break
                for l in range(k + 1, n):
                    if s3 + elements[l] == _TARGET:
                        return [elements[i], elements[j], elements[k], elements[l]]
                    if s3 + elements[l] > _TARGET:
                        break
    
    # If we didn't find a small subset, use a more general approach
    # Use a DP to find any subset that sums to 200
    
    # dp[s] = list of elements used to achieve sum s, or None
    dp = [None] * (_TARGET + 1)
    dp[0] = []
    
    for x in elements:
        # Update dp in reverse to avoid using x multiple times
        for s in range(_TARGET, x - 1, -1):
            if dp[s - x] is not None and dp[s] is None:
                dp[s] = dp[s - x] + [x]
    
    if dp[_TARGET] is not None:
        return dp[_TARGET]
    
    return None


def _greedy_blocking_set_v2():
    """
    Improved greedy blocking set:
    1. Find a 200-summing subset S
    2. For each element x in S, compute a "criticality score" 
       (e.g., how many 200-sums contain x)
    3. Remove the element with highest criticality
    4. Repeat until no 200-sum remains
    """
    universe = set(_UNIVERSE)
    blocking_set = []
    
    # Precompute: for each element, count how many 200-sums it participates in
    # This is expensive, so we'll approximate
    
    # Alternative: use a different heuristic
    # Remove elements in descending order, checking if 200-sum still exists
    # This is equivalent to the original greedy, but from the "blocking" perspective
    
    # Let's try a different approach: 
    # Start with all elements, and remove elements one by one
    # in an order that minimizes the number of 200-sums destroyed
    # until no 200-sum remains
    
    # Actually, the simplest approach is:
    # Greedily remove elements from [1,100] in some order,
    # and stop when no 200-sum remains.
    # The order determines the size of the blocking set.
    
    # Try different orders and keep the best (smallest blocking set)
    best_blocking = None
    best_size = 101
    
    orders = [
        list(range(100, 0, -1)),  # Descending
        list(range(1, 101)),      # Ascending
        list(range(100, 0, -2)) + list(range(99, 0, -2)),  # Evens then odds
        list(range(99, 0, -2)) + list(range(100, 0, -2)),  # Odds then evens
    ]
    
    for order in orders:
        universe = set(_UNIVERSE)
        blocking = []
        
        for x in order:
            if x not in universe:
                continue
            # Check if 200-sum still exists in universe
            if _has_sum_200(list(universe)):
                # Remove x
                universe.discard(x)
                blocking.append(x)
            else:
                break
        
        if len(blocking) < best_size:
            best_size = len(blocking)
            best_blocking = blocking
    
    return best_blocking


def _construct_from_blocking(blocking):
    """Given a blocking set, return the complement."""
    blocking_set = set(blocking)
    return [x for x in _UNIVERSE if x not in blocking_set]


def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # Strategy 1: Greedy blocking set with different orders
    best_candidate = None
    best_size = -1
    
    # Try multiple blocking set constructions
    for seed in range(50):
        if time.time() > deadline - 5:
            break
        
        # Use a randomized order
        rng = random.Random(seed)
        order = list(range(1, 101))
        rng.shuffle(order)
        
        universe = set(_UNIVERSE)
        blocking = []
        
        for x in order:
            if x not in universe:
                continue
            # Check if 200-sum still exists in universe
            if _has_sum_200(list(universe)):
                universe.discard(x)
                blocking.append(x)
            else:
                break
        
        candidate = _construct_from_blocking(blocking)
        if len(candidate) > best_size:
            best_size = len(candidate)
            best_candidate = candidate
    
    # Strategy 2: Original greedy approach (for comparison)
    if time.time() < deadline - 10:
        # Greedy descending
        selected = []
        reachable = 1
        for x in range(100, 0, -1):
            if ((reachable << x) & (1 << _TARGET)) == 0:
                selected.append(x)
                reachable |= (reachable << x)
                reachable &= _MASK
        
        # Local improvement
        selected_set = set(selected)
        excluded = [x for x in _UNIVERSE if x not in selected_set]
        reachable = 1
        for x in selected:
            reachable |= (reachable << x)
            reachable &= _MASK
        
        improved = True
        while improved:
            improved = False
            for x in excluded:
                if x in selected_set:
                    continue
                new_reachable = reachable | (reachable << x)
                if (new_reachable & (1 << _TARGET)) == 0:
                    selected_set.add(x)
                    reachable = new_reachable & _MASK
                    improved = True
        
        if len(selected_set) > best_size:
            best_size = len(selected_set)
            best_candidate = list(selected_set)
    
    # Final safety check
    if best_candidate is None or _has_sum_200(best_candidate):
        # Fallback: use a known valid set
        best_candidate = [100]
    
    return best_candidate
# EVOLVE-BLOCK-END
