"""Rewritten program: maximize set size with cardinality-separated subset sums."""
import time
import random
from itertools import combinations

def _check_validity(elements):
    """
    Check if a set of elements satisfies the cardinality-separated subset sum property.
    For any two subsets of different cardinalities, their sums must differ.
    Equivalently: the set of sums achievable by subsets of size i must be disjoint
    from the set of sums achievable by subsets of size j for all i != j.
    """
    n = len(elements)
    # sums_by_card[c] = set of sums achievable by subsets of cardinality c
    sums_by_card = [set() for _ in range(n + 1)]
    sums_by_card[0].add(0)
    
    for elem in elements:
        # Update in reverse to avoid using the same element twice
        for c in range(n, 0, -1):
            if sums_by_card[c - 1]:
                for s in sums_by_card[c - 1]:
                    sums_by_card[c].add(s + elem)
    
    # Check pairwise disjointness
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if sums_by_card[i] & sums_by_card[j]:
                return False
    return True

def _max_valid_size_greedy(deadline):
    """
    Greedy construction: add elements one by one, choosing the smallest candidate
    that maintains validity. Try multiple random orderings and return best found.
    """
    best_set = []
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Randomize the candidate order for diversity
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        current_set = []
        for candidate in candidates:
            if time.time() > deadline:
                break
            
            # Skip if already in set
            if candidate in current_set:
                continue
            
            # Try adding candidate
            test_set = current_set + [candidate]
            
            # Quick check: only need to verify new sums don't collide with existing
            # For efficiency, we do a full check but it's fast for small sets
            if _check_validity(test_set):
                current_set = test_set
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]
        
        # Early stop if we hit the theoretical max
        if len(best_set) >= 60:
            break
    
    return best_set

def _local_search_improve(elements, deadline):
    """
    Try to improve a valid set by swapping elements or adding new ones.
    """
    elements = list(elements)
    n = len(elements)
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try adding a new element
        for candidate in range(1, 257):
            if candidate in elements:
                continue
            test_set = elements + [candidate]
            if _check_validity(test_set):
                elements = test_set
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping: remove one, add another
        for i in range(len(elements)):
            if time.time() > deadline:
                return elements
            removed = elements[i]
            remaining = elements[:i] + elements[i+1:]
            for candidate in range(1, 257):
                if candidate in remaining:
                    continue
                test_set = remaining + [candidate]
                if _check_validity(test_set):
                    elements = test_set
                    improved = True
                    break
            if improved:
                break
    
    return elements

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55 second deadline (5s safety margin)
    
    # Phase 1: Greedy construction with random restarts
    best_set = _max_valid_size_greedy(deadline - 10.0)
    
    if time.time() > deadline:
        return list(best_set)
    
    # Phase 2: Local search improvement
    improved_set = _local_search_improve(best_set, deadline)
    
    if len(improved_set) > len(best_set):
        best_set = improved_set
    
    # Ensure we return Python ints in a list
    result = [int(x) for x in best_set]
    
    return result
