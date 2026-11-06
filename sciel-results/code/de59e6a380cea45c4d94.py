import time
import sys
import random
from math import gcd
from functools import reduce

# Set recursion limit
sys.setrecursionlimit(10000)

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin before 90s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    """
    Returns a nonempty subset of [1, 22] with all reciprocal subset sums distinct.
    Maximizes cardinality.
    
    Strategy: Multi-start randomized greedy with local search.
    1. Convert reciprocal sums to integer sums by multiplying by L = LCM(1..22).
    2. Try multiple random orderings of the 22 elements.
    3. For each ordering, use greedy construction to build a valid set.
    4. Apply local search (swap/add/remove) to improve the solution.
    5. Keep the best valid solution found within the time budget.
    """
    
    # Precompute LCM of 1..22
    L = reduce(_lcm, range(1, 23), 1)
    
    # Weights w[i] = L // i
    weights = {i: L // i for i in range(1, 23)}
    
    # Helper to check if a list of indices forms a dissociated set
    # Returns (is_valid, all_sums) where all_sums is the set of subset sums if valid
    def _check_dissociated(indices):
        sums = {0}
        for idx in indices:
            w = weights[idx]
            new_sums = set()
            for s in sums:
                ns = s + w
                new_sums.add(ns)
            # If any new sum is in existing sums, collision
            if new_sums & sums:
                return False, None
            sums = sums.union(new_sums)
        return True, sums
    
    # Helper to check if adding a weight to existing sums causes collision
    def _can_add(current_sums, w):
        for s in current_sums:
            if (s + w) in current_sums:
                return False
        return True
    
    # Helper to compute new sums after adding a weight (assumes no collision)
    def _add_weight(current_sums, w):
        new_sums = set(current_sums)
        for s in current_sums:
            new_sums.add(s + w)
        return new_sums
    
    # Greedy construction given an ordering of indices
    def _greedy_build(ordering):
        current_set = []
        current_sums = {0}
        for idx in ordering:
            w = weights[idx]
            if _can_add(current_sums, w):
                current_set.append(idx)
                current_sums = _add_weight(current_sums, w)
        return current_set, current_sums
    
    # Local search: try to improve a set by swapping elements
    def _local_search(current_set, current_sums, max_iterations=1000):
        best_set = list(current_set)
        best_sums = set(current_sums)
        best_size = len(current_set)
        
        for _ in range(max_iterations):
            if _time_left() < 0.5:
                break
            
            improved = False
            current_set_size = len(best_set)
            
            # Try adding elements not in the set
            for idx in range(1, 23):
                if idx in best_set:
                    continue
                w = weights[idx]
                if _can_add(best_sums, w):
                    # Try adding this element
                    new_set = best_set + [idx]
                    new_sums = _add_weight(best_sums, w)
                    
                    # Try to further improve by swapping
                    for remove_idx in list(best_set):
                        if remove_idx == idx:
                            continue
                        # Try: remove remove_idx, add idx
                        new_set2 = [x for x in new_set if x != remove_idx]
                        # Recompute sums for new_set2
                        valid, sums2 = _check_dissociated(new_set2)
                        if valid and len(new_set2) > best_size:
                            best_set = new_set2
                            best_sums = sums2
                            best_size = len(best_set)
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best_set, best_sums
    
    # Main search: multi-start randomized greedy with local search
    best_set = [1]
    best_sums = {0, weights[1]}
    best_size = 1
    
    # Try deterministic orderings first
    deterministic_orderings = [
        list(range(1, 23)),           # ascending
        list(range(22, 0, -1)),       # descending
        list(range(1, 23, 2)) + list(range(2, 23, 2)),  # odds then evens
        list(range(22, 0, -2)) + list(range(21, 0, -2)), # evens then odds
    ]
    
    for ordering in deterministic_orderings:
        if _time_left() < 1.0:
            break
        
        set_candidate, sums_candidate = _greedy_build(ordering)
        if len(set_candidate) > best_size:
            best_set = set_candidate
            best_sums = sums_candidate
            best_size = len(best_set)
        
        # Apply local search
        improved_set, improved_sums = _local_search(best_set, best_sums)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_sums = improved_sums
            best_size = len(best_set)
    
    # Try randomized orderings
    random.seed(42)
    num_random_trials = 500
    
    for trial in range(num_random_trials):
        if _time_left() < 1.0:
            break
        
        # Random ordering
        ordering = list(range(1, 23))
        random.shuffle(ordering)
        
        set_candidate, sums_candidate = _greedy_build(ordering)
        if len(set_candidate) > best_size:
            best_set = set_candidate
            best_sums = sums_candidate
            best_size = len(best_set)
        
        # Apply local search
        improved_set, improved_sums = _local_search(best_set, best_sums)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_sums = improved_sums
            best_size = len(best_set)
        
        # Early termination if we found a large set
        if best_size >= 12:
            break
    
    # Final verification
    valid, _ = _check_dissociated(best_set)
    if valid:
        return best_set
    
    # Fallback
    return [1]

if __name__ == "__main__":
    s = construct()
    print(f"Found set: {s}")
    print(f"Size: {len(s)}")
    # Verify
    L = reduce(_lcm, range(1, 23), 1)
    sums = {0}
    valid = True
    for x in s:
        w = L // x
        new_sums = set()
        for ss in sums:
            ns = ss + w
            new_sums.add(ns)
        if new_sums & sums:
            valid = False
            break
        sums = sums.union(new_sums)
    print(f"Valid: {valid}")
    print(f"Num sums: {len(sums)} (Expected {1 << len(s)})")
