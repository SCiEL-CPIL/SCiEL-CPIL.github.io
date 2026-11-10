"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_DEADLINE = 80.0  # safety margin before 90s kill

def _has_distinct_subset_products(nums):
    """Check if all 2^k subset products are distinct."""
    if not nums:
        return True
    products = {1}
    for x in nums:
        new_products = set()
        for p in products:
            np = p * x
            if np in products or np in new_products:
                return False
            new_products.add(np)
        products.update(new_products)
    return True

def _can_add(nums, candidate):
    """Check if adding candidate to nums preserves distinct subset products."""
    products = {1}
    for x in nums:
        new_products = set()
        for p in products:
            np = p * x
            new_products.add(np)
        products.update(new_products)
    # Now try adding candidate
    for p in list(products):
        np = p * candidate
        if np in products:
            return False
    return True

def _local_search(initial_set, candidates, deadline, rng):
    """Improve a set via swap-based local search."""
    current = list(initial_set)
    current_set = set(current)
    best = list(current)
    best_size = len(current)
    start_time = time.time()
    
    # Precompute all candidates not in current
    outside = [c for c in candidates if c not in current_set]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing one element and adding another
        for i, remove_elem in enumerate(current):
            if time.time() > deadline:
                break
            temp_set = [x for j, x in enumerate(current) if j != i]
            temp_set_set = set(temp_set)
            for add_elem in outside:
                if add_elem in temp_set_set:
                    continue
                trial = temp_set + [add_elem]
                if _has_distinct_subset_products(trial):
                    # Accept improvement if same size (swap) or better
                    if len(trial) > len(current):
                        current = trial
                        current_set = set(current)
                        outside = [c for c in candidates if c not in current_set]
                        if len(current) > best_size:
                            best_size = len(current)
                            best = list(current)
                        improved = True
                        break
            if time.time() > deadline:
                break
    
    return best

def _greedy_with_validation(candidates, deadline, rng):
    """Greedy construction with direct product validation."""
    best_set = []
    best_size = 0
    start_time = time.time()
    
    for attempt in range(200000):
        if time.time() > deadline:
            break
        
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        
        current_set = []
        
        for x in shuffled:
            if _can_add(current_set, x):
                current_set.append(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
            if best_size >= 13:
                break
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    all_candidates = list(range(2, 41))
    rng = random.Random(42)
    
    # Phase 1: Greedy with random restarts
    best_set = _greedy_with_validation(all_candidates, deadline, rng)
    
    # Phase 2: Local search to improve
    if time.time() < deadline - 5:
        improved = _local_search(best_set, all_candidates, deadline, rng)
        if len(improved) > len(best_set):
            best_set = improved
    
    # Final validation
    if not _has_distinct_subset_products(best_set):
        # Fallback: try a few more greedy attempts
        for _ in range(100):
            if time.time() > deadline - 2:
                break
            shuffled = all_candidates[:]
            rng.shuffle(shuffled)
            trial = []
            for x in shuffled:
                if _can_add(trial, x):
                    trial.append(x)
            if len(trial) > len(best_set) and _has_distinct_subset_products(trial):
                best_set = trial
    
    # Ensure we return a list of Python ints
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
