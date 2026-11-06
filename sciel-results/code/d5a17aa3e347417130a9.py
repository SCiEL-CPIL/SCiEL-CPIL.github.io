import time
import numpy as np
from itertools import combinations
import random
import math

# EVOLVE-BLOCK-START
def _precompute_weights():
    # LCM of 1..22
    L = 1
    for i in range(1, 23):
        L = math.lcm(L, i)
    weights = {}
    for i in range(1, 23):
        weights[i] = L // i
    return weights

_GLOBAL_WEIGHTS = _precompute_weights()

def _is_dissociated_subset(S):
    """
    Checks if a set of integers S is dissociated in the reciprocals.
    Uses pre-computed global weights for speed.
    Optimized with early termination and efficient subset-sum generation.
    """
    if not S:
        return True
    
    # Get the weights for the elements in S
    weights = [_GLOBAL_WEIGHTS[s] for s in S]
    n = len(weights)
    
    # For small sets, use a fast approach with a set of sums
    seen = set()
    seen.add(0)
    
    current_sums = [0]
    for w in weights:
        new_sums = [s + w for s in current_sums]
        for ns in new_sums:
            if ns in seen:
                return False
            seen.add(ns)
        current_sums.extend(new_sums)
        
    return True

def _can_add_to_dissociated(S, e):
    """
    Fast check: can we add element e to the dissociated set S?
    Only checks if adding e creates a collision with existing subset sums.
    """
    if e in S:
        return False
    
    w = _GLOBAL_WEIGHTS[e]
    
    # Generate all subset sums of S
    weights = [_GLOBAL_WEIGHTS[s] for s in S]
    
    seen = set()
    seen.add(0)
    
    current_sums = [0]
    for wt in weights:
        new_sums = [s + wt for s in current_sums]
        for ns in new_sums:
            if ns in seen:
                # S itself is not dissociated (shouldn't happen if S is valid)
                return False
            seen.add(ns)
        current_sums.extend(new_sums)
    
    # Now check if adding w creates any collision
    for s in current_sums:
        if s + w in seen:
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit
    
    best_set = []
    best_size = 0
    
    def try_greedy(elements_order):
        current_set = set()
        for e in elements_order:
            if e in current_set:
                continue
            if _can_add_to_dissociated(current_set, e):
                current_set.add(e)
        return current_set

    # Run multiple random restarts with different strategies
    all_elements = list(range(1, 23))
    
    # Heuristic orders first
    heuristic_orders = [
        list(range(1, 23)),           # Ascending
        list(range(22, 0, -1)),       # Descending
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],  # Odds then evens
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],  # Evens then odds
    ]
    
    for order in heuristic_orders:
        if time.time() > deadline:
            break
        current_set = try_greedy(order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
    
    # Random restarts
    random.seed(42)
    for _ in range(200):
        if time.time() > deadline:
            break
        elements = all_elements[:]
        random.shuffle(elements)
        current_set = try_greedy(elements)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
    
    # Local search: try to expand the best set
    if best_set:
        current_set = set(best_set)
        improved = True
        while improved:
            improved = False
            for e in range(1, 23):
                if e not in current_set and _can_add_to_dissociated(current_set, e):
                    current_set.add(e)
                    improved = True
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = list(current_set)
        
        # Try swaps: remove one element, try to add two others
        # This can help escape local optima
        for _ in range(50):
            if time.time() > deadline:
                break
            if not current_set:
                break
            
            # Pick a random element to remove
            to_remove = random.choice(list(current_set))
            trial_set = current_set - {to_remove}
            
            # Try to add any two elements
            added_any = False
            for e in range(1, 23):
                if e not in trial_set and _can_add_to_dissociated(trial_set, e):
                    trial_set.add(e)
                    added_any = True
            
            # Check if we improved
            if len(trial_set) > len(current_set):
                current_set = trial_set
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                # Continue local search from new state
                improved2 = True
                while improved2:
                    improved2 = False
                    for e in range(1, 23):
                        if e not in current_set and _can_add_to_dissociated(current_set, e):
                            current_set.add(e)
                            improved2 = True
                            if len(current_set) > best_size:
                                best_size = len(current_set)
                                best_set = list(current_set)
            elif len(trial_set) == len(current_set) and random.random() < 0.3:
                # Occasionally accept same-size swaps to explore
                current_set = trial_set
    
    if not best_set:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
# EVOLVE-BLOCK-END
