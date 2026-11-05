"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Initial Sidon set with prescribed elements
    a = [1, 2, 5, 11]
    sums = set()
    for i, x in enumerate(a):
        for y in a[i:]:
            sums.add(x + y)
    
    best_size = len(a)
    best_set = a[:]
    
    # Greedy construction to get a good starting point
    for x in range(12, 285):
        new = {x + y for y in a} | {2 * x}
        if len(new) == len(a) + 1 and not new & sums:
            a.append(x)
            sums |= new
    
    if len(a) > best_size:
        best_size = len(a)
        best_set = a[:]
    
    # Try to improve with backtracking search
    # We'll try to add elements in a smarter way
    
    def can_add(candidate, current_set, current_sums):
        """Check if candidate can be added to current_set"""
        for y in current_set:
            if y > candidate:
                continue
            s = candidate + y
            if s in current_sums:
                return False
        # Check 2*candidate
        if 2 * candidate in current_sums:
            return False
        return True
    
    def add_element(candidate, current_set, current_sums):
        """Add candidate to current_set and return updated sums"""
        new_sums = set(current_sums)
        for y in current_set:
            if y > candidate:
                continue
            new_sums.add(candidate + y)
        new_sums.add(2 * candidate)
        return new_sums
    
    # Try local search: remove one element and try to add multiple others
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        if len(best_set) <= best_size:
            continue
            
        # Try removing one element and rebuilding
        for remove_idx in range(len(best_set)):
            if time.time() > deadline:
                break
                
            test_set = best_set[:remove_idx] + best_set[remove_idx+1:]
            test_sums = set()
            for i, x in enumerate(test_set):
                for y in test_set[i:]:
                    test_sums.add(x + y)
            
            # Try to greedily add elements
            for x in range(12, 285):
                if x in test_set:
                    continue
                if can_add(x, test_set, test_sums):
                    test_set.append(x)
                    test_sums = add_element(x, test_set[:-1], test_sums)
            
            if len(test_set) > best_size:
                best_size = len(test_set)
                best_set = test_set[:]
    
    return best_set
# EVOLVE-BLOCK-END
