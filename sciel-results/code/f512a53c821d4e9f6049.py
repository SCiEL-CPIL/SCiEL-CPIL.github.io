"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    def check_dissociated(elements):
        """Check if all subset sums of reciprocals are distinct."""
        k = len(elements)
        if k == 0:
            return True
        reciprocals = [Fraction(1, x) for x in elements]
        # Generate all subset sums
        sums = set()
        sums.add(Fraction(0, 1))
        for r in reciprocals:
            new_sums = set()
            for s in sums:
                ns = s + r
                if ns in sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return len(sums) == (1 << k)
    
    # Greedy construction: try to build a large dissociated set
    # Start with all candidates and greedily add elements
    candidates = list(range(1, 23))
    
    best_set = []
    
    # Try different orderings to find a large set
    # Strategy: prefer larger denominators (smaller reciprocals) as they're less likely to cause collisions
    for order in [sorted(candidates, reverse=True), 
                  sorted(candidates),
                  [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]]:
        if time.time() > deadline:
            break
        
        current = []
        for elem in order:
            if time.time() > deadline:
                break
            if elem in current:
                continue
            test_set = current + [elem]
            if check_dissociated(test_set):
                current = test_set
                if len(current) > len(best_set):
                    best_set = current[:]
        
        if len(best_set) >= 12:  # Good enough, stop early
            break
    
    if not best_set:
        # Fallback to known good set
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best_set
# EVOLVE-BLOCK-END
