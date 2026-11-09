import itertools
from fractions import Fraction
import time

def construct():
    """
    Find the largest subset of [1, 24] such that no reciprocal of a selected element
    equals the sum of reciprocals of a subset of the OTHER selected elements.
    """
    N = 24
    elements = list(range(1, N + 1))
    
    # Precompute reciprocals as Fractions for exact arithmetic
    recips = {x: Fraction(1, x) for x in elements}
    
    # State:
    # current_set: list of selected elements
    # sums: set of all subset sums of reciprocals of current_set
    # sums_excl: dict mapping each y in current_set to the set of sums of S\{y}
    
    best_size = 0
    best_set = []
    
    def backtrack(idx, current_set, sums, sums_excl):
        """
        idx: next element index to consider (0-based, corresponding to element idx+1)
        current_set: list of selected elements
        sums: set of all subset sums of reciprocals of current_set
        sums_excl: dict {y: Sums(S\{y}) for y in current_set}
        """
        nonlocal best_size, best_set
        
        # Pruning: if even taking all remaining elements can't beat best_size, return
        remaining = 24 - idx
        if len(current_set) + remaining <= best_size:
            return
        
        if idx == 24:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
        
        # Option 1: Skip element idx+1
        backtrack(idx + 1, current_set, sums, sums_excl)
        
        # Option 2: Take element idx+1
        x = idx + 1
        rx = recips[x]
        
        # Check if rx is in sums (i.e., 1/x is a sum of subset of current_set)
        if rx in sums:
            return
            
        # Check if adding x breaks any existing y
        # For each y in current_set, we need 1/y - 1/x NOT to be in sums_excl[y]
        valid = True
        for y in current_set:
            ry = recips[y]
            diff = ry - rx
            if diff in sums_excl[y]:
                valid = False
                break
        
        if not valid:
            return
        
        # Commit x
        new_current_set = current_set + [x]
        
        # Update sums
        new_sums = set(sums)
        for s in sums:
            new_sums.add(s + rx)
        new_sums.add(rx)
        
        # Update sums_excl
        new_sums_excl = {}
        for y in current_set:
            old_excl = sums_excl[y]
            new_excl = set(old_excl)
            for s in old_excl:
                new_excl.add(s + rx)
            new_excl.add(rx)
            new_sums_excl[y] = new_excl
            
        # For the new element x, sums_excl[x] is just the old `sums`
        # Because current_set excluding x is the old current_set
        new_sums_excl[x] = set(sums)
        
        backtrack(idx + 1, new_current_set, new_sums, new_sums_excl)
        
    # Start backtracking
    backtrack(1, [], set(), {})
    
    if not best_set:
        return [1]
        
    return best_set

# EVOLVE-BLOCK-END
