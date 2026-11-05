from fractions import Fraction
import sys

sys.setrecursionlimit(10000)

def construct():
    best_cardinality = 0
    best_set = []
    current_set = []
    
    # Precompute reciprocals
    recips = {i: Fraction(1, i) for i in range(1, 25)}
    
    def backtrack(idx, current_sums):
        nonlocal best_cardinality, best_set, current_set
        
        # Pruning
        if len(current_set) + idx <= best_cardinality:
            return
            
        if idx == 0:
            if len(current_set) > best_cardinality:
                best_cardinality = len(current_set)
                best_set = list(current_set)
            return

        x = idx
        rx = recips[x]
        
        can_add = True
        if rx in current_sums:
            can_add = False
        else:
            for y in current_set:
                ry = recips[y]
                diff = ry - rx
                if diff > 0:
                    if diff in current_sums:
                        can_add = False
                        break
        
        if can_add:
            # Add x
            current_set.append(x)
            
            # Update sums
            # We need to create a new set of sums
            # To avoid modifying current_sums, we create new_sums
            # Optimization: if len(current_sums) is large, this is slow.
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + rx)
            
            backtrack(idx - 1, new_sums)
            
            # Remove x
            current_set.pop()
            
        # Skip x
        backtrack(idx - 1, current_sums)

    backtrack(24, {Fraction(0, 1)})
    
    return best_set
