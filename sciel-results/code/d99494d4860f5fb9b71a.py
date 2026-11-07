"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    n = 256
    deadline = time.time() + 50  # safety margin
    
    # Greedy approach: process numbers in descending order
    # Include a number if it's coprime to all already-selected numbers
    selected = []
    selected_set = set()
    
    for a in range(n, 0, -1):
        if time.time() > deadline:
            break
        
        # Check if a is coprime to all selected numbers
        can_add = True
        for b in selected:
            if math.gcd(a, b) > 1:
                can_add = False
                break
        
        if can_add:
            selected.append(a)
            selected_set.add(a)
    
    # Ensure we return Python ints in a list
    return [int(x) for x in selected]
# EVOLVE-BLOCK-END
