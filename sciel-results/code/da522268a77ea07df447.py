"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    N = 512
    # Precompute: for each x, list of (d, y1, y2) such that x, x+d, x+2d, x+3d all in [1,N]
    # Actually we need: x is in a 4-AP iff exists d>0 with x-3d, x-2d, x-d, x all selected
    # or x, x+d, x+2d, x+3d all selected (x is the first term)
    # For greedy: when considering adding x, check if it completes a 4-AP
    
    # Build conflict structure: for each x, which other elements would form a 4-AP with x
    # A 4-AP is (a, a+d, a+2d, a+3d). 
    # If we add x, it could be:
    #   position 0: x, x+d, x+2d, x+3d -> need x+d, x+2d, x+3d already in set
    #   position 1: x-d, x, x+d, x+2d -> need x-d, x+d, x+2d in set
    #   position 2: x-2d, x-d, x, x+d -> need x-2d, x-d, x+d in set
    #   position 3: x-3d, x-2d, x-d, x -> need x-3d, x-2d, x-d in set
    
    # Greedy construction
    selected = set()
    for x in range(1, N+1):
        if time.time() > deadline - 5:
            break
        can_add = True
        # Check if adding x creates a 4-AP
        # x as 4th term (index 3): x-3d, x-2d, x-d, x
        for d in range(1, (x)//3 + 1):
            if (x-3*d in selected) and (x-2*d in selected) and (x-d in selected):
                can_add = False
                break
        if can_add:
            # x as 1st term (index 0): x, x+d, x+2d, x+3d
            for d in range(1, (N - x)//3 + 1):
                if (x+d in selected) and (x+2*d in selected) and (x+3*d in selected):
                    can_add = False
                    break
        if can_add:
            # x as 2nd term (index 1): x-d, x, x+d, x+2d
            for d in range(1, min(x-1, (N-x)//2) + 1):
                if (x-d in selected) and (x+d in selected) and (x+2*d in selected):
                    can_add = False
                    break
        if can_add:
            # x as 3rd term (index 2): x-2d, x-d, x, x+d
            for d in range(1, min((x-1)//2, N-x) + 1):
                if (x-2*d in selected) and (x-d in selected) and (x+d in selected):
                    can_add = False
                    break
        if can_add:
            selected.add(x)
    
    # Local search: try to add more elements or swap
    # First, try to add any element that was skipped
    for x in range(1, N+1):
        if time.time() > deadline - 3:
            break
        if x in selected:
            continue
        # Check if x can be added
        can_add = True
        for d in range(1, (x)//3 + 1):
            if (x-3*d in selected) and (x-2*d in selected) and (x-d in selected):
                can_add = False
                break
        if can_add:
            for d in range(1, (N - x)//3 + 1):
                if (x+d in selected) and (x+2*d in selected) and (x+3*d in selected):
                    can_add = False
                    break
        if can_add:
            for d in range(1, min(x-1, (N-x)//2) + 1):
                if (x-d in selected) and (x+d in selected) and (x+2*d in selected):
                    can_add = False
                    break
        if can_add:
            for d in range(1, min((x-1)//2, N-x) + 1):
                if (x-2*d in selected) and (x-d in selected) and (x+d in selected):
                    can_add = False
                    break
        if can_add:
            selected.add(x)
    
    return sorted(selected)
# EVOLVE-BLOCK-END
