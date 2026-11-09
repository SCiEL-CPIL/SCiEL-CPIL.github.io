"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _greedy(order):
    selected, reachable = [], {0}
    for x in order:
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    return selected

def construct():
    deadline = time.time() + 85
    best = []
    
    # Try multiple orderings
    orderings = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
        list(range(50, 0, -1)) + list(range(100, 50, -1)),
        list(range(1, 51)) + list(range(100, 50, -1)),
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        result = _greedy(order)
        if len(result) > len(best):
            best = result
    
    # Also try: include all large numbers, then greedily add small ones
    if time.time() <= deadline:
        selected = list(range(51, 101))  # 50 numbers
        reachable = {0}
        for x in selected:
            new = {s + x for s in reachable if s + x <= 200}
            reachable |= new
        
        for x in range(1, 51):
            if time.time() > deadline:
                break
            new = {s + x for s in reachable if s + x <= 200}
            if 200 not in new:
                selected.append(x)
                reachable |= new
        
        if len(selected) > len(best):
            best = selected
    
    return best if best else [100]
# EVOLVE-BLOCK-END
