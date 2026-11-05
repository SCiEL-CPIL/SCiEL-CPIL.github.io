"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85
    
    def greedy(order):
        selected, reachable = [], {0}
        for x in order:
            new = {s + x for s in reachable if s + x <= 200}
            if 200 not in new:
                selected.append(x)
                reachable |= new
        return selected, reachable
    
    best = None
    best_reach = None
    
    # Original descending order
    sel, reach = greedy(list(range(100, 0, -1)))
    best, best_reach = sel, reach
    
    # Try ascending
    if time.time() < deadline:
        sel, reach = greedy(list(range(1, 101)))
        if len(sel) > len(best):
            best, best_reach = sel, reach
    
    # Try random permutations
    rng = random.Random(42)
    for _ in range(50):
        if time.time() >= deadline:
            break
        order = list(range(1, 101))
        rng.shuffle(order)
        sel, reach = greedy(order)
        if len(sel) > len(best):
            best, best_reach = sel, reach
    
    # Local search: add back excluded elements
    excluded = [x for x in range(1, 101) if x not in best]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in excluded[:]:
            new_reach = {s + x for s in best_reach if s + x <= 200}
            if 200 not in new_reach:
                best.append(x)
                best_reach |= new_reach
                excluded.remove(x)
                improved = True
    
    return best
# EVOLVE-BLOCK-END
