"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def _greedy_subset(elements, target=200, deadline=None):
    """Greedily select elements in given order, including if target not newly reachable."""
    selected = []
    reachable = 1  # bit 0 set, sum 0 reachable
    mask = (1 << (target + 1)) - 1
    for x in elements:
        new_reachable = reachable | ((reachable << x) & mask)
        if (new_reachable >> target) & 1 == 0:
            selected.append(x)
            reachable = new_reachable
        if deadline is not None and time.time() > deadline:
            break
    return selected

def construct():
    deadline = time.time() + 50  # safety margin before 60s limit
    all_elements = list(range(1, 101))
    
    best = []
    
    # Try descending order
    cand = _greedy_subset(sorted(all_elements, reverse=True), deadline=deadline)
    if len(cand) > len(best):
        best = cand
    
    # Try ascending order
    cand = _greedy_subset(sorted(all_elements), deadline=deadline)
    if len(cand) > len(best):
        best = cand
    
    # Try several random orderings
    rng = random.Random(42)
    for _ in range(50):
        if time.time() > deadline:
            break
        shuffled = all_elements[:]
        rng.shuffle(shuffled)
        cand = _greedy_subset(shuffled, deadline=deadline)
        if len(cand) > len(best):
            best = cand
    
    # Ensure nonempty
    if not best:
        best = [100]
    
    return best
# EVOLVE-BLOCK-END
