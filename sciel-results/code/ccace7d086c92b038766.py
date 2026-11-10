"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin before 60s limit
    
    def greedy(order):
        selected = []
        reachable = {0}
        for x in order:
            if time.time() > deadline:
                break
            new = {s + x for s in reachable if s + x <= 200}
            if 200 not in new:
                selected.append(x)
                reachable |= new
        return selected
    
    # Try descending (original) and ascending, pick best
    desc = greedy(range(100, 0, -1))
    asc = greedy(range(1, 101))
    
    best = desc if len(desc) >= len(asc) else asc
    
    # If time remains, try a few more orderings
    if time.time() < deadline - 10:
        import random
        rng = random.Random(42)
        for seed in range(100):
            if time.time() > deadline:
                break
            order = list(range(1, 101))
            rng = random.Random(seed)
            rng.shuffle(order)
            cand = greedy(order)
            if len(cand) > len(best):
                best = cand
    
    return best if best else [100]
# EVOLVE-BLOCK-END
