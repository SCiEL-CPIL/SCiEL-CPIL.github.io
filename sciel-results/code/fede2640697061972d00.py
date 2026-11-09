import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 50.0
    best_set = None
    best_size = -1
    
    # Fixed order baseline
    def greedy(order):
        selected = []
        reachable = 1  # bit 0 is set
        mask = (1 << 201) - 1
        for x in order:
            # Check if adding x creates sum 200
            new_reachable = reachable | (reachable << x)
            new_reachable &= mask
            if (new_reachable >> 200) & 1:
                continue
            selected.append(x)
            reachable = new_reachable
        return selected
    
    # 1. Deterministic greedy (descending)
    baseline = greedy(range(100, 0, -1))
    best_set = baseline
    best_size = len(baseline)
    
    # 2. Randomized greedy
    while time.time() < deadline:
        order = list(range(1, 101))
        random.shuffle(order)
        candidate = greedy(order)
        if len(candidate) > best_size:
            best_set = candidate
            best_size = len(candidate)
            
    return best_set
# EVOLVE-BLOCK-END
