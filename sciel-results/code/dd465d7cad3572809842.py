import random
import time

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin
    
    best_set = []
    best_size = 0
    
    # 1. Run deterministic greedy (descending)
    def run_greedy(order):
        mask = 1  # Bit 0 set
        selected = []
        for x in order:
            new_mask = mask | (mask << x)
            if not (new_mask & (1 << 200)):
                selected.append(x)
                mask = new_mask
        return selected

    # Initial deterministic runs
    candidates = [
        run_greedy(range(100, 0, -1)),
        run_greedy(range(1, 101)),
        run_greedy(range(1, 101, 2)),
        run_greedy(range(2, 101, 2))
    ]
    
    for c in candidates:
        if len(c) > best_size:
            best_size = len(c)
            best_set = c
            
    # 2. Randomized Greedy
    # Number of iterations depends on time
    # Estimate: 1 run ~ 0.0001s? 100 iterations * simple int ops.
    # Let's do as many as possible.
    
    while time.time() < deadline:
        random.shuffle_order = list(range(1, 101))
        random.shuffle(random.shuffle_order)
        c = run_greedy(random.shuffle_order)
        if len(c) > best_size:
            best_size = len(c)
            best_set = c
            
    return best_set
