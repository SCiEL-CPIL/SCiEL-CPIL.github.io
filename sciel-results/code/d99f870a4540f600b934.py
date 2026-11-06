import random
import time

def construct():
    start_time = time.time()
    limit = 118  # seconds
    
    def run_greedy(order, max_val=1000):
        A = []
        counts = [0] * (2 * max_val + 1)
        for x in order:
            ok = True
            if counts[2 * x] >= 2:
                ok = False
            if ok:
                for y in A:
                    if counts[x + y] >= 2:
                        ok = False
                        break
            if ok:
                A.append(x)
                counts[2 * x] += 1
                for i in range(len(A) - 1):
                    counts[x + A[i]] += 1
        return A

    best_set = []
    best_size = 0
    
    # 1. Deterministic Sorted
    A = run_greedy(list(range(1, 1001)))
    best_set = A
    best_size = len(A)
    
    # 2. Random Restarts
    while time.time() - start_time < limit:
        # Generate order
        r = random.random()
        if r < 0.6:
            order = list(range(1, 1001))
            random.shuffle(order)
        elif r < 0.8:
            # Weighted random: prefer larger numbers?
            order = list(range(1, 1001))
            random.shuffle(order)
        else:
            # Hybrid: Start with some large numbers, then random
            large = list(range(900, 1001))
            random.shuffle(large)
            rest = list(range(1, 900))
            random.shuffle(rest)
            order = large + rest
            
        A = run_greedy(order)
        if len(A) > best_size:
            best_size = len(A)
            best_set = A
            
    return best_set
