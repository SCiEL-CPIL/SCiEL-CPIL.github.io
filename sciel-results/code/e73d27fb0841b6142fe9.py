import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 512
    
    def check_adds_4ap(s, x):
        for d in range(1, n + 1):
            if x + d in s and x + 2*d in s and x + 3*d in s:
                return True
            if x - d in s and x + d in s and x + 2*d in s:
                return True
            if x - 2*d in s and x - d in s and x + d in s:
                return True
            if x - 3*d in s and x - 2*d in s and x - d in s:
                return True
        return False
    
    def greedy_with_order(order):
        s = set()
        for x in order:
            if not check_adds_4ap(s, x):
                s.add(x)
        return s
    
    # Multiple greedy strategies
    best_set = greedy_with_order(range(1, n + 1))
    best_size = len(best_set)
    
    # Backward
    if time.time() < deadline - 5:
        s = greedy_with_order(range(n, 0, -1))
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
    
    # Random restarts
    seed = 12345
    while time.time() < deadline - 10:
        rng = random.Random(seed)
        order = list(range(1, n + 1))
        rng.shuffle(order)
        s = greedy_with_order(order)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
        seed += 1
    
    # Simple local search: try to add skipped elements
    if time.time() < deadline - 5:
        s = set(best_set)
        skipped = [x for x in range(1, n + 1) if x not in s]
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for x in skipped:
                if x not in s and not check_adds_4ap(s, x):
                    s.add(x)
                    improved = True
            # Update skipped
            skipped = [x for x in range(1, n + 1) if x not in s]
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
    
    return sorted(best_set)
