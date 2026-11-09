import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    def greedy(order):
        reachable = 1
        mask = (1 << 201) - 1
        selected = []
        for x in order:
            target = 200 - x
            if 0 <= target <= 200 and (reachable >> target) & 1:
                continue
            selected.append(x)
            reachable = (reachable | (reachable << x)) & mask
        return selected
    
    best = []
    
    # Try descending
    if time.time() < deadline:
        result = greedy(range(100, 0, -1))
        if len(result) > len(best):
            best = result
    
    # Try ascending
    if time.time() < deadline:
        result = greedy(range(1, 101))
        if len(result) > len(best):
            best = result
    
    # Try random permutations
    random.seed(42)
    attempts = 0
    while time.time() < deadline and attempts < 200:
        order = list(range(1, 101))
        random.shuffle(order)
        result = greedy(order)
        if len(result) > len(best):
            best = result
        attempts += 1
    
    # Try min-new-sums greedy
    if time.time() < deadline:
        reachable = 1
        mask = (1 << 201) - 1
        remaining = set(range(1, 101))
        selected = []
        while remaining:
            if time.time() > deadline:
                break
            best_c = None
            best_count = float('inf')
            for x in remaining:
                target = 200 - x
                if 0 <= target <= 200 and (reachable >> target) & 1:
                    continue
                new_r = (reachable | (reachable << x)) & mask
                cnt = bin(new_r & ~reachable).count('1')
                if cnt < best_count:
                    best_count = cnt
                    best_c = x
            if best_c is None:
                break
            selected.append(best_c)
            remaining.remove(best_c)
            reachable = (reachable | (reachable << best_c)) & mask
        if len(selected) > len(best):
            best = selected
    
    return best
