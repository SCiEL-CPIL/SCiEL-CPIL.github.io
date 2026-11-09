import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def greedy(order):
        chosen = []
        counts = {}
        for x in order:
            new_sums = [x + y for y in chosen] + [2 * x]
            # Check if any sum would exceed 2
            valid = True
            for s in new_sums:
                if counts.get(s, 0) + 1 > 2:
                    valid = False
                    break
            if valid:
                chosen.append(x)
                for s in new_sums:
                    counts[s] = counts.get(s, 0) + 1
        return chosen

    # 1. Deterministic greedy (baseline)
    best_set = greedy(range(1, 1001))
    best_size = len(best_set)
    
    # 2. Randomized greedy runs
    # We will run as many as possible within the time budget
    # Each run is O(N^2) roughly, but with early breaks.
    # N=1000, so 10^6 ops per run. Python can do ~10^7 ops/sec? 
    # Let's estimate: 1000 items. Inner loop avg 500 items. 500k checks.
    # 500k checks * 100 runs = 50M ops. Might be tight in 110s.
    # Let's optimize the check.
    
    # Optimized Greedy
    def greedy_fast(order):
        chosen = []
        counts = {}
        for x in order:
            # Pre-check sums
            # x + y for y in chosen
            # 2*x
            # We need to check if counts[s] + 1 <= 2 for all new s
            # This is equivalent to counts[s] < 2 for all new s
            
            # Optimization: 
            # If we maintain counts, we just check the new sums.
            ok = True
            for y in chosen:
                s = x + y
                if counts.get(s, 0) >= 2:
                    ok = False
                    break
            if ok:
                s2 = 2 * x
                if counts.get(s2, 0) >= 2:
                    ok = False
            
            if ok:
                chosen.append(x)
                for y in chosen[:-1]: # y < x in value? No, chosen is just list.
                    # Wait, the loop above iterated chosen.
                    # We need to update counts for all pairs involving x.
                    pass
                
                # Re-do update correctly
                for y in chosen[:-1]:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                s2 = 2 * x
                counts[s2] = counts.get(s2, 0) + 1
        
        return chosen

    # Re-run baseline with fast version to be sure
    best_set = greedy_fast(range(1, 1001))
    best_size = len(best_set)
    
    # Randomized runs
    # Shuffle 1..1000
    # We can try different random seeds or just shuffle.
    
    while time.time() < deadline:
        order = list(range(1, 1001))
        random.shuffle(order)
        current_set = greedy_fast(order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
    
    return best_set
