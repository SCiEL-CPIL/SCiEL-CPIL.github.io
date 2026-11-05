# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    n = 4096
    
    def triple_sums_with_new(x, selected):
        """Compute all triple sums involving x with elements of selected.
        selected must be sorted. Returns (sums_set, ok) where ok is False if collision."""
        k = len(selected)
        sums = set()
        # (x, x, x)
        s = 3 * x
        sums.add(s)
        # (x, x, y) and (x, y, y) for each y in selected
        for y in selected:
            s = 2 * x + y
            if s in sums:
                return sums, False
            sums.add(s)
            s = x + 2 * y
            if s in sums:
                return sums, False
            sums.add(s)
        # (x, y, z) for y <= z in selected
        for i in range(k):
            y = selected[i]
            for j in range(i, k):
                z = selected[j]
                s = x + y + z
                if s in sums:
                    return sums, False
                sums.add(s)
        return sums, True

    def all_triple_sums(selected):
        """Compute all triple sums for a sorted list."""
        k = len(selected)
        sums = set()
        for i in range(k):
            a = selected[i]
            for j in range(i, k):
                b = selected[j]
                for l in range(j, k):
                    c = selected[l]
                    sums.add(a + b + c)
        return sums

    def greedy_build(order, time_limit=None):
        """Greedy build from a given order. Returns (selected, sums)."""
        selected = []
        sums = set()
        for x in order:
            if time_limit is not None and time.time() > time_limit:
                break
            new_sums, ok = triple_sums_with_new(x, selected)
            if ok:
                selected.append(x)
                sums.update(new_sums)
        return selected, sums

    def local_search_improve(selected, sums, target_size, max_iterations=5000):
        """Try to increase size by removing 1 and adding 2 elements."""
        current = list(selected)
        current_sums = set(sums)
        current_size = len(current)
        rng = np.random.default_rng(12345)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            if current_size >= target_size:
                break
            
            # Try random removal
            idx = rng.integers(0, len(current))
            y_remove = current[idx]
            
            # Build reduced set
            reduced = [x for x in current if x != y_remove]
            reduced_sums = all_triple_sums(reduced)
            
            # Try to add 2 new elements
            # Sample candidates from [1, n] excluding current
            for attempt in range(30):
                x1 = rng.integers(1, n + 1)
                if x1 in current:
                    continue
                new_sums1, ok1 = triple_sums_with_new(x1, reduced)
                if not ok1:
                    continue
                
                temp = reduced + [x1]
                temp_sums = reduced_sums | new_sums1
                
                for attempt2 in range(30):
                    x2 = rng.integers(1, n + 1)
                    if x2 in current or x2 == x1:
                        continue
                    new_sums2, ok2 = triple_sums_with_new(x2, temp)
                    if ok2:
                        # Success
                        new_set = temp + [x2]
                        new_sums = temp_sums | new_sums2
                        current = new_set
                        current_sums = new_sums
                        current_size += 1
                        break
                if current_size >= target_size:
                    break
            if current_size >= target_size:
                break
        
        return current, current_sums, current_size

    best_set = []
    best_sums = set()
    best_size = 0
    
    # Phase 1: Deterministic greedy from 1
    order = list(range(1, n + 1))
    s, sums = greedy_build(order, deadline - 30)
    best_set = s
    best_sums = sums
    best_size = len(s)
    
    # Phase 2: Randomized greedy restarts
    rng = np.random.default_rng(42)
    max_restarts = 300
    for i in range(max_restarts):
        if time.time() > deadline - 25:
            break
        order = rng.integers(1, n + 1, size=n)
        # Shuffle to avoid bias
        np.random.shuffle(order)
        s, sums = greedy_build(order, deadline - 25)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            best_sums = sums
    
    # Phase 3: Local search to push toward 40+
    if best_size < 45:
        current, current_sums, current_size = local_search_improve(
            best_set, best_sums, 45, max_iterations=10000
        )
        if current_size > best_size:
            best_size = current_size
            best_set = current
            best_sums = current_sums
    
    # Phase 4: More aggressive local search
    if best_size < 50:
        current, current_sums, current_size = local_search_improve(
            best_set, best_sums, 50, max_iterations=20000
        )
        if current_size > best_size:
            best_size = current_size
            best_set = current
            best_sums = current_sums
    
    return best_set

# EVOLVE-BLOCK-END
