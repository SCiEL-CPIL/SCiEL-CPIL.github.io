import time
import random

def construct():
    N = 1000
    start_time = time.time()
    deadline = start_time + 140
    
    best_size = 0
    best_set = []
    
    def greedy_build(order):
        counts = [0] * (2 * N + 1)
        S = []
        S_set = set()
        for x in order:
            valid = True
            for y in S:
                s = x + y
                if counts[s] >= 2:
                    valid = False
                    break
            if valid:
                s = 2 * x
                if counts[s] >= 2:
                    valid = False
            
            if valid:
                for y in S:
                    counts[x + y] += 1
                counts[2 * x] += 1
                S.append(x)
                S_set.add(x)
        return S, counts, S_set
    
    def try_order(order):
        nonlocal best_size, best_set
        S, counts, S_set = greedy_build(order)
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # 1. Try to add any element
            added = False
            for cand in range(1, N + 1):
                if cand in S_set:
                    continue
                valid = True
                for y in S:
                    s = cand + y
                    if counts[s] >= 2:
                        valid = False
                        break
                if valid:
                    s = 2 * cand
                    if counts[s] >= 2:
                        valid = False
                if valid:
                    for y in S:
                        counts[cand + y] += 1
                    counts[2 * cand] += 1
                    S.append(cand)
                    S_set.add(cand)
                    added = True
                    break
            
            if added:
                improved = True
                continue
            
            # 2. No single add possible. Try remove-and-refill
            best_rem = -1
            best_adds = 0
            best_new_S = None
            best_new_counts = None
            best_new_S_set = None
            
            for i, x in enumerate(S):
                # Simulate removing x
                new_counts = list(counts)
                new_S = S[:i] + S[i+1:]
                new_S_set = S_set.copy()
                new_S_set.discard(x)
                
                # Remove x's contributions
                for y in new_S:
                    s = x + y
                    if new_counts[s] > 0:
                        new_counts[s] -= 1
                s2 = 2 * x
                if new_counts[s2] > 0:
                    new_counts[s2] -= 1
                
                # Try to add as many as possible
                added_count = 0
                for cand in range(1, N + 1):
                    if cand in new_S_set:
                        continue
                    valid = True
                    for y in new_S:
                        s = cand + y
                        if new_counts[s] >= 2:
                            valid = False
                            break
                    if valid:
                        s2 = 2 * cand
                        if new_counts[s2] >= 2:
                            valid = False
                    if valid:
                        for y in new_S:
                            new_counts[cand + y] += 1
                        new_counts[2 * cand] += 1
                        new_S.append(cand)
                        new_S_set.add(cand)
                        added_count += 1
                
                if added_count > best_adds:
                    best_adds = added_count
                    best_rem = i
                    best_new_S = new_S
                    best_new_counts = new_counts
                    best_new_S_set = new_S_set
            
            if best_adds > 0:
                S = best_new_S
                counts = best_new_counts
                S_set = best_new_S_set
                improved = True
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = list(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)

    # 1. Natural order
    try_order(list(range(1, N + 1)))
    
    # 2. Reverse order
    if time.time() < deadline - 30:
        try_order(list(range(N, 0, -1)))
        
    # 3. Random shuffles
    rng = random.Random(42)
    shuffle_count = 0
    while time.time() < deadline - 20:
        order = list(range(1, N + 1))
        rng.shuffle(order)
        try_order(order)
        shuffle_count += 1
        if shuffle_count >= 20:
            break
            
    return best_set
