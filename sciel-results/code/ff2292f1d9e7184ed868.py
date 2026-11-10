import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_set = []
    best_size = 0
    
    # Deterministic greedy
    chosen = []
    counts = {}
    for x in range(1, 1001):
        ok = True
        for y in chosen:
            if y <= x and counts.get(x + y, 0) >= 2:
                ok = False
                break
        if ok and counts.get(2 * x, 0) >= 2:
            ok = False
        if ok:
            chosen.append(x)
            for y in chosen[:-1]:
                if y <= x:
                    counts[x + y] = counts.get(x + y, 0) + 1
            counts[2 * x] = counts.get(2 * x, 0) + 1
    
    best_size = len(chosen)
    best_set = chosen[:]
    
    # Randomized greedy
    for _ in range(3000):
        if time.time() > deadline:
            break
        cand = list(range(1, 1001))
        random.shuffle(cand)
        chosen = []
        counts = {}
        for x in cand:
            ok = True
            for y in chosen:
                if y <= x and counts.get(x + y, 0) >= 2:
                    ok = False
                    break
            if ok and counts.get(2 * x, 0) >= 2:
                ok = False
            if ok:
                chosen.append(x)
                for y in chosen[:-1]:
                    if y <= x:
                        counts[x + y] = counts.get(x + y, 0) + 1
                counts[2 * x] = counts.get(2 * x, 0) + 1
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen[:]
    
    return best_set
