import random
import time

def score(S):
    S_set = set(S)
    diffs = set()
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i + 1, n):
            b = S[j]
            d = b - a
            c = b + d
            if c in S_set:
                diffs.add(d)
    return len(diffs)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Initialize with a few random sets and pick the best
    best_set = None
    best_score = -1
    
    for _ in range(10):
        s = sorted(random.sample(range(1, 513), 32))
        sc = score(s)
        if sc > best_score:
            best_score = sc
            best_set = s
            
    # Also consider seed
    seed_set = sorted(range(1, 33))
    seed_score = score(seed_set)
    if seed_score > best_score:
        best_score = seed_score
        best_set = seed_set
        
    no_improve = 0
    while time.time() < deadline:
        idx = random.randint(0, 31)
        u = best_set[idx]
        
        # Find a replacement
        # To avoid infinite loop if v==u (impossible since u in set, v not in set)
        while True:
            v = random.randint(1, 512)
            if v not in best_set:
                break
                
        best_set[idx] = v
        best_set.sort()
        
        new_score = score(best_set)
        
        if new_score > best_score:
            best_score = new_score
            no_improve = 0
        else:
            best_set[idx] = u
            best_set.sort()
            no_improve += 1
            # Simple restart logic: if stuck for a long time, re-randomize a few elements?
            # For now, just keep trying.
            
    return best_set
