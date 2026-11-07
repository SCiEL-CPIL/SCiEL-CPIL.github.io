import time
import random

def _valid(Sset):
    # Sset: set of ints. Check no distinct a<b<c in Sset with a+b,a+c,b+c all in Sset.
    elems = sorted(Sset)
    n = len(elems)
    for i in range(n):
        a = elems[i]
        for j in range(i+1, n):
            b = elems[j]
            ab = a + b
            if ab not in Sset:
                continue
            for k in range(j+1, n):
                c = elems[k]
                if (a + c) in Sset and (b + c) in Sset:
                    return False
    return True

def construct():
    deadline = time.time() + 55.0
    best_size = 0
    best_set = set()
    
    # Initial strong seed: {64..128}
    S_seed = set(range(64, 129))
    if _valid(S_seed):
        best_size = len(S_seed)
        best_set = set(S_seed)
    
    # Randomized Greedy Restarts
    # We try to build a set by adding elements in random order if valid.
    # We can also try to start from the best_set and try to add remaining elements in random order.
    
    while time.time() < deadline:
        # Strategy 1: Start empty, random order
        S = set()
        order = random.sample(range(1, 129), 128)
        for x in order:
            trial = S | {x}
            if _valid(trial):
                S = trial
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
        # Strategy 2: Start from best_set, try to add remaining in random order
        # This is faster as it starts larger.
        S = set(best_set)
        remaining = [x for x in range(1, 129) if x not in S]
        random.shuffle(remaining)
        for x in remaining:
            trial = S | {x}
            if _valid(trial):
                S = trial
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    result = sorted(best_set)
    result = [int(x) for x in result]
    return result
