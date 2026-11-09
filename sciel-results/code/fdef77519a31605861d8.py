import random
import time

def score(S):
    """
    Calculate the number of distinct d > 0 such that a, a+d, a+2d in S.
    """
    S_list = list(S)
    S_set = set(S_list)
    diffs = set()
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            c = S_list[j]
            s = a + c
            if s % 2 == 0:
                b = s // 2
                if b in S_set:
                    d = c - b
                    diffs.add(d)
    return len(diffs)

def greedy_construct():
    """
    Construct a set of 32 integers greedily to maximize score.
    """
    S = set()
    candidates = set(range(1, 513))
    
    for _ in range(32):
        best_score = -1
        best_elem = None
        # To speed up, we can sample if candidates are many, but 512 is small enough
        # However, as S grows, score calculation gets more expensive.
        for y in candidates:
            new_S = S | {y}
            sc = score(new_S)
            if sc > best_score:
                best_score = sc
                best_elem = y
        S.add(best_elem)
        candidates.remove(best_elem)
        
    return S

def local_search(S, time_limit=10):
    """
    Refine the set using local search (swap moves).
    """
    S = set(S)
    current_score = score(S)
    start_time = time.time()
    
    # Pre-compute list of all possible values for faster random selection
    all_vals = list(range(1, 513))
    
    while time.time() - start_time < time_limit:
        # Pick random x in S
        x = random.choice(list(S))
        # Pick random y not in S
        y = random.choice(all_vals)
        while y in S:
            y = random.choice(all_vals)
        
        # Apply move
        S.remove(x)
        S.add(y)
        new_score = score(S)
        
        if new_score >= current_score:
            current_score = new_score
        else:
            # Revert
            S.remove(y)
            S.add(x)
            
    return S, current_score

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin
    
    best_S = None
    best_score = -1
    
    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline:
        # Greedy construction
        S = greedy_construct()
        sc = score(S)
        
        # Local search refinement
        # Allocate remaining time for local search, but cap it per restart
        time_left = deadline - time.time()
        if time_left < 5:
            break
            
        ls_time = min(10, time_left - 2) # Leave 2s buffer
        if ls_time > 0:
            S, sc = local_search(S, ls_time)
            
        if sc > best_score:
            best_score = sc
            best_S = S
            num_restarts += 1
            
        # If we have very little time, stop
        if time.time() > deadline - 5:
            break
            
    if best_S is None:
        # Fallback: just return a random set or the initial range
        return list(range(1, 33))
        
    return list(best_S)
