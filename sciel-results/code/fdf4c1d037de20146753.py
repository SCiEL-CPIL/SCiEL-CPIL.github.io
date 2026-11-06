import time
import random

def _can_add(x, S, used_sums, pair_sums):
    """
    Returns (valid, new_triple_sums, new_pair_sums)
    """
    n = len(S)
    new_triple_sums = []
    new_pair_sums = []
    
    # 1. 3*x
    s = 3 * x
    if used_sums[s]:
        return False, [], []
    new_triple_sums.append(s)
    
    # 2. 2*x + s_i
    for s_i in S:
        s = 2 * x + s_i
        if used_sums[s]:
            return False, [], []
        new_triple_sums.append(s)
        
    # 3. x + 2*s_i
    for s_i in S:
        s = x + 2 * s_i
        if used_sums[s]:
            return False, [], []
        new_triple_sums.append(s)
        
    # 4. x + s_i + s_j (using precomputed pair_sums)
    for p in pair_sums:
        s = x + p
        if used_sums[s]:
            return False, [], []
        new_triple_sums.append(s)
        
    # Compute new pair sums for future checks
    # Pairs involving x:
    # 2*x
    new_pair_sums.append(2 * x)
    # x + s_i
    for s_i in S:
        new_pair_sums.append(x + s_i)
        
    return True, new_triple_sums, new_pair_sums

def _greedy_random(seed, max_val=4096, time_limit=5):
    rng = random.Random(seed)
    candidates = list(range(1, max_val + 1))
    rng.shuffle(candidates)
    
    S = []
    used_sums = bytearray(3 * max_val + 1)
    pair_sums = []
    
    start_time = time.time()
    
    for x in candidates:
        if time.time() - start_time > time_limit:
            break
            
        valid, new_triple_sums, new_pair_sums = _can_add(x, S, used_sums, pair_sums)
        if valid:
            S.append(x)
            for s in new_triple_sums:
                used_sums[s] = 1
            pair_sums.extend(new_pair_sums)
                
    S.sort()
    return S

def _greedy_linear(max_val=4096):
    S = []
    used_sums = bytearray(3 * max_val + 1)
    pair_sums = []
    
    for x in range(1, max_val + 1):
        valid, new_triple_sums, new_pair_sums = _can_add(x, S, used_sums, pair_sums)
        if valid:
            S.append(x)
            for s in new_triple_sums:
                used_sums[s] = 1
            pair_sums.extend(new_pair_sums)
            
    S.sort()
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = []
    best_len = 0
    
    # 1. Baseline
    linear_set = _greedy_linear(4096)
    if len(linear_set) > best_len:
        best_set = linear_set
        best_len = len(linear_set)
        
    # 2. Randomized Restarts
    seed = 42
    while time.time() < deadline - 5:
        current_set = _greedy_random(seed, 4096, time_limit=5)
        if len(current_set) > best_len:
            best_set = current_set
            best_len = len(current_set)
        seed += 1
        
    return best_set
