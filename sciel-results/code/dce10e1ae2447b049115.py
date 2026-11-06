import time
import random
import numpy as np

def _count_diffs_bitset(S):
    """Count distinct d>0 with an AP in S, using bitset operations."""
    if len(S) < 3:
        return 0
    S_list = sorted(S)
    min_val = S_list[0]
    max_val = S_list[-1]
    # Bitset over range [1, 512]
    bits = np.zeros(513, dtype=np.uint8)
    for x in S_list:
        bits[x] = 1
    diffs_found = 0
    for d in range(1, (max_val - min_val) // 2 + 1):
        # Check if any a such that a, a+d, a+2d all in S
        # a ranges from max(1, min_val) to max_val - 2d
        a_min = max(1, min_val)
        a_max = max_val - 2 * d
        if a_min > a_max:
            continue
        # bits[a], bits[a+d], bits[a+2d]
        seg1 = bits[a_min:a_max + 1]
        seg2 = bits[a_min + d:a_max + 1 + d]
        seg3 = bits[a_min + 2 * d:a_max + 1 + 2 * d]
        if np.any(seg1 & seg2 & seg3):
            diffs_found += 1
    return diffs_found

def _get_new_diffs_fast(S_set, x):
    """Return set of new differences realized by adding x to S_set."""
    new_diffs = set()
    # x as start: x, x+d, x+2d
    max_d_start = (512 - x) // 2
    for d in range(1, max_d_start + 1):
        if (x + d) in S_set and (x + 2 * d) in S_set:
            new_diffs.add(d)
    # x as middle: x-d, x, x+d
    max_d_mid = min(x - 1, 512 - x)
    for d in range(1, max_d_mid + 1):
        if (x - d) in S_set and (x + d) in S_set:
            new_diffs.add(d)
    # x as end: x-2d, x-d, x
    max_d_end = (x - 1) // 2
    for d in range(1, max_d_end + 1):
        if (x - 2 * d) in S_set and (x - d) in S_set:
            new_diffs.add(d)
    return new_diffs

def _cluster_score(S_set, x, current_diffs):
    """
    Score a candidate x by:
    1. Number of truly new diffs
    2. Bonus for diffs that complete multiple APs at once (cluster density)
    """
    new_diffs = _get_new_diffs_fast(S_set, x)
    truly_new = new_diffs - current_diffs
    base = len(truly_new)
    
    # Cluster bonus: for each new diff, count how many APs it completes
    # (i.e., how many starting points a work)
    cluster_bonus = 0
    for d in truly_new:
        # Count APs with this diff that include x
        cnt = 0
        # x as start
        if x + d <= 512 and x + 2 * d <= 512:
            if (x + d) in S_set and (x + 2 * d) in S_set:
                cnt += 1
        # x as middle
        if x - d >= 1 and x + d <= 512:
            if (x - d) in S_set and (x + d) in S_set:
                cnt += 1
        # x as end
        if x - 2 * d >= 1 and x - d >= 1:
            if (x - 2 * d) in S_set and (x - d) in S_set:
                cnt += 1
        # Bonus: more APs for same diff = denser cluster
        if cnt > 1:
            cluster_bonus += cnt - 1
    
    return base, cluster_bonus

def _greedy_construct_cluster(N, MAX_VAL, rng, deadline):
    """Greedy construction with cluster-aware scoring."""
    S_set = set()
    current_diffs = set()
    
    for step in range(N):
        if time.time() > deadline:
            break
        
        best_x = -1
        best_base = -1
        best_cluster = -1
        best_new_diffs = None
        
        candidates = [x for x in range(1, MAX_VAL + 1) if x not in S_set]
        
        for x in candidates:
            base, cluster = _cluster_score(S_set, x, current_diffs)
            
            if base > best_base or (base == best_base and cluster > best_cluster):
                best_base = base
                best_cluster = cluster
                best_x = x
                best_new_diffs = _get_new_diffs_fast(S_set, x) - current_diffs
            elif base == best_base and cluster == best_cluster and base > 0:
                # Random tie-breaking
                if rng.random() < 0.3:
                    best_x = x
                    best_new_diffs = _get_new_diffs_fast(S_set, x) - current_diffs
        
        if best_x == -1:
            best_x = rng.choice(candidates)
            best_new_diffs = set()
        
        S_set.add(best_x)
        if best_new_diffs:
            current_diffs.update(best_new_diffs)
    
    return S_set

def _local_search(S_list, N, MAX_VAL, rng, deadline, initial_score):
    """Local search with swap moves."""
    S_set_local = set(S_list)
    current_score = initial_score
    best_score = current_score
    best_set = list(S_set_local)
    
    # Precompute for faster scoring
    while time.time() < deadline:
        i = rng.randint(0, N - 1)
        x = rng.randint(1, MAX_VAL)
        while x in S_set_local:
            x = rng.randint(1, MAX_VAL)
        
        old_x = S_list[i]
        
        S_set_local.remove(old_x)
        S_set_local.add(x)
        
        new_score = _count_diffs_bitset(S_set_local)
        
        if new_score > current_score:
            current_score = new_score
            S_list[i] = x
            if current_score > best_score:
                best_score = current_score
                best_set = list(S_set_local)
        else:
            S_set_local.remove(x)
            S_set_local.add(old_x)
    
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    N = 32
    MAX_VAL = 512
    
    best_score = 0
    best_set = None
    
    num_restarts = 15
    
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        rng = random.Random(r * 12345 + 67890)
        
        # Phase 1: Cluster-aware greedy construction
        ls_deadline = deadline - 10
        S_set = _greedy_construct_cluster(N, MAX_VAL, rng, deadline - 20)
        
        if len(S_set) < N:
            # Fill remaining randomly
            remaining = [x for x in range(1, MAX_VAL + 1) if x not in S_set]
            while len(S_set) < N and remaining:
                S_set.add(rng.choice(remaining))
                remaining.remove(list(S_set)[-1])
        
        S_list = list(S_set)
        score = _count_diffs_bitset(S_set)
        
        if score > best_score:
            best_score = score
            best_set = list(S_set)
        
        # Phase 2: Local search
        if time.time() < deadline - 5:
            ls_budget = min(3.0, deadline - 5 - time.time())
            S_list, ls_score = _local_search(S_list, N, MAX_VAL, rng, time.time() + ls_budget, score)
            
            if ls_score > best_score:
                best_score = ls_score
                best_set = S_list
    
    if best_set is None:
        return list(range(1, N + 1))
    
    return best_set

# EVOLVE-BLOCK-END
