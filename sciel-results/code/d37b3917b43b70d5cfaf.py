import time
import random
import numpy as np

# Precomputation
def precompute_aps(n):
    aps = []
    for d in range(1, n//3 + 1):
        for a in range(1, n - 3*d + 1):
            b = a + d
            c = b + d
            e = c + d
            if e <= n:
                aps.append((a, b, c, e))
    return aps

APS = precompute_aps(512)
APS_CONTAINING = [[] for _ in range(513)]
for i, ap in enumerate(APS):
    for x in ap:
        APS_CONTAINING[x].append(i)

def get_base3_set(n):
    s = set()
    for x in range(1, n+1):
        # check base 3 digits
        y = x
        valid = True
        while y > 0:
            if y % 3 == 2:
                valid = False
                break
            y //= 3
        if valid:
            s.add(x)
    return s

def greedy_fill(initial_set, n=512, deadline=None):
    S = set(initial_set)
    ap_counts = [0] * len(APS)
    
    # Initialize counts
    for x in S:
        for ap_idx in APS_CONTAINING[x]:
            ap_counts[ap_idx] += 1
            
    candidates = [x for x in range(1, n+1) if x not in S]
    random.shuffle(candidates)
    
    for x in candidates:
        if deadline is not None and time.time() > deadline:
            break
        valid = True
        for ap_idx in APS_CONTAINING[x]:
            if ap_counts[ap_idx] >= 3:
                valid = False
                break
        if valid:
            S.add(x)
            for ap_idx in APS_CONTAINING[x]:
                ap_counts[ap_idx] += 1
                
    return S

def local_search(S, n=512, deadline=None):
    """Try to improve S by swapping elements"""
    S = set(S)
    ap_counts = [0] * len(APS)
    
    # Initialize counts
    for x in S:
        for ap_idx in APS_CONTAINING[x]:
            ap_counts[ap_idx] += 1
            
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        # Try to find a swap: remove one element, add another
        elements = list(S)
        random.shuffle(elements)
        
        for x in elements:
            if deadline is not None and time.time() > deadline:
                return S
            
            # Try removing x
            for ap_idx in APS_CONTAINING[x]:
                ap_counts[ap_idx] -= 1
            
            # Try adding some candidate
            candidates = [y for y in range(1, n+1) if y not in S and y != x]
            random.shuffle(candidates)
            
            added = False
            for y in candidates:
                valid = True
                for ap_idx in APS_CONTAINING[y]:
                    if ap_counts[ap_idx] >= 3:
                        valid = False
                        break
                if valid:
                    S.remove(x)
                    S.add(y)
                    for ap_idx in APS_CONTAINING[y]:
                        ap_counts[ap_idx] += 1
                    added = True
                    improved = True
                    break
            
            if not added:
                # Restore x
                S.add(x)
                for ap_idx in APS_CONTAINING[x]:
                    ap_counts[ap_idx] += 1
            else:
                break  # Restart the outer loop with new S
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    best_S = get_base3_set(512)
    best_size = len(best_S)
    
    # Run multiple rounds of greedy fill with different random seeds
    for round_num in range(20):
        if time.time() > deadline:
            break
        
        # Try different initial sets
        if round_num == 0:
            current_S = set(best_S)
        else:
            # Perturb best_S
            current_S = set(best_S)
            removal_rate = random.choice([0.05, 0.1, 0.15, 0.2])
            num_remove = int(len(current_S) * removal_rate)
            if num_remove > 0:
                to_remove = random.sample(list(current_S), num_remove)
                current_S -= set(to_remove)
        
        new_S = greedy_fill(current_S, deadline=deadline)
        
        if len(new_S) > best_size:
            best_size = len(new_S)
            best_S = new_S
    
    # Local search on the best solution
    if time.time() < deadline:
        improved_S = local_search(best_S, deadline=deadline)
        if len(improved_S) > best_size:
            best_size = len(improved_S)
            best_S = improved_S
    
    return sorted(best_S)
