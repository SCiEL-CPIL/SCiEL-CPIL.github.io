import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    N = 512
    # Precompute all 4-APs
    # AP: a, a+d, a+2d, a+3d
    # a >= 1, a+3d <= N
    # d >= 1
    aps = []
    element_to_aps = [[] for _ in range(N + 1)]
    
    for d in range(1, (N - 3) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            idx = len(aps)
            aps.append(ap)
            for x in ap:
                element_to_aps[x].append(idx)
                
    total_aps = len(aps)
    
    best_set = set()
    best_size = 0
    
    # Run multiple random greedy constructions
    # Each construction is fast
    
    for seed_counter in range(10000):
        if time.time() > deadline:
            break
            
        random.seed(seed_counter)
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S = set()
        
        for x in order:
            # Check if x can be added to S
            can_add = True
            for idx in element_to_aps[x]:
                ap = aps[idx]
                count_in_S = 0
                for y in ap:
                    if y != x and y in S:
                        count_in_S += 1
                        if count_in_S == 3:
                            break
                if count_in_S == 3:
                    can_add = False
                    break
            
            if can_add:
                S.add(x)
                
        # Fill pass: try to add any missing element
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if x not in S:
                    can_add = True
                    for idx in element_to_aps[x]:
                        ap = aps[idx]
                        count_in_S = 0
                        for y in ap:
                            if y != x and y in S:
                                count_in_S += 1
                                if count_in_S == 3:
                                    break
                        if count_in_S == 3:
                            can_add = False
                            break
                    if can_add:
                        S.add(x)
                        changed = True
                        if time.time() > deadline:
                            break
            if time.time() > deadline:
                break
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    return sorted(best_set)
