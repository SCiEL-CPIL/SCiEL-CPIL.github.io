import time
import random

# EVOLVE-BLOCK-START

def construct():
    N = 512
    start_time = time.time()
    time_limit = 55.0  # Safety margin
    
    # 1. Precompute 4-APs
    # ap_list: list of tuples (a, b, c, d)
    # aps_containing: list of lists, aps_containing[i] = list of indices of APs containing i
    ap_list = []
    aps_containing = [[] for _ in range(N + 1)]
    
    # Generate all 4-APs
    # a, a+d, a+2d, a+3d <= N
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            d_val = a + 3 * d
            ap_idx = len(ap_list)
            ap_list.append((a, b, c, d_val))
            for v in (a, b, c, d_val):
                aps_containing[v].append(ap_idx)
    
    num_aps = len(ap_list)
    
    # 2. Greedy Construction with Random Restarts
    def is_valid_addition(ap_count, x):
        # Check if adding x would complete any 4-AP
        for ap_idx in aps_containing[x]:
            if ap_count[ap_idx] == 3:
                return False
        return True
        
    def greedy_construct(order):
        S = set()
        ap_count = [0] * num_aps
        for x in order:
            if is_valid_addition(ap_count, x):
                S.add(x)
                for ap_idx in aps_containing[x]:
                    ap_count[ap_idx] += 1
        return S, ap_count

    def local_search(S, ap_count):
        # S is a set, ap_count is a list
        best_S = S.copy()
        best_ap_count = ap_count.copy()
        best_size = len(S)
        
        # Pre-compute list of all elements for random selection
        all_elements = list(range(1, N + 1))
        
        while time.time() - start_time < time_limit:
            # Try to add a random element
            added = False
            for _ in range(100): # Try a few random additions
                if not S:
                    break
                x = random.choice(all_elements)
                if x in S:
                    continue
                if is_valid_addition(ap_count, x):
                    S.add(x)
                    for ap_idx in aps_containing[x]:
                        ap_count[ap_idx] += 1
                    added = True
                    if len(S) > best_size:
                        best_S = S.copy()
                        best_ap_count = ap_count.copy()
                        best_size = len(S)
                    break
            
            if not added:
                # Try Swap: remove y, add x
                swapped = False
                for _ in range(50):
                    if not S:
                        break
                    x = random.choice(all_elements)
                    if x in S:
                        continue
                    y = random.choice(list(S))
                    
                    # Check if S \ {y} + {x} is valid
                    # We need to temporarily remove y, check x, then decide
                    # Optimization: check if x is valid in S \ {y}
                    # x is valid in S \ {y} if:
                    # 1. x is not in S (already checked)
                    # 2. For any AP P containing x, count(P in S \ {y}) < 3
                    #    count(P in S \ {y}) = count(P in S) - (1 if y in P else 0)
                    #    So we need count(P in S) - (1 if y in P else 0) < 3
                    #    i.e., count(P in S) < 3 + (1 if y in P else 0)
                    
                    valid = True
                    for ap_idx in aps_containing[x]:
                        current_count = ap_count[ap_idx]
                        y_in_ap = 1 if (ap_list[ap_idx][0] == y or ap_list[ap_idx][1] == y or 
                                        ap_list[ap_idx][2] == y or ap_list[ap_idx][3] == y) else 0
                        if current_count - y_in_ap >= 3:
                            valid = False
                            break
                    
                    if valid:
                        # Perform swap
                        S.remove(y)
                        for ap_idx in aps_containing[y]:
                            ap_count[ap_idx] -= 1
                        
                        S.add(x)
                        for ap_idx in aps_containing[x]:
                            ap_count[ap_idx] += 1
                        
                        swapped = True
                        break
                
                if not swapped:
                    # Try Remove and Refill
                    if S:
                        y = random.choice(list(S))
                        S_old = S.copy()
                        ap_count_old = ap_count.copy()
                        
                        S.remove(y)
                        for ap_idx in aps_containing[y]:
                            ap_count[ap_idx] -= 1
                        
                        # Try to refill
                        # Shuffle remaining elements
                        candidates = [z for z in all_elements if z not in S]
                        random.shuffle(candidates)
                        
                        for z in candidates:
                            if is_valid_addition(ap_count, z):
                                S.add(z)
                                for ap_idx in aps_containing[z]:
                                    ap_count[ap_idx] += 1
                            
                        if len(S) > len(S_old):
                            best_S = S.copy()
                            best_ap_count = ap_count.copy()
                            best_size = len(S)
                        else:
                            S = S_old
                            ap_count = ap_count_old
        
        return best_S, best_ap_count

    best_S = set()
    best_size = 0
    
    # Run multiple greedy restarts
    # Number of restarts depends on time
    # Each greedy run is O(N * degree) ~ 512 * 100 ~ 50k ops. Very fast.
    # We can run many restarts.
    
    num_restarts = 200
    
    for i in range(num_restarts):
        order = list(range(1, N + 1))
        random.shuffle(order)
        S, ap_count = greedy_construct(order)
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
            
        # Apply local search to this S
        if time.time() - start_time < time_limit - 5:
            S_ls, ap_count_ls = local_search(S, ap_count)
            if len(S_ls) > best_size:
                best_S = S_ls
                best_size = len(S_ls)
                
    return sorted(best_S)

# EVOLVE-BLOCK-END
