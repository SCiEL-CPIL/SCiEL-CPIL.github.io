import time
import random
from math import gcd

def construct():
    start_time = time.time()
    time_limit = 55.0  # 5s margin
    
    N = 128
    G = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            G[i][j] = gcd(i, j)
            G[j][i] = G[i][j]
            
    def is_valid_addition(S_set, S_list, x):
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            g_ax = G[a][x]
            # Optimization: if g_ax is 1, can it be part of a bad triple?
            # gcd(a,b)=1, gcd(a,x)=1, gcd(b,x)=1.
            # If x is coprime to a and b, and a,b are coprime, then triple is bad (1,1,1).
            # So we must check all.
            for j in range(i + 1, n):
                b = S_list[j]
                g_bx = G[b][x]
                if g_ax == g_bx:
                    if G[a][b] == g_ax:
                        return False
        return True

    # Greedy init
    S_set = set()
    S_list = []
    for x in range(1, N + 1):
        if is_valid_addition(S_set, S_list, x):
            S_set.add(x)
            S_list.append(x)
            
    best_S_set = set(S_set)
    best_S_list = list(S_list)
    best_size = len(S_list)
    
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        current_size = len(S_list)
        
        # Try to add
        added = False
        # Try 50 random candidates
        for _ in range(50):
            x = random.randint(1, N)
            if x in S_set:
                continue
            if is_valid_addition(S_set, S_list, x):
                S_set.add(x)
                S_list.append(x)
                if len(S_list) > best_size:
                    best_size = len(S_list)
                    best_S_set = set(S_set)
                    best_S_list = list(S_list)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                added = True
                break
        
        if added:
            continue
            
        # Try to swap
        swapped = False
        for _ in range(50):
            # Pick x_in from S_list
            idx_in = random.randint(0, len(S_list) - 1)
            x_in = S_list[idx_in]
            
            # Pick x_out from [1, N] not in S_set
            while True:
                x_out = random.randint(1, N)
                if x_out not in S_set:
                    break
                    
            # Perform swap
            # Remove x_in
            S_set.remove(x_in)
            S_list.pop(idx_in)
            
            # Check if x_out can be added
            if is_valid_addition(S_set, S_list, x_out):
                S_set.add(x_out)
                S_list.append(x_out)
                if len(S_list) > best_size:
                    best_size = len(S_list)
                    best_S_set = set(S_set)
                    best_S_list = list(S_list)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                swapped = True
                break
            else:
                # Restore
                S_set.add(x_in)
                S_list.append(x_in) # Note: order changed, but set/list consistency is maintained for validity checks
                # Actually, S_list order doesn't matter for validity, only content.
                # But for next iteration, S_list must match S_set.
                # Since we just appended, it matches.
        
        if swapped:
            continue
            
        # Stuck: Shake up
        # Remove 1-3 random elements to create space
        # This allows "remove 1, add 2" moves in subsequent steps
        if S_list:
            num_remove = random.randint(1, 3)
            for _ in range(min(num_remove, len(S_list))):
                idx = random.randint(0, len(S_list) - 1)
                x_to_remove = S_list[idx]
                S_set.remove(x_to_remove)
                S_list.pop(idx)
            no_improve_count += 10 # Penalize shake-up
            
        # If stuck for too long, restart from best
        if no_improve_count > 200:
            S_set = set(best_S_set)
            S_list = list(best_S_list)
            no_improve_count = 0
            # Perturb best solution
            if S_list:
                for _ in range(random.randint(1, 5)):
                    if S_list:
                        idx = random.randint(0, len(S_list) - 1)
                        x_to_remove = S_list[idx]
                        S_set.remove(x_to_remove)
                        S_list.pop(idx)

    return list(best_S_list)
