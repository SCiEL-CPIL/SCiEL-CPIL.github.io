import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    N = 256
    
    # Precompute 4-APs
    all_4aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            all_4aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    elem_to_4aps = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(all_4aps):
        for x in ap:
            elem_to_4aps[x].append(idx)
            
    # Precompute 3-APs
    all_3aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            all_3aps.append((a, a + d, a + 2 * d))
            
    elem_to_3aps = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(all_3aps):
        for x in ap:
            elem_to_3aps[x].append(idx)

    A = np.zeros(N + 1, dtype=bool)
    current_3ap_count = 0
    
    def is_valid_add(x):
        for idx in elem_to_4aps[x]:
            ap = all_4aps[idx]
            cnt = 0
            for y in ap:
                if y != x and A[y]:
                    cnt += 1
                    if cnt >= 3:
                        return False
        return True
        
    def is_valid_swap(x, y):
        for idx in elem_to_4aps[y]:
            ap = all_4aps[idx]
            cnt = 0
            for z in ap:
                if z != y:
                    if A[z] and z != x:
                        cnt += 1
                        if cnt >= 3:
                            return False
        return True

    def gain_add(x):
        gain = 0
        for idx in elem_to_3aps[x]:
            ap = all_3aps[idx]
            cnt = 0
            for y in ap:
                if y != x and A[y]:
                    cnt += 1
            if cnt == 2:
                gain += 1
        return gain

    def loss_remove(x):
        loss = 0
        for idx in elem_to_3aps[x]:
            ap = all_3aps[idx]
            cnt = 0
            for y in ap:
                if y != x and A[y]:
                    cnt += 1
            if cnt == 2:
                loss += 1
        return loss

    def delta_swap(x, y):
        loss = 0
        for idx in elem_to_3aps[x]:
            ap = all_3aps[idx]
            cnt = 0
            for z in ap:
                if z != x and A[z]:
                    cnt += 1
            if cnt == 2:
                loss += 1
                
        gain = 0
        for idx in elem_to_3aps[y]:
            ap = all_3aps[idx]
            cnt = 0
            for z in ap:
                if z != y and A[z] and z != x:
                    cnt += 1
            if cnt == 2:
                gain += 1
                
        return gain - loss

    # Greedy Initialization
    remaining = set(range(1, N + 1))
    while remaining:
        best_x = None
        best_gain = -1
        for x in remaining:
            if is_valid_add(x):
                g = gain_add(x)
                if g > best_gain:
                    best_gain = g
                    best_x = x
        if best_x is None:
            break
        A[best_x] = True
        current_3ap_count += best_gain
        remaining.remove(best_x)
        
    best_set = A.copy()
    best_score = current_3ap_count

    T = 5.0
    cooling = 0.9999
    last_restart_time = start_time
    restart_interval = 5.0 # Restart every 5 seconds
    
    while time.time() < deadline:
        # Check if we should restart
        if time.time() - last_restart_time > restart_interval:
            # Perturb the current best set to escape local optimum
            # We copy best_set to A
            A[:] = best_set
            current_3ap_count = best_score
            
            # Perform a few random valid swaps to perturb
            # Note: is_valid_swap uses A. Since A is now best_set, this is consistent.
            for _ in range(10):
                x = random.randint(1, N)
                y = random.randint(1, N)
                if A[x] and not A[y]:
                    if is_valid_swap(x, y):
                        # Apply swap to A to perturb
                        A[x] = False
                        A[y] = True
                        # Update score? 
                        # We don't strictly need the exact score for the perturbed state 
                        # if we are going to run SA which tracks it, 
                        # but we need a consistent starting score.
                        # Let's just recalculate it to be safe.
                        pass
            
            # Recalculate score for the perturbed A
            current_3ap_count = 0
            for idx, ap in enumerate(all_3aps):
                if A[ap[0]] and A[ap[1]] and A[ap[2]]:
                    current_3ap_count += 1
            
            # Reset Temperature
            T = 5.0
            last_restart_time = time.time()

        r = random.random()
        
        if r < 0.3:
            x = random.randint(1, N)
            if not A[x]:
                if is_valid_add(x):
                    delta = gain_add(x)
                    if delta >= 0 or random.random() < np.exp(delta / T):
                        A[x] = True
                        current_3ap_count += delta
                        if current_3ap_count > best_score:
                            best_score = current_3ap_count
                            best_set = A.copy()
        elif r < 0.6:
            x = random.randint(1, N)
            if A[x]:
                delta = -loss_remove(x)
                if delta >= 0 or random.random() < np.exp(delta / T):
                    A[x] = False
                    current_3ap_count += delta
                    if current_3ap_count > best_score:
                        best_score = current_3ap_count
                        best_set = A.copy()
        else:
            x = random.randint(1, N)
            y = random.randint(1, N)
            if A[x] and not A[y]:
                if is_valid_swap(x, y):
                    delta = delta_swap(x, y)
                    if delta >= 0 or random.random() < np.exp(delta / T):
                        A[x] = False
                        A[y] = True
                        current_3ap_count += delta
                        if current_3ap_count > best_score:
                            best_score = current_3ap_count
                            best_set = A.copy()
        
        T *= cooling
        if T < 0.01:
            T = 0.01

    res = [i for i in range(1, N + 1) if best_set[i]]
    if not res:
        return [1]
    return res
