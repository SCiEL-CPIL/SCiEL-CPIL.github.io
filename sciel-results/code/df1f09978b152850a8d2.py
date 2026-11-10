import time
import random

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85
    n = 256
    
    # Precompute
    three_aps = []
    for d in range(1, n + 1):
        for a in range(1, n - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))
    
    four_aps = []
    for d in range(1, n + 1):
        for a in range(1, n - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    elem_3aps = {x: [] for x in range(1, n + 1)}
    for i, (a, b, c) in enumerate(three_aps):
        elem_3aps[a].append(i)
        elem_3aps[b].append(i)
        elem_3aps[c].append(i)
    
    elem_4aps = {x: [] for x in range(1, n + 1)}
    for i, (a, b, c, d4) in enumerate(four_aps):
        elem_4aps[a].append(i)
        elem_4aps[b].append(i)
        elem_4aps[c].append(i)
        elem_4aps[d4].append(i)
    
    def check_4ap_free(selected, x):
        for ap_idx in elem_4aps[x]:
            a, b, c, d4 = four_aps[ap_idx]
            others = [e for e in (a, b, c, d4) if e != x]
            if all(e in selected for e in others):
                return False
        return True
    
    def count_new_3aps(selected, x):
        cnt = 0
        for ap_idx in elem_3aps[x]:
            a, b, c = three_aps[ap_idx]
            others = [e for e in (a, b, c) if e != x]
            if all(e in selected for e in others):
                cnt += 1
        return cnt
    
    best_set = set()
    best_count = 0
    
    # Multiple greedy starts
    seeds = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67]
    
    for s in seeds:
        if time.time() > DEADLINE:
            break
        
        selected = {s}
        cur_count = 0
        
        # Greedy phase
        while time.time() < DEADLINE:
            best_gain = 0
            best_x = None
            for x in range(1, n + 1):
                if x in selected:
                    continue
                if not check_4ap_free(selected, x):
                    continue
                g = count_new_3aps(selected, x)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x is None:
                break
            selected.add(best_x)
            cur_count += best_gain
        
        # Add phase: try to add any element that increases 3-APs
        if time.time() < DEADLINE:
            improved = True
            while improved and time.time() < DEADLINE:
                improved = False
                for x in range(1, n + 1):
                    if x in selected:
                        continue
                    if not check_4ap_free(selected, x):
                        continue
                    g = count_new_3aps(selected, x)
                    if g > 0:
                        selected.add(x)
                        cur_count += g
                        improved = True
        
        if cur_count > best_count:
            best_count = cur_count
            best_set = set(selected)
    
    # Swap-based local search
    if time.time() < DEADLINE and best_set:
        current = set(best_set)
        current_count = best_count
        
        for _ in range(50):
            if time.time() > DEADLINE:
                break
            best_net = 0
            best_action = None
            
            for rem in list(current):
                trial = set(current)
                trial.discard(rem)
                lost = count_new_3aps(trial, rem)  # 3-APs that rem was completing
                added = 0
                for x in range(1, n + 1):
                    if x in trial:
                        continue
                    if not check_4ap_free(trial, x):
                        continue
                    g = count_new_3aps(trial, x)
                    if g > 0:
                        trial.add(x)
                        added += g
                net = added - lost
                if net > best_net:
                    best_net = net
                    best_action = (rem, set(trial), current_count + net)
            
            if best_action is None or best_net <= 0:
                break
            rem, new_set, new_count = best_action
            current = new_set
            current_count = new_count
        
        if current_count > best_count:
            best_count = current_count
            best_set = current
    
    return sorted(best_set)
