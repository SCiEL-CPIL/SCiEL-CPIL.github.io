import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 58.0  # Safety margin
    
    N = 128
    
    # Pre-compute all 4-term APs
    aps = []
    ap_indices = [[] for _ in range(N)]
    for d in range(1, N // 3 + 1):
        for i in range(N - 3 * d):
            ap = (i, i + d, i + 2 * d, i + 3 * d)
            ap_idx = len(aps)
            aps.append(ap)
            for idx in ap:
                ap_indices[idx].append(ap_idx)
                
    num_aps = len(aps)
    
    def count_distinct_in_ap(ap, colors):
        return len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]})
    
    def is_valid_ap(ap, colors):
        return count_distinct_in_ap(ap, colors) >= 3

    def compute_violation_count(colors):
        count = 0
        for ap in aps:
            if count_distinct_in_ap(ap, colors) < 3:
                count += 1
        return count

    def run_seed():
        colors=[]
        for x in range(1,129):
            for color in range(128):
                if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                    and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                    colors.append(color);break
        return colors

    seed_colors = run_seed()
    best_colors = seed_colors.copy()
    best_k = len(set(seed_colors))
    
    # Try local search for each target_k from 3 up to best_k-1
    for target_k in range(3, best_k):
        if time.time() > DEADLINE - 2:
            break
            
        k_start = time.time()
        k_budget = 4.0
        
        found = False
        for restart in range(200):
            if time.time() > k_start + k_budget:
                break
            
            # Random initialization
            colors = [random.randint(0, target_k - 1) for _ in range(N)]
            
            # Local search with proper revert logic
            current_violations = compute_violation_count(colors)
            
            max_steps = 5000
            for step in range(max_steps):
                if time.time() > k_start + k_budget:
                    break
                    
                if current_violations == 0:
                    found = True
                    break
                    
                # Pick a random position to try changing
                idx = random.randint(0, N - 1)
                old_c = colors[idx]
                new_c = random.randint(0, target_k - 1)
                
                if new_c == old_c:
                    continue
                
                # Try the move
                colors[idx] = new_c
                new_violations = compute_violation_count(colors)
                
                if new_violations < current_violations:
                    # Accept
                    current_violations = new_violations
                else:
                    # Revert
                    colors[idx] = old_c
            
            if found:
                k_used = len(set(colors))
                if k_used < best_k:
                    best_colors = colors.copy()
                    best_k = k_used
                return best_colors

    return best_colors
