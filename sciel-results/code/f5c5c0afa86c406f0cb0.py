"""
4-term progression-free set construction for [1, 512].
Uses Simulated Annealing with a penalty function on the number of 4-APs.
"""
import numpy as np
import time

N = 512
K = 4

# Precompute all 4-APs
aps = []
for a in range(1, N + 1):
    max_d = (N - a) // 3
    for d in range(1, max_d + 1):
        aps.append((a, a + d, a + 2 * d, a + 3 * d))

num_aps = len(aps)
# Map vertex -> list of AP indices
vertex_to_aps = [[] for _ in range(N + 1)]
for i, ap in enumerate(aps):
    for v in ap:
        vertex_to_aps[v].append(i)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # State
    S = np.zeros(N + 1, dtype=bool)
    count_in_ap = np.zeros(num_aps, dtype=int)
    bad_count = 0
    current_size = 0
    
    # Greedy initialization
    for x in range(1, N + 1):
        valid = True
        for ap_idx in vertex_to_aps[x]:
            if count_in_ap[ap_idx] == 3:
                valid = False
                break
        
        if valid:
            S[x] = True
            current_size += 1
            for ap_idx in vertex_to_aps[x]:
                count_in_ap[ap_idx] += 1
                if count_in_ap[ap_idx] == 4:
                    bad_count += 1
            
    best_size = current_size
    best_S = S.copy()
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    rng = np.random.default_rng(42)
    M = 100.0 # Penalty weight
    
    while time.time() < deadline and T > 0.01:
        x = rng.integers(1, N + 1)
        
        if S[x]:
            # Remove x
            delta_bad = 0
            for ap_idx in vertex_to_aps[x]:
                if count_in_ap[ap_idx] == 4:
                    delta_bad -= 1
            
            delta_obj = -1.0 - M * delta_bad # -1 size, -M * delta_bad (delta_bad is negative, so +)
            
            if delta_obj > 0 or (rng.random() < np.exp(delta_obj / T)):
                S[x] = False
                current_size -= 1
                bad_count += delta_bad
                for ap_idx in vertex_to_aps[x]:
                    count_in_ap[ap_idx] -= 1
                    
                if bad_count == 0 and current_size > best_size:
                    best_size = current_size
                    best_S = S.copy()
        else:
            # Add x
            delta_bad = 0
            for ap_idx in vertex_to_aps[x]:
                if count_in_ap[ap_idx] == 3:
                    delta_bad += 1
            
            delta_obj = 1.0 - M * delta_bad
            
            if delta_obj > 0 or (rng.random() < np.exp(delta_obj / T)):
                S[x] = True
                current_size += 1
                bad_count += delta_bad
                for ap_idx in vertex_to_aps[x]:
                    count_in_ap[ap_idx] += 1
                    
                if bad_count == 0 and current_size > best_size:
                    best_size = current_size
                    best_S = S.copy()
        
        T *= cooling

    # Final local search on best_S
    S_final = best_S.copy()
    count_final = np.zeros(num_aps, dtype=int)
    for x in range(1, N + 1):
        if S_final[x]:
            for ap_idx in vertex_to_aps[x]:
                count_final[ap_idx] += 1
                
    improved = True
    while improved:
        improved = False
        for x in range(1, N + 1):
            if not S_final[x]:
                valid = True
                for ap_idx in vertex_to_aps[x]:
                    if count_final[ap_idx] == 3:
                        valid = False
                        break
                if valid:
                    S_final[x] = True
                    for ap_idx in vertex_to_aps[x]:
                        count_final[ap_idx] += 1
                    improved = True
                    
    return sorted(np.where(S_final)[0])
