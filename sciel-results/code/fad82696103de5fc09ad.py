import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Precompute APs and Incidence Matrix
    ap_terms = []
    for d in range(1, N + 1):
        for s in range(0, N - d + 1):
            terms = list(range(s, N, d))
            if len(terms) > 1:
                ap_terms.append(terms)
                
    num_aps = len(ap_terms)
    
    # Build M (num_aps x N)
    M = np.zeros((num_aps, N), dtype=np.float32)
    pos_to_aps = [[] for _ in range(N)]
    
    for j, terms in enumerate(ap_terms):
        for i in terms:
            M[j, i] = 1.0
            pos_to_aps[i].append(j)
            
    pos_to_aps_arr = [np.array(aps, dtype=np.int32) for aps in pos_to_aps]
    
    # Compute C = M.T @ M and precompute columns for fast updates
    C = M.T @ M
    K = np.diag(C).astype(np.float32)
    C_cols = [C[:, k] for k in range(N)]  # Precomputed columns
    
    def get_hard_max(sums):
        return int(np.max(np.abs(sums)))
    
    best_signs = None
    best_hard_max = N
    
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000]
    
    for seed in seeds:
        if time.time() > deadline - 8:
            break
            
        rng = random.Random(seed)
        
        # Initialize with random signs
        signs = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.float32)
        
        # Initial sums
        sums = M @ signs
        
        # Initial G = C @ signs
        G = C @ signs
        
        current_hard_max = get_hard_max(sums)
        
        # Phase 1: Fast SA with sum-of-squares objective
        T = 10.0
        steps = 0
        best_hard_this_run = current_hard_max
        
        while time.time() < deadline - 6:
            # Pick random index to flip
            k = rng.randint(0, N - 1)
            
            delta = -2.0 * signs[k]
            
            # Compute delta_J using precomputed G and K
            delta_J = 2.0 * delta * G[k] + (delta * delta) * K[k]
            
            # Metropolis Criterion
            accept = False
            if delta_J < 0:
                accept = True
            else:
                if delta_J / T > 100:
                    accept = False
                else:
                    accept = rng.random() < math.exp(-delta_J / T)
                    
            if accept:
                signs[k] = -signs[k]
                idxs = pos_to_aps_arr[k]
                sums[idxs] += delta
                G += delta * C_cols[k]
                
                steps += 1
                
                # Check hard max periodically
                if steps % 30 == 0:
                    current_hard_max = get_hard_max(sums)
                    if current_hard_max < best_hard_max:
                        best_hard_max = current_hard_max
                        best_signs = signs.copy()
                    elif current_hard_max < best_hard_this_run:
                        best_hard_this_run = current_hard_max
                        
            # Annealing - faster cooling
            if steps % 200 == 0:
                T *= 0.96
                if T < 0.1:
                    T = 0.1
                    
            if best_hard_max <= 1:
                break
                
        # Phase 2: Focused refinement on worst APs
        if best_signs is not None and time.time() < deadline - 2:
            signs = best_signs.copy()
            sums = M @ signs
            G = C @ signs
            current_hard_max = get_hard_max(sums)
            
            abs_sums = np.abs(sums)
            worst_aps = np.where(abs_sums >= current_hard_max - 1)[0]
            
            polish_steps = 0
            polish_T = 1.0
            best_polish_hard = current_hard_max
            
            while time.time() < deadline - 1 and polish_steps < 30000:
                # Focus on positions that affect the worst APs
                candidate_positions = set()
                for ap_idx in worst_aps[:15]:
                    candidate_positions.update(ap_terms[ap_idx])
                
                if not candidate_positions:
                    break
                    
                k = rng.choice(list(candidate_positions))
                
                delta = -2.0 * signs[k]
                
                idxs = pos_to_aps_arr[k]
                affected_sums = sums[idxs]
                
                old_max = np.max(np.abs(affected_sums))
                new_sums_affected = affected_sums + delta
                new_max = np.max(np.abs(new_sums_affected))
                
                # Accept if reduces max, or keeps same but reduces sum of squares
                if new_max < old_max:
                    accept = True
                elif new_max == old_max:
                    delta_J = 2.0 * delta * G[k] + (delta * delta) * K[k]
                    accept = delta_J < 0 or rng.random() < math.exp(-delta_J / polish_T)
                else:
                    accept = False
                
                if accept:
                    signs[k] = -signs[k]
                    sums[idxs] += delta
                    G += delta * C_cols[k]
                    polish_steps += 1
                    
                    if polish_steps % 40 == 0:
                        current_hard_max = get_hard_max(sums)
                        if current_hard_max < best_polish_hard:
                            best_polish_hard = current_hard_max
                        if current_hard_max < best_hard_max:
                            best_hard_max = current_hard_max
                            best_signs = signs.copy()
                        
                        # Update worst APs
                        abs_sums = np.abs(sums)
                        worst_aps = np.where(abs_sums >= current_hard_max - 1)[0]
                        
                        if len(worst_aps) == 0:
                            break
                            
                # Cool down
                if polish_steps % 800 == 0:
                    polish_T *= 0.9
                    if polish_T < 0.1:
                        polish_T = 0.1
                        
    if best_signs is None:
        return [1 if random.random() < 0.5 else -1 for _ in range(N)]
        
    return [int(s) for s in best_signs]
