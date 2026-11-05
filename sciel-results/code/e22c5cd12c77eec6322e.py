import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    N = q ** n
    
    if radius != 1:
        raise NotImplementedError("Radius > 1 not optimized")

    sphere_size = 1 + n * (q - 1)
    
    # Precompute neighbors
    neighbors = np.empty((N, sphere_size), dtype=int)
    for i in range(N):
        idx = 0
        neighbors[i, idx] = i
        idx += 1
        val = i
        for pos in range(n):
            orig_val = val % q
            val //= q
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                neighbors[i, idx] = i + (new_val - orig_val) * (q ** pos)
                idx += 1

    best_C = None
    best_len = N + 1
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Partition-based constructive approach:
    # Decompose space into 2187 planes (subspaces of dimension 2)
    # For each plane, select a minimum dominating set derived from ternary [2,1,2] code structure
    # Then apply global consistency and local search
    
    # Generate all 2D subspaces (planes) through origin in {0,1,2}^7
    # A plane is defined by two basis vectors
    # For efficiency, we'll use a structured partition approach
    
    # Strategy: Use a grid-like partition where we group words by their first (n-2) coordinates
    # Each group forms a "plane" of size q^2 = 9
    # For each plane, we need a minimum dominating set
    
    # For q=3, n=7: each plane has 9 words, we need ~3 codewords per plane (perfect 1-cover)
    # Total planes: 3^(7-2) = 243
    # Initial estimate: 243 * 3 = 729 codewords (upper bound)
    
    # Better approach: Use a more sophisticated partition
    # Let's use a greedy constructive approach with structured initialization
    
    def greedy_construct(seed=None):
        if seed is not None:
            random.seed(seed)
        
        covered = np.zeros(N, dtype=bool)
        cover_count = np.zeros(N, dtype=int)
        C = []
        
        # Greedy phase with randomization
        while covered.sum() < N:
            gains = (~covered[neighbors]).sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Find all indices with max gain
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1:
                # Randomly pick one to explore different basins
                best_idx = candidates[random.randint(0, len(candidates) - 1)]
            else:
                best_idx = candidates[0]
                
            C.append(best_idx)
            cover_count[neighbors[best_idx]] += 1
            covered[neighbors[best_idx]] = True
        
        return C, cover_count
    
    def local_search(C, cover_count):
        in_C = np.zeros(N, dtype=bool)
        for c in C:
            in_C[c] = True
            
        improved = True
        while improved:
            improved = False
            
            # Step A: Remove redundant codewords
            for i in range(len(C)):
                c = C[i]
                nb = neighbors[c]
                cover_count[nb] -= 1
                
                if np.any(cover_count[nb] == 0):
                    cover_count[nb] += 1
                else:
                    C.pop(i)
                    in_C[c] = False
                    improved = True
                    break
            
            if not improved and C:
                # Step B: Swap optimization
                best_swap = None
                best_delta = 0
                
                for c in C:
                    nb_c = neighbors[c]
                    critical_mask = (cover_count[nb_c] == 1)
                    critical_words = nb_c[critical_mask]
                    
                    if len(critical_words) == 0:
                        continue
                    
                    # Find candidates that cover all critical words
                    if len(critical_words) == 1:
                        candidates = neighbors[critical_words[0]]
                    else:
                        mask0 = np.zeros(N, dtype=bool)
                        mask0[neighbors[critical_words[0]]] = True
                        for w in critical_words[1:]:
                            mask_w = np.zeros(N, dtype=bool)
                            mask_w[neighbors[w]] = True
                            mask0 &= mask_w
                            if not mask0.any():
                                break
                        if not mask0.any():
                            continue
                        candidates = np.where(mask0)[0]
                    
                    mask = ~in_C[candidates]
                    if not np.any(mask):
                        continue
                    
                    valid_candidates = candidates[mask]
                    
                    for cp in valid_candidates:
                        nb_cp = neighbors[cp]
                        
                        in_c = np.zeros(N, dtype=bool)
                        in_c[nb_c] = True
                        in_cp = np.zeros(N, dtype=bool)
                        in_cp[nb_cp] = True
                        
                        only_c_mask = in_c & ~in_cp
                        only_cp_mask = in_cp & ~in_c
                        
                        if np.any(cover_count[only_c_mask] == 1):
                            continue
                        
                        delta = np.sum(cover_count[only_c_mask] == 2) - np.sum(cover_count[only_cp_mask] == 1)
                        
                        if delta < best_delta:
                            best_delta = delta
                            best_swap = (c, cp)
                
                if best_swap:
                    c, cp = best_swap
                    nb_c = neighbors[c]
                    nb_cp = neighbors[cp]
                    
                    cover_count[nb_c] -= 1
                    cover_count[nb_cp] += 1
                    
                    C.remove(c)
                    C.append(cp)
                    in_C[c] = False
                    in_C[cp] = True
                    improved = True
        
        return C, cover_count
    
    # Try multiple random seeds for greedy construction
    for seed in range(20):
        if time.time() - start_time > time_limit:
            break
        
        C, cover_count = greedy_construct(seed)
        C, cover_count = local_search(C, cover_count)
        
        if len(C) < best_len:
            best_len = len(C)
            best_C = C.copy()
    
    # Final optimization: try to remove more codewords
    if best_C:
        in_C = np.zeros(N, dtype=bool)
        for c in best_C:
            in_C[c] = True
        cover_count = np.zeros(N, dtype=int)
        for c in best_C:
            cover_count[neighbors[c]] += 1
        
        improved = True
        while improved:
            improved = False
            for i in range(len(best_C)):
                c = best_C[i]
                nb = neighbors[c]
                cover_count[nb] -= 1
                if np.any(cover_count[nb] == 0):
                    cover_count[nb] += 1
                else:
                    best_C.pop(i)
                    in_C[c] = False
                    improved = True
                    break
    
    result_words = []
    for idx in best_C:
        word = []
        val = idx
        for _ in range(n):
            word.append(val % q)
            val //= q
        result_words.append(word)
        
    return result_words
