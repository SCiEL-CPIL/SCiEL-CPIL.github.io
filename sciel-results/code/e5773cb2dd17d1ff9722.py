import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    N = q ** n
    
    if radius != 1:
        raise NotImplementedError("Radius > 1 not optimized")

    sphere_size = 1 + n * (q - 1)
    
    # Precompute neighbors for all words
    neighbors = np.empty((N, sphere_size), dtype=int)
    for i in range(N):
        idx = 0
        neighbors[i, idx] = i
        idx += 1
        val = i
        for pos in range(n):
            power = q ** pos
            orig_val = (i // power) % q
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                neighbors[i, idx] = i + (new_val - orig_val) * power
                idx += 1

    start_time = time.time()
    time_limit = 110  # Leave 10s buffer

    best_C = None
    best_len = N + 1

    while time.time() - start_time < time_limit:
        # Phase 1: Greedy construction with randomization
        covered = np.zeros(N, dtype=bool)
        cover_count = np.zeros(N, dtype=int)
        C = []
        
        while covered.sum() < N:
            gains = (~covered[neighbors]).sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1:
                best_idx = candidates[random.randint(0, len(candidates) - 1)]
            else:
                best_idx = candidates[0]
                
            C.append(best_idx)
            cover_count[neighbors[best_idx]] += 1
            covered[neighbors[best_idx]] = True
        
        if len(C) >= best_len:
            continue
            
        # Phase 2: Local search optimization
        in_C = np.zeros(N, dtype=bool)
        for c in C:
            in_C[c] = True
            
        improved = True
        while improved:
            improved = False
            
            # Step A: Remove redundant codewords
            i = 0
            while i < len(C):
                c = C[i]
                nb = neighbors[c]
                cover_count[nb] -= 1
                
                if np.any(cover_count[nb] == 0):
                    cover_count[nb] += 1
                else:
                    C.pop(i)
                    in_C[c] = False
                    improved = True
                    continue
                i += 1
            
            if not improved and C:
                # Step B: Try swaps to improve
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
                    
                    # Try each candidate swap
                    for cp in valid_candidates:
                        nb_cp = neighbors[cp]
                        
                        # Compute what changes
                        in_c_mask = np.zeros(N, dtype=bool)
                        in_c_mask[nb_c] = True
                        in_cp_mask = np.zeros(N, dtype=bool)
                        in_cp_mask[nb_cp] = True
                        
                        only_c = in_c_mask & ~in_cp_mask
                        only_cp = in_cp_mask & ~in_c_mask
                        
                        # Check validity
                        if np.any(cover_count[only_c] == 1):
                            continue
                        
                        delta = np.sum(cover_count[only_c] == 2) - np.sum(cover_count[only_cp] == 1)
                        
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
        
        if len(C) < best_len:
            best_len = len(C)
            best_C = C.copy()
            
    # Final pass: try to remove more codewords from best solution
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
            i = 0
            while i < len(best_C):
                c = best_C[i]
                nb = neighbors[c]
                cover_count[nb] -= 1
                if np.any(cover_count[nb] == 0):
                    cover_count[nb] += 1
                else:
                    best_C.pop(i)
                    in_C[c] = False
                    improved = True
                    continue
                i += 1
    
    # Convert indices to words
    result_words = []
    for idx in best_C:
        word = []
        val = idx
        for _ in range(n):
            word.append(val % q)
            val //= q
        result_words.append(word)
        
    return result_words
