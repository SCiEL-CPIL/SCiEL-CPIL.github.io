import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    N = q ** n
    
    if radius != 1:
        raise NotImplementedError("Radius > 1 not optimized")

    sphere_size = 1 + n * (q - 1)
    
    # Precompute neighbors as lists and masks
    neighbors_list = []
    neighbors_mask = [0] * N
    
    powers = [1]
    for i in range(1, n):
        powers.append(powers[-1] * q)
        
    for i in range(N):
        nb_list = [i]
        mask = 1 << i
        val = i
        for pos in range(n):
            orig_val = val % q
            val //= q
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_idx = i + (new_val - orig_val) * powers[pos]
                nb_list.append(new_idx)
                mask |= (1 << new_idx)
        neighbors_list.append(nb_list)
        neighbors_mask[i] = mask

    full_mask = (1 << N) - 1
    best_C = None
    best_len = N + 1
    start_time = time.time()
    time_limit = 110
    
    while time.time() - start_time < time_limit:
        covered_mask = 0
        cover_count = np.zeros(N, dtype=int)
        C = []
        in_C = [False] * N
        
        # 1. Greedy Phase
        while covered_mask != full_mask:
            best_idx = -1
            best_gain = -1
            # Random tie break
            candidates = []
            for i in range(N):
                if in_C[i]:
                    continue
                gain = bin(neighbors_mask[i] & ~covered_mask).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    candidates = [i]
                elif gain == best_gain:
                    candidates.append(i)
            
            if best_idx == -1:
                break
            if len(candidates) > 1:
                best_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            C.append(best_idx)
            in_C[best_idx] = True
            covered_mask |= neighbors_mask[best_idx]
            for nb in neighbors_list[best_idx]:
                cover_count[nb] += 1
        
        if len(C) >= best_len:
            continue
            
        # 2. Local Search
        improved = True
        while improved:
            improved = False
            
            # Step A: Remove redundant
            for i in range(len(C)):
                c = C[i]
                # Check if any neighbor has count 1
                redundant = False
                for nb in neighbors_list[c]:
                    if cover_count[nb] == 1:
                        redundant = True
                        break
                if not redundant:
                    # Remove c
                    for nb in neighbors_list[c]:
                        cover_count[nb] -= 1
                    C.pop(i)
                    in_C[c] = False
                    improved = True
                    break
            
            if not improved and C:
                # Step B: Swap
                best_swap = None
                best_delta = 0 # We want to minimize "cost" or maximize "redundancy potential"
                
                # Try to find a swap that increases the minimum cover count or similar
                # Simple heuristic: Swap if it doesn't decrease the number of words with count >= 2?
                # Or just try all swaps and keep the one that allows a removal?
                
                # For speed, limit swaps per pass or use a better metric.
                # Metric: Sum of cover counts? No.
                # Let's try: For each c, find critical words. Find x that covers them.
                # If such x exists, and x is not in C, we can swap.
                # Does this help? It maintains coverage.
                # It might make another word redundant.
                
                for c in C:
                    critical = []
                    for nb in neighbors_list[c]:
                        if cover_count[nb] == 1:
                            critical.append(nb)
                    
                    if not critical:
                        continue # Should have been removed in Step A
                        
                    # Find candidates x such that N(x) covers all critical
                    # x not in C
                    found_swap = False
                    for x in range(N):
                        if in_C[x]:
                            continue
                        # Check if x covers all critical
                        ok = True
                        for w in critical:
                            if not (neighbors_mask[x] & (1 << w)):
                                ok = False
                                break
                        if ok:
                            # Check if swap is beneficial?
                            # Just do it?
                            # We need to ensure we don't loop.
                            # Only swap if it increases "redundancy"
                            # Let's calculate delta in cover counts.
                            # Remove c, add x.
                            # If this makes some other word redundant, good.
                            
                            # For now, just perform the swap if it results in a different configuration
                            # and hope the next removal step cleans up.
                            # To avoid oscillation, only swap if x is "better" than c?
                            # Define better: x covers more words with count 1?
                            
                            # Simple: Just do the first valid swap found?
                            # This might be slow.
                            
                            # Let's just do it.
                            for nb in neighbors_list[c]:
                                cover_count[nb] -= 1
                            for nb in neighbors_list[x]:
                                cover_count[nb] += 1
                            
                            C.remove(c)
                            C.append(x)
                            in_C[c] = False
                            in_C[x] = True
                            improved = True
                            found_swap = True
                            break
                    if found_swap:
                        break
                        
        if len(C) < best_len:
            best_len = len(C)
            best_C = C.copy()
            
    # Final cleanup
    if best_C:
        in_C = [False] * N
        for c in best_C:
            in_C[c] = True
        cover_count = np.zeros(N, dtype=int)
        for c in best_C:
            for nb in neighbors_list[c]:
                cover_count[nb] += 1
        
        improved = True
        while improved:
            improved = False
            for i in range(len(best_C)):
                c = best_C[i]
                redundant = False
                for nb in neighbors_list[c]:
                    if cover_count[nb] == 1:
                        redundant = True
                        break
                if not redundant:
                    for nb in neighbors_list[c]:
                        cover_count[nb] -= 1
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
