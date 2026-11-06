import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try to find valid coloring for n=26
    for target_n in [26, 27, 28]:
        if time.time() > deadline - 10: break
        
        # Initialize
        n = target_n
        # colors as flat list or 2D array? 2D is easier for logic, flat for return.
        # We use a list of lists for masks.
        # masks[c][v] is a Python int (bitmask)
        
        # Random init
        colors = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                colors[i][j] = colors[j][i] = random.randint(0, 4)
                
        # Build masks
        masks = [[0]*n for _ in range(5)]
        for i in range(n):
            for j in range(i+1, n):
                c = colors[i][j]
                masks[c][i] |= (1 << j)
                masks[c][j] |= (1 << i)
                
        # Calculate initial energy
        # E = sum_c sum_{i<j} C(|N_c(i) & N_c(j)|, 2)
        
        def calc_energy(masks, n):
            e = 0
            for c in range(5):
                for i in range(n):
                    mi = masks[c][i]
                    for j in range(i+1, n):
                        inter = mi & masks[c][j]
                        # count bits
                        k = bin(inter).count('1')
                        if k >= 2:
                            e += k * (k - 1) // 2
            return e
            
        current_energy = calc_energy(masks, n)
        
        # Local Search
        T = 1.0
        cooling = 0.9999
        min_T = 0.01
        
        while time.time() < deadline - 5:
            if current_energy == 0:
                # Found valid coloring for target_n
                best_n = n
                # Flatten colors
                flat_colors = []
                for i in range(n):
                    for j in range(i+1, n):
                        flat_colors.append(colors[i][j])
                best_colors = flat_colors
                # Try next n if time permits, or just break if we want to be safe?
                # The task says maximize n. If we found 26, we should try 27 if time allows.
                # But 27 is very hard. Let's break and try to polish or just return.
                # Actually, if we found 26, score is 26/11 = 2.36.
                # If we try 27 and fail, we still have 26.
                # Let's break the loop for this target_n and move to next.
                break
                
            # Pick random edge
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j: continue
            if i > j: i, j = j, i
            
            c_old = colors[i][j]
            c_new = random.randint(0, 4)
            if c_old == c_new: continue
            
            # Calculate delta energy
            # We need to update masks temporarily to check?
            # Or calculate delta analytically.
            
            # Affected pairs are those involving i or j.
            # Let's just compute energy before and after for the affected subset?
            # No, the global energy only changes for terms involving i or j.
            # E_old_part = sum of terms for pairs (k,l) where k in {i,j} or l in {i,j}
            # E_new_part = same with updated masks
            
            # To be safe and simple, let's just do a "trial" update on masks,
            # calculate the change in energy for the affected pairs,
            # and revert if rejected.
            
            # Save old masks for i, j in c_old and c_new
            # Actually, we can just compute the delta directly.
            
            # Let's define a helper to get the contribution of a pair (u, v) to the energy of color c
            def pair_contrib(u, v, c, masks):
                inter = masks[c][u] & masks[c][v]
                k = bin(inter).count('1')
                return k * (k - 1) // 2 if k >= 2 else 0
            
            # Pairs to check:
            # 1. (i, j) itself? No, energy is sum over pairs of vertices.
            #    The term is for pair of vertices (u, v).
            #    The change in masks for i and j affects:
            #    - Pairs (i, k) for all k != i
            #    - Pairs (j, k) for all k != j
            #    - Pair (i, j)
            
            # Let's list all pairs (u, v) with u < v that involve i or j.
            affected = set()
            for k in range(n):
                if k == i or k == j: continue
                a, b = (i, k) if i < k else (k, i)
                affected.add((a, b))
                a, b = (j, k) if j < k else (k, j)
                affected.add((a, b))
            affected.add((i, j)) # if i<j
            
            delta_e = 0
            for c in range(5):
                if c not in [c_old, c_new]: continue
                
                # For each affected pair, calculate contrib before and after
                # This requires knowing the new masks.
                # Let's simulate the mask change for c_old and c_new for vertices i and j.
                
                # This is getting complicated to do analytically without modifying global state.
                # Given n=26, modifying global state, calculating delta, and reverting is fast enough.
                pass
                
            # Strategy:
            # 1. Modify masks for c_old (remove i,j) and c_new (add i,j)
            # 2. Calculate energy change for affected pairs
            # 3. If accepted, keep changes and update colors. Else revert.
            
            # Save states
            old_mi_cold = masks[c_old][i]
            old_mj_cold = masks[c_old][j]
            old_mi_cnew = masks[c_new][i]
            old_mj_cnew = masks[c_new][j]
            
            # Apply changes
            masks[c_old][i] &= ~(1 << j)
            masks[c_old][j] &= ~(1 << i)
            masks[c_new][i] |= (1 << j)
            masks[c_new][j] |= (1 << i)
            
            # Calculate delta
            # We need sum of pair_contrib for affected pairs in c_old and c_new.
            # Note: pair_contrib depends on masks.
            
            delta = 0
            for u, v in affected:
                # Check if this pair's contribution changed in c_old or c_new
                # It only changes if u or v is i or j.
                # Which it is, by construction of 'affected'.
                
                # But we only care about colors c_old and c_new.
                # For other colors, masks didn't change, so contrib is same.
                
                # For c_old:
                delta -= pair_contrib(u, v, c_old, masks) # subtract new
                # Wait, we need (New - Old).
                # So delta += New - Old.
                # We overwrote masks. We need Old values.
                # This approach requires storing old contribs or recalculating.
                
                # Let's restart this specific block with a cleaner "calculate delta" logic.
                pass
            
            # Revert masks for calculation
            masks[c_old][i] = old_mi_cold
            masks[c_old][j] = old_mj_cold
            masks[c_new][i] = old_mi_cnew
            masks[c_new][j] = old_mj_cnew
            
            # Calculate Old Energy for affected pairs
            old_e = 0
            for u, v in affected:
                old_e += pair_contrib(u, v, c_old, masks)
                old_e += pair_contrib(u, v, c_new, masks)
                
            # Apply changes again
            masks[c_old][i] &= ~(1 << j)
            masks[c_old][j] &= ~(1 << i)
            masks[c_new][i] |= (1 << j)
            masks[c_new][j] |= (1 << i)
            
            # Calculate New Energy for affected pairs
            new_e = 0
            for u, v in affected:
                new_e += pair_contrib(u, v, c_old, masks)
                new_e += pair_contrib(u, v, c_new, masks)
                
            delta = new_e - old_e
            
            # Metropolis step
            if delta <= 0 or random.random() < np.exp(-delta / T):
                # Accept
                colors[i][j] = c_new
                colors[j][i] = c_new
                current_energy += delta
            else:
                # Reject, revert masks
                masks[c_old][i] = old_mi_cold
                masks[c_old][j] = old_mj_cold
                masks[c_new][i] = old_mi_cnew
                masks[c_new][j] = old_mj_cnew
                
            T *= cooling
            if T < min_T:
                # Restart with new random init if stuck?
                # Or just break and try next target_n
                break
                
        # End while for target_n
        
    return {'n': best_n, 'colors': best_colors}
