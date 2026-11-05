import random
import time

def construct():
    target_ns = [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]
    
    start_time = time.time()
    time_limit = 115.0
    
    # Helper to get index
    def get_idx(n, i, j):
        if i > j: i, j = j, i
        return i * n - i * (i + 1) // 2 + (j - i - 1)

    def count_c4s_edge_mask(c, i, j, adj_masks, n):
        # adj_masks[c] is a list of ints
        ni = adj_masks[c][i] & ~(1 << j)
        nj = adj_masks[c][j] & ~(1 << i)
        
        count = 0
        mask = nj
        while mask:
            lsb = mask & -mask
            x = lsb.bit_length() - 1
            # Count y in ni such that (x,y) is in E_c
            # adj_masks[c][x] & ni
            count += (adj_masks[c][x] & ni).bit_count()
            mask &= mask - 1
        return count

    def total_energy_mask(colors, n):
        adj_masks = [[0] * n for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(n, i, j)
                c = colors[idx]
                adj_masks[c][i] |= (1 << j)
                adj_masks[c][j] |= (1 << i)
        
        total = 0
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    # Check if edge (i,j) is color c
                    if (adj_masks[c][i] >> j) & 1:
                        total += count_c4s_edge_mask(c, i, j, adj_masks, n)
        return total // 4

    # Main loop
    best_n_found = 0
    best_colors_found = None
    
    for n in target_ns:
        if time.time() - start_time > time_limit:
            break
            
        # Try ILS
        best_energy = float('inf')
        best_colors = None
        
        # Number of ILS iterations
        max_iters = 10
        sa_steps_per_iter = 2000
        T_start = 1.0
        T_end = 0.01
        cooling = (T_end / T_start) ** (1.0 / sa_steps_per_iter)
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            # Init random
            colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
            
            # Build masks
            adj_masks = [[0] * n for _ in range(5)]
            for i in range(n):
                for j in range(i + 1, n):
                    idx = get_idx(n, i, j)
                    c = colors[idx]
                    adj_masks[c][i] |= (1 << j)
                    adj_masks[c][j] |= (1 << i)
                    
            energy = total_energy_mask(colors, n)
            
            if energy == 0:
                best_energy = 0
                best_colors = colors.copy()
                break
                
            T = T_start
            for step in range(sa_steps_per_iter):
                if time.time() - start_time > time_limit:
                    break
                    
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j: continue
                
                idx_ij = get_idx(n, i, j)
                c_old = colors[idx_ij]
                c_new = random.randint(0, 4)
                if c_old == c_new: continue
                
                # Delta calculation
                # Remove (i,j) from c_old
                # Add (i,j) to c_new
                
                # Note: count_c4s_edge_mask assumes (i,j) is PRESENT in the mask for the color.
                # For c_old, (i,j) IS present.
                # For c_new, (i,j) is NOT present.
                
                # Delta for c_old (removal):
                # Current contribution of (i,j) in c_old.
                contrib_old = count_c4s_edge_mask(c_old, i, j, adj_masks, n)
                
                # Delta for c_new (addition):
                # We need to know how many C4s would be created.
                # This is the number of (x,y) in E_c_new such that x in N_c_new(j) \ {i}, y in N_c_new(i) \ {j}.
                # Since (i,j) is not in c_new, N_c_new(j) does not contain i, and N_c_new(i) does not contain j.
                # So we can just call count_c4s_edge_mask on c_new?
                # Wait, count_c4s_edge_mask checks if (i,j) is in the mask? No, it just uses the masks.
                # If (i,j) is not in the mask, the function still calculates the number of paths i-x-y-j.
                # These paths, combined with edge (i,j), form C4s.
                # So yes, calling count_c4s_edge_mask(c_new, i, j, adj_masks, n) gives the number of C4s created.
                
                contrib_new = count_c4s_edge_mask(c_new, i, j, adj_masks, n)
                
                delta_E = -contrib_old + contrib_new
                
                # Acceptance
                if delta_E < 0 or (T > 0 and random.random() < pow(2.718, -delta_E / T)):
                    # Apply
                    colors[idx_ij] = c_new
                    
                    # Update masks
                    adj_masks[c_old][i] &= ~(1 << j)
                    adj_masks[c_old][j] &= ~(1 << i)
                    adj_masks[c_new][i] |= (1 << j)
                    adj_masks[c_new][j] |= (1 << i)
                    
                    energy += delta_E
                    
                    if energy < best_energy:
                        best_energy = energy
                        best_colors = colors.copy()
                        if energy == 0:
                            break
                            
                T *= cooling
                
            if best_energy == 0:
                break
                
            # Perturb if not found
            # Randomly recolor 10% of edges
            for _ in range(n * n // 10):
                idx = random.randint(0, len(colors) - 1)
                colors[idx] = random.randint(0, 4)
            # Rebuild masks for next SA run? 
            # Actually, the SA loop above uses `adj_masks` which is updated in place.
            # If we perturb `colors`, we must rebuild `adj_masks` before the next SA run.
            # The current structure rebuilds masks at the start of each ILS iteration.
            
        if best_energy == 0:
            return {'n': n, 'colors': best_colors}
            
    # Fallback
    if best_colors_found:
         return {'n': best_n_found, 'colors': best_colors_found}
         
    # Return smallest valid if found? 
    # The loop returns on success. If it finishes without success, return a small valid one.
    # We can try n=11 quickly?
    # For now, return n=11 with a simple valid coloring (e.g. all 0s? No, K_11 has C4s. 
    # K_5 is C4-free? No, K_5 has C4s. 
    # A 5-cycle C_5 is C4-free. 
    # For n=5, color edges of C_5 with 0, rest with 1,2,3,4? 
    # Actually, just return n=1 with no colors? 
    # The problem asks for 1 <= n <= 29.
    # A valid coloring for n=1 is trivial.
    return {'n': 1, 'colors': []}
