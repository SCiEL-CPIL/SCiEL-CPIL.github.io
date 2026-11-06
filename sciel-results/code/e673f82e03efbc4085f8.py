import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try to find larger n using fast local search
    for n in range(12, 27):
        if time.time() > deadline - 5.0:
            break
        
        # Initialize
        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        # Precompute edge map
        edge_map = {}
        inv_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[(i, j)] = idx
                inv_map[idx] = (i, j)
                idx += 1
        
        # Initialize adj_mask
        adj_mask = [[0] * n for _ in range(5)]
        for e_idx in range(num_edges):
            i, j = inv_map[e_idx]
            c = colors[e_idx]
            adj_mask[c][i] |= (1 << j)
            adj_mask[c][j] |= (1 << i)
            
        # Calculate initial violations
        def calc_violations(adj_mask, n):
            v = 0
            for c in range(5):
                for u in range(n):
                    for vrt in range(u + 1, n):
                        common = adj_mask[c][u] & adj_mask[c][vrt]
                        cnt = common.bit_count()
                        if cnt > 1:
                            v += cnt - 1
            return v
            
        current_violations = calc_violations(adj_mask, n)
        
        # Local Search
        temp = 1.0
        cooling = 0.9995
        max_iters = 200000 # High number, but fast update
        
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break
                
            if current_violations == 0:
                best_n = n
                best_colors = colors[:]
                break
                
            e_idx = random.randint(0, num_edges - 1)
            i, j = inv_map[e_idx]
            old_color = colors[e_idx]
            new_color = random.randint(0, 4)
            if new_color == old_color:
                continue
                
            # Calculate delta violations
            delta = 0
            
            # Remove from old_color
            if old_color != -1:
                c = old_color
                # Affected pairs: (i, w) for w in N_c(j), (j, w) for w in N_c(i)
                mask_j = adj_mask[c][j]
                temp_mask = mask_j
                while temp_mask:
                    lsb = temp_mask & -temp_mask
                    w = lsb.bit_length() - 1
                    temp_mask ^= lsb
                    if w == i: continue
                    
                    # Pair (i, w)
                    n_old = (adj_mask[c][i] & adj_mask[c][w]).bit_count()
                    n_new = n_old - 1
                    delta += max(0, n_new - 1) - max(0, n_old - 1)
                    
                    # Pair (j, w)
                    if (adj_mask[c][w] >> i) & 1:
                        n_old_jw = (adj_mask[c][j] & adj_mask[c][w]).bit_count()
                        n_new_jw = n_old_jw - 1
                        delta += max(0, n_new_jw - 1) - max(0, n_old_jw - 1)
                
                # Update masks
                adj_mask[c][i] &= ~(1 << j)
                adj_mask[c][j] &= ~(1 << i)
            
            # Add to new_color
            if new_color != -1:
                c = new_color
                mask_j = adj_mask[c][j]
                temp_mask = mask_j
                while temp_mask:
                    lsb = temp_mask & -temp_mask
                    w = lsb.bit_length() - 1
                    temp_mask ^= lsb
                    if w == i: continue
                    
                    # Pair (i, w)
                    n_old = (adj_mask[c][i] & adj_mask[c][w]).bit_count()
                    n_new = n_old + 1
                    delta += max(0, n_new - 1) - max(0, n_old - 1)
                    
                    # Pair (j, w)
                    if (adj_mask[c][w] >> i) & 1:
                        n_old_jw = (adj_mask[c][j] & adj_mask[c][w]).bit_count()
                        n_new_jw = n_old_jw + 1
                        delta += max(0, n_new_jw - 1) - max(0, n_old_jw - 1)
                
                # Update masks
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
            
            new_violations = current_violations + delta
            
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_violations = new_violations
                colors[e_idx] = new_color
            else:
                # Revert
                if new_color != -1:
                    c = new_color
                    adj_mask[c][i] &= ~(1 << j)
                    adj_mask[c][j] &= ~(1 << i)
                
                if old_color != -1:
                    c = old_color
                    adj_mask[c][i] |= (1 << j)
                    adj_mask[c][j] |= (1 << i)
                
            temp *= cooling
        
        if current_violations != 0:
            # If we failed to find a solution for this n, larger n is likely harder
            # But we might have found a good solution for a previous n?
            # We only update best_n if we find 0.
            # If we don't find 0, we keep the previous best.
            # Should we break?
            # If n=12 fails, n=13 is harder.
            break

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
