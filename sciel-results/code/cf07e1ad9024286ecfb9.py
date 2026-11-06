import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    n = 26
    num_colors = 5
    
    # We will represent the coloring as a list of lists col[i][j]
    # And maintain adjacency bitmasks for each color: adj_mask[c][i] is an int where bit j is set if edge (i,j) is color c.
    
    def init_random():
        col = [[0]*n for _ in range(n)]
        adj_mask = [[0]*n for _ in range(num_colors)]
        for i in range(n):
            for j in range(i+1, n):
                c = random.randint(0, num_colors-1)
                col[i][j] = c
                col[j][i] = c
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
        return col, adj_mask

    def count_penalty(adj_mask):
        total = 0
        for c in range(num_colors):
            # For each pair u < v, count common neighbors in color c
            for u in range(n):
                for v in range(u+1, n):
                    common = adj_mask[c][u] & adj_mask[c][v]
                    k = bin(common).count('1')
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total

    def fast_solve():
        col, adj_mask = init_random()
        current_penalty = count_penalty(adj_mask)
        
        if current_penalty == 0:
            return col

        best_penalty = current_penalty
        best_col = [row[:] for row in col]
        
        # Local Search with Restarts
        max_restarts = 50
        for _ in range(max_restarts):
            if time.time() > deadline:
                break
            
            col, adj_mask = init_random()
            current_penalty = count_penalty(adj_mask)
            
            if current_penalty == 0:
                return col
            
            best_local_penalty = current_penalty
            best_local_col = [row[:] for row in col]
            
            # Simulated Annealing style loop, but very fast
            temp = 1.0
            no_improve = 0
            max_iter = 200000
            
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                    
                if current_penalty == 0:
                    return col
                
                # Pick random edge
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                    
                old_c = col[i][j]
                new_c = random.randint(0, num_colors-1)
                if new_c == old_c:
                    continue
                
                # Perform swap
                col[i][j] = new_c
                col[j][i] = new_c
                
                # Update masks
                adj_mask[old_c][i] &= ~(1 << j)
                adj_mask[old_c][j] &= ~(1 << i)
                adj_mask[new_c][i] |= (1 << j)
                adj_mask[new_c][j] |= (1 << i)
                
                # Recalculate penalty efficiently
                # Only pairs involving i or j change
                # Old penalty contribution of pairs (i,k) and (j,k) for all k != i,j
                # Plus pair (i,j) itself?
                # The formula sums over all pairs u<v.
                # Pairs affected: (i, k) for k!=i,j, (j, k) for k!=i,j, and (i, j).
                
                old_pen_contrib = 0
                new_pen_contrib = 0
                
                # We need to compare old and new states.
                # Since we already updated the masks, we can't easily get "old" contribution without reverting or storing old masks.
                # It is faster to just revert if the delta is positive.
                # To get delta:
                # 1. Calculate penalty of the *current* (new) state for affected pairs.
                # 2. Revert change.
                # 3. Calculate penalty of the *old* state for affected pairs.
                # 4. Delta = New - Old.
                # 5. If Delta < 0 or SA accept, keep change. Else revert.
                
                # Actually, calculating the penalty for the whole graph is O(n^2 * num_colors) ~ 5*676 = 3380 ops.
                # With 200k iterations, that's 676M ops. Might be slow in Python.
                # Let's optimize the "affected pairs" calculation.
                
                def calc_affected_penalty(i, j, adj_mask):
                    pen = 0
                    # Pairs (i, k)
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        for c in range(num_colors):
                            # Note: The color of edge (i,k) and (j,k) might be anything.
                            # We just look at the common neighbors in each color c.
                            # This definition of penalty depends on the graph structure, not edge colors of i,k directly, 
                            # but rather the sets N_c(i) and N_c(k).
                            common = adj_mask[c][i] & adj_mask[c][k]
                            cnt = bin(common).count('1')
                            if cnt >= 2:
                                pen += cnt * (cnt - 1) // 2
                        # Wait, the sum is over ALL colors c for the pair (i,k).
                        # The loop above does that.
                    
                    # Pairs (j, k)
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        for c in range(num_colors):
                            common = adj_mask[c][j] & adj_mask[c][k]
                            cnt = bin(common).count('1')
                            if cnt >= 2:
                                pen += cnt * (cnt - 1) // 2
                    
                    # Pair (i, j)
                    for c in range(num_colors):
                        common = adj_mask[c][i] & adj_mask[c][j]
                        cnt = bin(common).count('1')
                        if cnt >= 2:
                            pen += cnt * (cnt - 1) // 2
                            
                    return pen

                # This function is O(n * num_colors) for the affected part.
                # n=26, colors=5. 26*5 = 130 ops per pair set. 3 sets. ~400 ops.
                # Much better than 3380.
                
                new_pen = calc_affected_penalty(i, j, adj_mask)
                
                # Revert to calculate old_pen
                col[i][j] = old_c
                col[j][i] = old_c
                adj_mask[old_c][i] |= (1 << j)
                adj_mask[old_c][j] |= (1 << i)
                adj_mask[new_c][i] &= ~(1 << j)
                adj_mask[new_c][j] &= ~(1 << i)
                
                old_pen = calc_affected_penalty(i, j, adj_mask)
                
                delta = new_pen - old_pen
                
                if delta < 0 or (delta > 0 and random.random() < np.exp(-delta / temp)):
                    # Accept change
                    col[i][j] = new_c
                    col[j][i] = new_c
                    adj_mask[old_c][i] &= ~(1 << j)
                    adj_mask[old_c][j] &= ~(1 << i)
                    adj_mask[new_c][i] |= (1 << j)
                    adj_mask[new_c][j] |= (1 << i)
                    
                    current_penalty += delta
                    if current_penalty < best_local_penalty:
                        best_local_penalty = current_penalty
                        best_local_col = [row[:] for row in col]
                        if best_local_penalty < best_penalty:
                            best_penalty = best_local_penalty
                            best_col = [row[:] for row in col]
                    no_improve = 0
                else:
                    # Reject (already reverted)
                    no_improve += 1
                
                if no_improve > 1000:
                    break
                temp *= 0.999
                
            if best_local_penalty == 0:
                return best_local_col

        return best_col

    best_col = None
    # Run multiple times to ensure we find it
    while time.time() < deadline:
        sol = fast_solve()
        if sol is None:
            continue
        
        # Verify
        col, _ = init_random() # Re-init masks for verification? No, build from sol
        adj_mask = [[0]*n for _ in range(num_colors)]
        for i in range(n):
            for j in range(i+1, n):
                c = sol[i][j]
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
        
        pen = count_penalty(adj_mask)
        if pen == 0:
            best_col = sol
            break
        else:
            if best_col is None:
                best_col = sol

    if best_col is not None:
        res_colors = []
        for i in range(n):
            for j in range(i+1, n):
                res_colors.append(best_col[i][j])
        return {'n': n, 'colors': res_colors}
    
    # Fallback to n=11
    n_seed = 11
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    return {'n': n_seed, 'colors': colors_seed}
