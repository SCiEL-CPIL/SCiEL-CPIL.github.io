"""Construct 5-colourings of K_n without monochromatic C4, maximising n."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110.0

    def full_check(n, colors):
        """Verify no colour class has a C4."""
        adj = [[0] * n for _ in range(5)] # Bitmasks
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    # Count common neighbors
                    # common = (adj[c][i] & adj[c][j]).bit_count()
                    if (adj[c][i] & adj[c][j]).bit_count() >= 2:
                        return False
        return True

    def local_search(n, max_time=10.0):
        """
        Find a valid 5-coloring of K_n using simulated annealing.
        Uses bitmask adjacency for fast updates.
        """
        total_edges = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        def init_random():
            colors = [random.randint(0, 4) for _ in range(total_edges)]
            adj = [[0] * n for _ in range(5)]
            common = [[[0]*n for _ in range(n)] for _ in range(5)]
            
            for idx, (i, j) in enumerate(edges):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                
            for c in range(5):
                for i in range(n):
                    for j in range(i+1, n):
                        common[c][i][j] = (adj[c][i] & adj[c][j]).bit_count()
                        common[c][j][i] = common[c][i][j]
                        
            return colors, adj, common

        def calc_cost(common):
            cost = 0
            for c in range(5):
                for i in range(n):
                    for j in range(i+1, n):
                        k = common[c][i][j]
                        if k >= 2:
                            cost += k * (k - 1) // 2
            return cost

        def cost_delta(k_old, k_new):
            def c_val(k):
                return k * (k - 1) // 2 if k >= 2 else 0
            return c_val(k_new) - c_val(k_old)

        def flip_edge(idx, new_color):
            nonlocal current_cost
            i, j = edges[idx]
            old_color = colors[idx]
            if old_color == new_color:
                return 0
            
            delta_cost = 0
            oc = old_color
            nc = new_color
            
            # 1. Remove (i, j) from oc
            # Neighbors of j in oc: K_j
            # For each k in K_j, common[oc][i][k] decreases by 1
            # Neighbors of i in oc: K_i
            # For each k in K_i, common[oc][j][k] decreases by 1
            
            # Get neighbors lists
            # We can iterate over bits in adj[oc][j]
            mask_j = adj[oc][j]
            mask_i = adj[oc][i]
            
            # Iterate k in mask_j
            m = mask_j
            while m:
                lsb = m & -m
                k = lsb.bit_length() - 1
                m ^= lsb
                if k == i: continue
                old_val = common[oc][i][k]
                new_val = old_val - 1
                delta_cost += cost_delta(old_val, new_val)
                common[oc][i][k] = new_val
                common[oc][k][i] = new_val
                
            # Iterate k in mask_i
            m = mask_i
            while m:
                lsb = m & -m
                k = lsb.bit_length() - 1
                m ^= lsb
                if k == j: continue
                old_val = common[oc][j][k]
                new_val = old_val - 1
                delta_cost += cost_delta(old_val, new_val)
                common[oc][j][k] = new_val
                common[oc][k][j] = new_val
                
            # Update adj
            adj[oc][i] &= ~(1 << j)
            adj[oc][j] &= ~(1 << i)
            
            # 2. Add (i, j) to nc
            mask_j_nc = adj[nc][j]
            mask_i_nc = adj[nc][i]
            
            # Iterate k in mask_j_nc
            m = mask_j_nc
            while m:
                lsb = m & -m
                k = lsb.bit_length() - 1
                m ^= lsb
                if k == i: continue
                old_val = common[nc][i][k]
                new_val = old_val + 1
                delta_cost += cost_delta(old_val, new_val)
                common[nc][i][k] = new_val
                common[nc][k][i] = new_val
                
            # Iterate k in mask_i_nc
            m = mask_i_nc
            while m:
                lsb = m & -m
                k = lsb.bit_length() - 1
                m ^= lsb
                if k == j: continue
                old_val = common[nc][j][k]
                new_val = old_val + 1
                delta_cost += cost_delta(old_val, new_val)
                common[nc][j][k] = new_val
                common[nc][k][j] = new_val
                
            # Update adj
            adj[nc][i] |= (1 << j)
            adj[nc][j] |= (1 << i)
            
            colors[idx] = new_color
            return delta_cost

        colors, adj, common = init_random()
        current_cost = calc_cost(common)
        best_cost = current_cost
        best_colors = colors[:]
        
        temperature = 1.0
        cooling = 0.999
        min_temp = 0.01
        
        while time.time() < start + max_time and temperature > min_temp:
            idx = random.randint(0, total_edges - 1)
            i, j = edges[idx]
            current_color = colors[idx]
            new_color = random.randint(0, 4)
            if new_color == current_color:
                continue
            
            delta = flip_edge(idx, new_color)
            
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                if current_cost + delta < best_cost:
                    best_cost = current_cost + delta
                    best_colors = colors[:]
                current_cost += delta
            else:
                # Undo
                flip_edge(idx, current_color)
                # Note: flip_edge returns delta, but we don't use it here.
                # We must ensure state is restored.
                # Since flip_edge modifies state, calling it again with old color restores it.
                # However, we need to be careful with current_cost.
                # We didn't update current_cost in the reject branch, so it's fine.
            
            temperature *= cooling
            
        if best_cost == 0:
            return best_colors
        return None

    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try n=26 first with multiple restarts
    # Then 25, 24...
    
    for n_try in [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12]:
        if time.time() > deadline - 5:
            break
            
        # Allocate more time to 26
        if n_try == 26:
            time_budget = 40.0
        elif n_try >= 24:
            time_budget = 15.0
        else:
            time_budget = 5.0
            
        # Multiple restarts for 26
        restarts = 3 if n_try == 26 else 1
        res = None
        for _ in range(restarts):
            if time.time() > start + max_time: # dummy, use deadline
                break
            if time.time() > deadline - 5:
                break
            # Adjust max_time per restart
            t_left = deadline - 5 - time.time()
            if t_left <= 0: break
            t_alloc = min(time_budget / restarts, t_left)
            res = local_search(n_try, max_time=t_alloc)
            if res is not None:
                break
                
        if res is not None:
            if full_check(n_try, res):
                best_n = n_try
                best_colors = res
                print(f"Found valid construction for n={n_try}")
                break
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
