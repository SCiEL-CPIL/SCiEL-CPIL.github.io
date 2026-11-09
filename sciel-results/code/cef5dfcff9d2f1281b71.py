import time
import random
import numpy as np
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    
    best_n = 17
    best_edges = None
    
    # Seed: Paley 17
    squares = {x * x % 17 for x in range(1, 17)}
    edges_17 = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    best_edges = edges_17

    def build_masks(n, adj):
        """Build bitmask neighborhoods from numpy adj matrix."""
        masks = [0] * n
        for u in range(n):
            for v in range(n):
                if adj[u, v]:
                    masks[u] |= (1 << v)
        return masks

    def has_k4(n, masks):
        """Check if any K4 exists. Early exit."""
        for u in range(n):
            for v in range(u + 1, n):
                if not (masks[u] & (1 << v)):
                    continue
                common = masks[u] & masks[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if masks[wi] & (1 << xi):
                            return True
        return False

    def has_ind6(n, masks):
        """Check if any independent 6-set exists. Early exit."""
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(masks[u] | (1 << u))
        
        for u in range(n):
            for v in range(u + 1, n):
                if not (comp_masks[u] & (1 << v)):
                    continue
                common = comp_masks[u] & comp_masks[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if comp_masks[wi] & (1 << xi):
                            common2 = comp_masks[u] & comp_masks[v] & comp_masks[wi] & comp_masks[xi]
                            common2 &= ~((1 << (xi + 1)) - 1)
                            y_mask = common2
                            while y_mask:
                                lsb_y = y_mask & -y_mask
                                yi = lsb_y.bit_length() - 1
                                y_mask ^= lsb_y
                                z_mask = common2 & ~((1 << (yi + 1)) - 1)
                                while z_mask:
                                    lsb_z = z_mask & -z_mask
                                    zi = lsb_z.bit_length() - 1
                                    z_mask ^= lsb_z
                                    if comp_masks[yi] & (1 << zi):
                                        return True
        return False

    def cost(n, adj):
        """Returns (k4_count, ind6_count)."""
        masks = build_masks(n, adj)
        # We just need to know if they exist for the search, 
        # but counting helps with annealing.
        # For speed, we just check existence in the loop, 
        # but for cost, we can count small numbers.
        k4 = 0
        ind6 = 0
        
        # Count K4s (capped)
        for u in range(n):
            for v in range(u + 1, n):
                if not (masks[u] & (1 << v)):
                    continue
                common = masks[u] & masks[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if masks[wi] & (1 << xi):
                            k4 += 1
                            if k4 > 100: return k4 + ind6
        
        # Count Ind6s (capped)
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(masks[u] | (1 << u))
            
        for u in range(n):
            for v in range(u + 1, n):
                if not (comp_masks[u] & (1 << v)):
                    continue
                common = comp_masks[u] & comp_masks[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if comp_masks[wi] & (1 << xi):
                            common2 = comp_masks[u] & comp_masks[v] & comp_masks[wi] & comp_masks[xi]
                            common2 &= ~((1 << (xi + 1)) - 1)
                            y_mask = common2
                            while y_mask:
                                lsb_y = y_mask & -y_mask
                                yi = lsb_y.bit_length() - 1
                                y_mask ^= lsb_y
                                z_mask = common2 & ~((1 << (yi + 1)) - 1)
                                while z_mask:
                                    lsb_z = z_mask & -z_mask
                                    zi = lsb_z.bit_length() - 1
                                    z_mask ^= lsb_z
                                    if comp_masks[yi] & (1 << zi):
                                        ind6 += 1
                                        if ind6 > 100: return k4 + ind6
                                        
        return k4 + ind6

    def is_valid(n, adj):
        """Check validity with early exit."""
        masks = build_masks(n, adj)
        if has_k4(n, masks):
            return False
        if has_ind6(n, masks):
            return False
        return True

    def try_find_graph(n, time_budget):
        """Local search to find valid graph on n vertices."""
        end_time = time.time() + time_budget
        
        best_cost = float('inf')
        best_adj = None
        
        # Try multiple restarts
        num_restarts = 20
        restart_budget = time_budget / num_restarts
        
        for restart in range(num_restarts):
            if time.time() >= end_time:
                break
            
            # Random initial graph, biased towards sparse
            # K4-free graphs are often sparse
            adj = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.3: # 30% edge probability
                        adj[i, j] = True
                        adj[j, i] = True
            
            current_cost = cost(n, adj)
            if current_cost == 0:
                edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
                return edges
            
            # Simulated Annealing
            T = 5.0
            T_min = 0.01
            cooling = 0.999
            no_improve = 0
            max_no_improve = 1000
            
            while time.time() < end_time and no_improve < max_no_improve:
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                
                old_state = adj[u, v]
                adj[u, v] = not old_state
                adj[v, u] = not old_state
                
                new_cost = cost(n, adj)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    no_improve = 0
                    if new_cost == 0:
                        edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
                        return edges
                else:
                    delta = new_cost - current_cost
                    T = max(T_min, T * cooling)
                    if random.random() < np.exp(-delta / T):
                        current_cost = new_cost
                        no_improve += 1
                    else:
                        adj[u, v] = old_state
                        adj[v, u] = old_state
                        no_improve += 1
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_adj = adj.copy()
        
        if best_adj is not None and is_valid(n, best_adj):
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if best_adj[i, j]]
            return edges
        
        return None

    # Try to find graphs from n=35 downward
    # Start with 35 as it's the known record
    for n in range(35, 16, -1):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        # Allocate more time to higher n
        if n >= 35:
            budget = min(remaining * 0.5, 30.0)
        elif n >= 30:
            budget = min(remaining * 0.3, 20.0)
        else:
            budget = min(remaining * 0.2, 10.0)
        
        edges = try_find_graph(n, budget)
        if edges is not None:
            best_n = n
            best_edges = edges
            break

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
