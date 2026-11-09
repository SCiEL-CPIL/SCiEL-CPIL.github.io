import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    def build_masks(n, adj):
        """Build list of integer bitmasks for adjacency."""
        masks = [0] * n
        for u in range(n):
            m = 0
            for v in range(n):
                if adj[u, v]:
                    m |= (1 << v)
            masks[u] = m
        return masks

    def count_k4(n, masks):
        """Count number of K4s in the graph using bitmasks."""
        k4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (masks[u] & (1 << v)):
                    continue
                # Common neighbors of u and v, with index > v
                common = masks[u] & masks[v]
                common &= ~((1 << (v + 1)) - 1)
                
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    # Common neighbors of u, v, w with index > w
                    x_mask = common & masks[wi]
                    x_mask &= ~((1 << (wi + 1)) - 1)
                    k4_count += bin(x_mask).count('1')
        return k4_count

    def count_ind6(n, masks):
        """Count number of independent sets of size 6 using bitmasks on complement."""
        full_mask = (1 << n) - 1
        # comp_masks[u] has bits set for vertices NOT adjacent to u (and not u itself)
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(masks[u] | (1 << u))
        
        ind6_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (comp_masks[u] & (1 << v)):
                    continue
                # Common non-neighbors of u and v, with index > v
                common = comp_masks[u] & comp_masks[v]
                common &= ~((1 << (v + 1)) - 1)
                
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    # Common non-neighbors of u, v, w with index > w
                    x_mask = common & comp_masks[wi]
                    x_mask &= ~((1 << (wi + 1)) - 1)
                    
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        # Common non-neighbors of u, v, w, x with index > x
                        y_mask = x_mask & comp_masks[xi]
                        y_mask &= ~((1 << (xi + 1)) - 1)
                        
                        while y_mask:
                            lsb_y = y_mask & -y_mask
                            yi = lsb_y.bit_length() - 1
                            y_mask ^= lsb_y
                            # Common non-neighbors of u, v, w, x, y with index > y
                            z_mask = y_mask & comp_masks[yi]
                            z_mask &= ~((1 << (yi + 1)) - 1)
                            ind6_count += bin(z_mask).count('1')
        return ind6_count

    def total_violations(n, adj):
        masks = build_masks(n, adj)
        return count_k4(n, masks) + count_ind6(n, masks)

    def is_valid(n, adj):
        masks = build_masks(n, adj)
        if count_k4(n, masks) > 0:
            return False
        if count_ind6(n, masks) > 0:
            return False
        return True

    def edges_from_adj(n, adj):
        return [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]

    best_n = 16
    # Fallback: Clebsch graph (16 vertices, valid)
    clebsch_edges = [
        (0, 6), (0, 7), (0, 10), (0, 11),
        (1, 5), (1, 8), (1, 10), (1, 13),
        (2, 4), (2, 9), (2, 11), (2, 13),
        (3, 4), (3, 7), (3, 9), (3, 12),
        (5, 6), (5, 8), (5, 12), (5, 15),
        (6, 7), (6, 9), (6, 14), (6, 15),
        (7, 8), (7, 11), (7, 14), (7, 15),
        (8, 9), (8, 12), (8, 15),
        (9, 10), (9, 14),
        (10, 11), (10, 13), (10, 14),
        (11, 12), (11, 14), (11, 15),
        (12, 13), (12, 15),
        (13, 14), (13, 15)
    ]
    best_edges = clebsch_edges

    # Try to find larger valid graphs
    for target_n in [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]:
        if time.time() > deadline - 5.0:
            break
        
        # If we already have a better n, skip smaller ones unless we are desperate
        if target_n <= best_n:
            continue

        found = False
        # Budget for this n
        remaining = deadline - time.time()
        if target_n >= 35:
            budget = min(remaining * 0.5, 40.0)
        elif target_n >= 30:
            budget = min(remaining * 0.3, 20.0)
        else:
            budget = min(remaining * 0.2, 10.0)

        num_restarts = max(1, int(budget / 2.0))
        
        for restart in range(num_restarts):
            if time.time() > deadline - 2.0 or found:
                break
            
            # Initialize with random graph
            # Density around 0.5 is a good start for Ramsey graphs
            adj = np.zeros((target_n, target_n), dtype=bool)
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj[i, j] = True
                        adj[j, i] = True
            
            current_cost = total_violations(target_n, adj)
            
            if current_cost == 0:
                found = True
                best_n = target_n
                best_edges = edges_from_adj(target_n, adj)
                break
            
            # Simulated Annealing
            T = 5.0
            T_min = 0.1
            cooling = 0.99
            no_improve = 0
            max_steps = int(budget * 20) # Adjust based on budget
            
            for step in range(max_steps):
                if time.time() > deadline - 1.0:
                    break
                if no_improve > 500:
                    break
                
                # Random edge flip
                u = random.randint(0, target_n - 1)
                v = random.randint(0, target_n - 1)
                if u == v:
                    continue
                
                old = adj[u, v]
                adj[u, v] = not old
                adj[v, u] = not old
                
                new_cost = total_violations(target_n, adj)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    no_improve = 0
                    if new_cost == 0:
                        found = True
                        best_n = target_n
                        best_edges = edges_from_adj(target_n, adj)
                        break
                else:
                    delta = new_cost - current_cost
                    # Accept with probability exp(-delta/T)
                    if random.random() < np.exp(-delta / T):
                        current_cost = new_cost
                    else:
                        adj[u, v] = old
                        adj[v, u] = old
                    no_improve += 1
                
                T = max(T_min, T * cooling)

        if found and target_n == 35:
            break

    return {'n': best_n, 'edges': best_edges}
