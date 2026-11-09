import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def build_masks(n, adj):
        masks = [0] * n
        for u in range(n):
            m = 0
            for v in range(n):
                if adj[u, v]:
                    m |= (1 << v)
            masks[u] = m
        return masks

    def count_k4_fast(n, masks):
        """Count K4s using bit operations."""
        k4_count = 0
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
                            k4_count += 1
        return k4_count

    def count_ind6_fast(n, masks):
        """Count independent 6-sets using bit operations."""
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(masks[u] | (1 << u))
        
        ind6_count = 0
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
                                        ind6_count += 1
        return ind6_count

    def total_violations(n, adj):
        masks = build_masks(n, adj)
        return count_k4_fast(n, masks) + count_ind6_fast(n, masks)

    def is_valid(n, adj):
        masks = build_masks(n, adj)
        if count_k4_fast(n, masks) > 0:
            return False
        if count_ind6_fast(n, masks) > 0:
            return False
        return True

    def edges_from_adj(n, adj):
        return [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]

    # Clebsch graph: 16 vertices, K4-free, alpha=5. Valid baseline.
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
    
    best_n = 16
    best_edges = clebsch_edges

    # Verify baseline
    adj_16 = np.zeros((16, 16), dtype=bool)
    for u, v in clebsch_edges:
        adj_16[u, v] = True
        adj_16[v, u] = True
    
    if not is_valid(16, adj_16):
        best_n = 5
        best_edges = []
    else:
        best_n = 16
        best_edges = [list(e) for e in clebsch_edges]

    # Local search: start from a random graph and optimize
    def local_search(n, max_time=30.0):
        """Local search to find a valid graph on n vertices."""
        start = time.time()
        deadline_local = start + max_time
        
        # Initialize with a random graph at moderate density
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.35:
                    adj[i, j] = True
                    adj[j, i] = True
        
        masks = build_masks(n, adj)
        current_k4 = count_k4_fast(n, masks)
        current_ind6 = count_ind6_fast(n, masks)
        current_cost = current_k4 + current_ind6
        
        if current_cost == 0:
            return edges_from_adj(n, adj)
        
        T = 2.0
        best_cost = current_cost
        best_adj = adj.copy()
        no_improve = 0
        
        while time.time() < deadline_local:
            # Pick a random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            old_val = adj[u, v]
            adj[u, v] = not old_val
            adj[v, u] = not old_val
            
            # Update masks
            if old_val:
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)
            else:
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
            
            new_k4 = count_k4_fast(n, masks)
            new_ind6 = count_ind6_fast(n, masks)
            new_cost = new_k4 + new_ind6
            
            if new_cost < current_cost:
                current_cost = new_cost
                no_improve = 0
                if new_cost == 0:
                    return edges_from_adj(n, adj)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj = adj.copy()
            else:
                delta = new_cost - current_cost
                T = max(0.05, T * 0.9995)
                if random.random() < np.exp(-delta / T):
                    current_cost = new_cost
                else:
                    # Revert
                    adj[u, v] = old_val
                    adj[v, u] = old_val
                    if old_val:
                        masks[u] |= (1 << v)
                        masks[v] |= (1 << u)
                    else:
                        masks[u] &= ~(1 << v)
                        masks[v] &= ~(1 << u)
                no_improve += 1
            
            if no_improve > 500:
                # Reset to best and restart temperature
                adj = best_adj.copy()
                masks = build_masks(n, adj)
                current_cost = best_cost
                T = 1.0
                no_improve = 0
        
        # Return best found
        if best_cost < current_cost:
            adj = best_adj.copy()
        if best_cost == 0:
            return edges_from_adj(n, adj)
        return None

    # Try local search for increasing n
    for target_n in range(17, 36):
        if time.time() > deadline - 5:
            break
        
        remaining = deadline - time.time()
        if remaining < 10.0:
            break
        
        # Try multiple restarts
        found = False
        for attempt in range(3):
            if time.time() > deadline - 5 or found:
                break
            
            result = local_search(target_n, max_time=min(remaining - 5, 20.0))
            if result is not None:
                found = True
                best_n = target_n
                best_edges = result
                break
        
        if found and best_n >= 35:
            break
    
    # If we didn't find anything better, try a different initialization strategy
    if best_n < 30 and time.time() < deadline - 15:
        for target_n in range(best_n + 1, min(best_n + 8, 36)):
            if time.time() > deadline - 5:
                break
            
            # Try starting from a structured graph
            adj = np.zeros((target_n, target_n), dtype=bool)
            
            # Create a random but structured initialization
            # Use a toroidal-like pattern
            for i in range(target_n):
                for j in range(target_n):
                    if i == j:
                        continue
                    diff = abs(i - j)
                    # Connect if difference is in a random set
                    if diff <= target_n // 4 and random.random() < 0.5:
                        adj[i, j] = True
            
            # Make symmetric
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if adj[i, j] != adj[j, i]:
                        adj[i, j] = adj[i, j] or adj[j, i]
                        adj[j, i] = adj[i, j]
            
            # Run local search from this start
            result = local_search(target_n, max_time=15.0)
            if result is not None:
                best_n = target_n
                best_edges = result
                break
    
    return {'n': best_n, 'edges': best_edges}
