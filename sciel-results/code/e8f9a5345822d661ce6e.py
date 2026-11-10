"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n = 20
    num_pairs = n * (n - 1) // 2
    
    # Precompute pair indices
    pair_to_idx = {}
    pair_list = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            pair_to_idx[(i, j)] = idx
            pair_to_idx[(j, i)] = idx
            pair_list.append((i, j))
            idx += 1
    
    # Precompute for each vertex, which pairs it's in
    vertex_pairs = [[] for _ in range(n)]
    for idx, (i, j) in enumerate(pair_list):
        vertex_pairs[i].append(idx)
        vertex_pairs[j].append(idx)
    
    # Precompute: for each pair (u,v), the list of vertices w such that
    # adding edge (u,v) would create a K2,3 if w is a common neighbor of both u and v
    # Actually, we need: for each potential edge (u,v), the set of "conflict pairs"
    # that would be violated if (u,v) is added.
    
    # For K2,3-free: no two vertices share 3+ common neighbors.
    # If we add edge (u,v), we need to check:
    # 1. u and v don't already share 3+ common neighbors
    # 2. For each w in N(v), u and w don't share 2+ common neighbors (since v would become a common neighbor)
    # 3. For each w in N(u), v and w don't share 2+ common neighbors
    
    # Use adjacency matrix representation with NumPy for vectorized operations
    # adj_matrix[i,j] = 1 if edge exists
    
    def popcount_mask(x):
        return bin(x).count('1')
    
    def can_add_fast(adj_mask, u, v):
        """Fast feasibility check using bitmask representation."""
        if (adj_mask[u] >> v) & 1:
            return False
        
        # Check pair (u,v): common neighbors <= 2
        common_uv = adj_mask[u] & adj_mask[v]
        if popcount_mask(common_uv) > 2:
            return False
        
        # Check pairs (u, w) for w in N(v)
        nv = adj_mask[v]
        w = 0
        while w < n:
            if (nv >> w) & 1 and w != u:
                common_uw = adj_mask[u] & adj_mask[w]
                if popcount_mask(common_uw) > 1:
                    return False
            w += 1
        
        # Check pairs (v, w) for w in N(u)
        nu = adj_mask[u]
        w = 0
        while w < n:
            if (nu >> w) & 1 and w != v:
                common_vw = adj_mask[v] & adj_mask[w]
                if popcount_mask(common_vw) > 1:
                    return False
            w += 1
        
        return True
    
    def get_edge_count_mask(adj_mask):
        count = 0
        for i in range(n):
            count += popcount_mask(adj_mask[i])
        return count // 2
    
    def is_valid_mask(adj_mask):
        for i in range(n):
            for j in range(i + 1, n):
                if popcount_mask(adj_mask[i] & adj_mask[j]) > 2:
                    return False
        return True
    
    # Multiple restarts with different random seeds
    best_edges_global = 0
    best_adj_global = None
    
    num_restarts = 5
    time_per_restart = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() >= deadline - 5:
            break
        
        # Initialize with a random graph
        adj_mask = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        
        # Ensure initial state is valid
        if not is_valid_mask(adj_mask):
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj_mask[i] >> j) & 1 and not is_valid_mask(adj_mask):
                        adj_mask[i] &= ~(1 << j)
                        adj_mask[j] &= ~(1 << i)
        
        current_edges = get_edge_count_mask(adj_mask)
        best_adj = adj_mask[:]
        best_edges = current_edges
        
        # Simulated annealing with QUBO-inspired energy
        T = 5.0
        cooling = 0.9995
        min_T = 0.05
        
        # Precompute all non-edges for faster sampling
        def get_non_edges(adj_mask):
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not ((adj_mask[i] >> j) & 1):
                        non_edges.append((i, j))
            return non_edges
        
        def get_edges(adj_mask):
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj_mask[i] >> j) & 1:
                        edges.append((i, j))
            return edges
        
        # Main SA loop
        while time.time() < deadline - 5:
            # Try adding a random edge
            if random.random() < 0.75:
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if not ((adj_mask[u] >> v) & 1):
                    if can_add_fast(adj_mask, u, v):
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                        new_edges = current_edges + 1
                        if new_edges > best_edges:
                            best_edges = new_edges
                            best_adj = adj_mask[:]
                        current_edges = new_edges
                    else:
                        # Try swap: remove a random edge, then try to add this one
                        existing = get_edges(adj_mask)
                        if existing:
                            rem_u, rem_v = random.choice(existing)
                            # Temporarily remove
                            adj_mask[rem_u] &= ~(1 << rem_v)
                            adj_mask[rem_v] &= ~(1 << rem_u)
                            
                            if can_add_fast(adj_mask, u, v):
                                adj_mask[u] |= (1 << v)
                                adj_mask[v] |= (1 << u)
                                # Accept with probability based on temperature
                                if random.random() < min(1.0, T / 5.0):
                                    pass  # Keep the swap
                                else:
                                    # Revert
                                    adj_mask[u] &= ~(1 << v)
                                    adj_mask[v] &= ~(1 << u)
                                    adj_mask[rem_u] |= (1 << rem_v)
                                    adj_mask[rem_v] |= (1 << rem_u)
                            else:
                                # Revert
                                adj_mask[rem_u] |= (1 << rem_v)
                                adj_mask[rem_v] |= (1 << rem_u)
            else:
                # Try swapping: remove one edge, add another
                existing = get_edges(adj_mask)
                if not existing:
                    continue
                
                rem_u, rem_v = random.choice(existing)
                
                # Try to find an edge to add
                added = False
                for _ in range(15):
                    add_u = random.randint(0, n - 1)
                    add_v = random.randint(0, n - 1)
                    if add_u == add_v:
                        continue
                    if (add_u, add_v) == (rem_u, rem_v) or (add_u, add_v) == (rem_v, rem_u):
                        continue
                    if (adj_mask[add_u] >> add_v) & 1:
                        continue
                    
                    # Temporarily remove the edge
                    adj_mask[rem_u] &= ~(1 << rem_v)
                    adj_mask[rem_v] &= ~(1 << rem_u)
                    
                    if can_add_fast(adj_mask, add_u, add_v):
                        adj_mask[add_u] |= (1 << add_v)
                        adj_mask[add_v] |= (1 << add_u)
                        # Accept with some probability
                        if random.random() < min(1.0, T / 3.0):
                            added = True
                            break
                        else:
                            # Revert
                            adj_mask[add_u] &= ~(1 << add_v)
                            adj_mask[add_v] &= ~(1 << add_u)
                            adj_mask[rem_u] |= (1 << rem_v)
                            adj_mask[rem_v] |= (1 << rem_u)
                            added = False
                            break
                    else:
                        # Revert
                        adj_mask[rem_u] |= (1 << rem_v)
                        adj_mask[rem_v] |= (1 << rem_u)
                        added = False
                        break
                
                if not added:
                    # Ensure the removed edge is back
                    if not ((adj_mask[rem_u] >> rem_v) & 1):
                        adj_mask[rem_u] |= (1 << rem_v)
                        adj_mask[rem_v] |= (1 << rem_u)
            
            # Cooling
            T *= cooling
            if T < min_T:
                T = min_T
        
        # Update global best
        if best_edges > best_edges_global:
            best_edges_global = best_edges
            best_adj_global = best_adj[:]
    
    # If no valid graph found, return empty
    if best_adj_global is None:
        return []
    
    # Extract edges from best_adj_global
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj_global[i] >> j) & 1:
                edges.append([i, j])
    
    return edges
# EVOLVE-BLOCK-END
