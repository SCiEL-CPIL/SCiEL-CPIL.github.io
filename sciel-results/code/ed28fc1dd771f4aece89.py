import time
import random

def construct():
    n = 15
    start = time.time()
    deadline = start + 80  # safety margin
    
    # Bitmask-based adjacency
    # adj[i] is an integer bitmask of neighbors of vertex i
    
    def is_valid(adj, n):
        """Check if no simple 3-edge path has endpoints with equal degree."""
        deg = [adj[i].bit_count() for i in range(n)]
        for u in range(n):
            neighbors_u = adj[u]
            # Iterate over edges (u, v) where v > u
            v_mask = neighbors_u & ~((1 << (u + 1)) - 1)
            while v_mask:
                v_bit = v_mask & (-v_mask)
                v = v_bit.bit_length() - 1
                v_mask ^= v_bit
                # s in N(u) \ {v}, t in N(v) \ {u, s}, s != t
                s_mask = neighbors_u & ~v_bit
                while s_mask:
                    s_bit = s_mask & (-s_mask)
                    s = s_bit.bit_length() - 1
                    s_mask ^= s_bit
                    t_mask = adj[v] & ~(1 << u) & ~s_bit
                    while t_mask:
                        t_bit = t_mask & (-t_mask)
                        t = t_bit.bit_length() - 1
                        t_mask ^= t_bit
                        if deg[s] == deg[t]:
                            return False
        return True
    
    def try_add_edges(adj, deadline):
        """Greedy: try to add all missing edges that maintain validity."""
        added = True
        while added:
            if time.time() > deadline:
                break
            added = False
            for i in range(n):
                if time.time() > deadline:
                    return adj
                for j in range(i + 1, n):
                    if not (adj[i] >> j) & 1:
                        # Try adding edge (i, j)
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        if is_valid(adj, n):
                            added = True
                        else:
                            adj[i] &= ~(1 << j)
                            adj[j] &= ~(1 << i)
        return adj
    
    def edges_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2
    
    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges
    
    best_adj = None
    best_count = -1
    
    # Try all bipartite splits
    for split in range(2, n - 1):
        if time.time() > deadline - 5:
            break
        adj = [0] * n
        for i in range(split):
            for j in range(split, n):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        adj = try_add_edges(adj, deadline)
        cnt = edges_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj)
    
    # Local search: try removing edges and re-greedy-filling
    if time.time() < deadline - 10:
        # Multiple local search attempts
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            # Start from best or a random perturbation
            if attempt == 0 or random.random() < 0.5:
                adj = list(best_adj)
            else:
                # Random bipartite start
                split = random.randint(2, n - 2)
                adj = [0] * n
                for i in range(split):
                    for j in range(split, n):
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Perturbation: remove 1-3 random edges
            adj = list(adj)
            num_remove = random.randint(1, 3)
            for _ in range(num_remove):
                # Pick a random existing edge
                all_edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[i] >> j) & 1:
                            all_edges.append((i, j))
                if not all_edges:
                    break
                i, j = random.choice(all_edges)
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
            
            # Greedy fill
            adj = try_add_edges(adj, deadline)
            cnt = edges_count(adj)
            if cnt > best_count:
                best_count = cnt
                best_adj = list(adj)
    
    # Second round of local search from best
    if time.time() < deadline - 5:
        for attempt in range(30):
            if time.time() > deadline - 2:
                break
            adj = list(best_adj)
            # Remove 1-2 random edges
            num_remove = random.randint(1, 2)
            for _ in range(num_remove):
                all_edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[i] >> j) & 1:
                            all_edges.append((i, j))
                if not all_edges:
                    break
                i, j = random.choice(all_edges)
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
            
            adj = try_add_edges(adj, deadline)
            cnt = edges_count(adj)
            if cnt > best_count:
                best_count = cnt
                best_adj = list(adj)
    
    # Final: try adding edges one more time
    if time.time() < deadline - 1:
        best_adj = try_add_edges(list(best_adj), deadline)
        best_count = edges_count(best_adj)
    
    # Convert to proper types
    result = adj_to_edges(best_adj)
    return result
