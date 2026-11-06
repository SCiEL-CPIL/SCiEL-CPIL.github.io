import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill
    
    n = 16
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def is_triangle_free_bitmask(adj_mask):
        """Check triangle-free using bitmasks: no vertex u,v,w all mutually connected."""
        for u in range(n):
            for v in range(u+1, n):
                if adj_mask[u] & (1 << v):
                    if adj_mask[u] & adj_mask[v]:
                        return False
        return True
    
    def is_3_colorable_bitmask(adj_mask):
        """Check 3-colorability using backtracking with bitmask neighbor tracking."""
        # Precompute neighbor lists from bitmask
        neighbors = []
        for u in range(n):
            nbrs = []
            for v in range(n):
                if adj_mask[u] & (1 << v):
                    nbrs.append(v)
            neighbors.append(nbrs)
        
        # Order vertices by degree descending for better pruning
        order = sorted(range(n), key=lambda x: -len(neighbors[x]))
        
        color = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Compute forbidden colors
            forbidden = 0
            for u in neighbors[v]:
                cu = color[u]
                if cu != -1:
                    forbidden |= (1 << cu)
            for c in range(3):
                if not (forbidden & (1 << c)):
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)
    
    def edge_count(adj_mask):
        count = 0
        for i in range(n):
            count += bin(adj_mask[i]).count('1')
        return count // 2
    
    def groetzsch_graph():
        """Grötzsch graph: Mycielski of C5, 11 vertices, 20 edges, 4-chromatic triangle-free."""
        edges = [(0,1),(1,2),(2,3),(3,4),(4,0)]
        edges += [(5,1),(5,4),(6,2),(6,0),(7,3),(7,1),(8,4),(8,2),(9,0),(9,3)]
        edges += [(10,5),(10,6),(10,7),(10,8),(10,9)]
        return edges
    
    def build_adj_mask(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def greedy_add_edges(adj_mask, deadline):
        """Greedily add edges maintaining triangle-free and non-3-colorable."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try all non-edges in random order
            non_edges = []
            for u, v in all_pairs:
                if not (adj_mask[u] & (1 << v)):
                    non_edges.append((u, v))
            random.shuffle(non_edges)
            for u, v in non_edges:
                if time.time() >= deadline:
                    break
                # Check triangle-free: no common neighbor
                if adj_mask[u] & adj_mask[v]:
                    continue
                # Try adding edge
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                # Check 3-colorability
                if not is_3_colorable_bitmask(adj_mask):
                    improved = True
                else:
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
        return adj_mask
    
    def local_search_swap(adj_mask, deadline):
        """Try 1-1 swaps: remove one edge, add another, maintaining constraints."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if adj_mask[u] & (1 << v):
                        current_edges.append((u, v))
            random.shuffle(current_edges)
            
            for u, v in current_edges:
                if time.time() >= deadline:
                    return adj_mask, False
                # Remove edge (u,v)
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                
                found_swap = False
                for a, b in all_pairs:
                    if time.time() >= deadline:
                        break
                    if (a, b) == (u, v) or (a, b) == (v, u):
                        continue
                    if adj_mask[a] & (1 << b):
                        continue
                    # Check triangle-free
                    if adj_mask[a] & adj_mask[b]:
                        continue
                    adj_mask[a] |= (1 << b)
                    adj_mask[b] |= (1 << a)
                    if not is_3_colorable_bitmask(adj_mask):
                        improved = True
                        found_swap = True
                        break
                    adj_mask[a] &= ~(1 << b)
                    adj_mask[b] &= ~(1 << a)
                
                if found_swap:
                    return adj_mask, True
                
                # Restore edge (u,v)
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        return adj_mask, False
    
    # Start with Grötzsch graph on vertices 0-10, vertices 11-15 isolated
    initial_edges = groetzsch_graph()
    best_adj = build_adj_mask(initial_edges)
    best_count = edge_count(best_adj)
    
    # First, greedily add edges
    best_adj = greedy_add_edges(best_adj, deadline)
    best_count = edge_count(best_adj)
    
    # Then do local search with multiple restarts
    max_restarts = 30
    for restart in range(max_restarts):
        if time.time() >= deadline:
            break
        
        adj = best_adj[:]
        
        # Do local search
        adj, improved = local_search_swap(adj, deadline)
        
        if improved:
            new_count = edge_count(adj)
            if new_count > best_count:
                best_count = new_count
                best_adj = adj[:]
    
    # Final greedy pass
    best_adj = greedy_add_edges(best_adj, deadline)
    best_count = edge_count(best_adj)
    
    # Reconstruct edge list
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                final_edges.append([i, j])
    
    return final_edges
