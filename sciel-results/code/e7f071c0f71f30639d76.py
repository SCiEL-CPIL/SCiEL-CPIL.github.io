import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 20
    rng = random.Random(42)
    
    # Precompute all pairs and triples? 
    # Pairs: 190
    # Triples: 1140
    # We might not need precomputed triples if we iterate on the fly.
    
    def bit_count(x):
        return x.bit_count()
        
    def is_valid_add(adj, u, v):
        """
        Check if adding edge (u,v) to a valid graph adj keeps it valid.
        Assumes (u,v) is not currently an edge.
        """
        # Check triples containing u: (u, y, z)
        # Count increases if v is in N(y) and N(z).
        # Need old_count(u, y, z) <= 1.
        # Old count is bit_count(adj[u] & adj[y] & adj[z])
        
        # Check triples containing v: (v, y, z)
        # Count increases if u is in N(y) and N(z).
        # Need old_count(v, y, z) <= 1.
        
        for y in range(n):
            if y == u or y == v:
                continue
            ny = adj[y]
            # Check if v is neighbor of y
            if (ny >> v) & 1:
                for z in range(y + 1, n):
                    if z == u or z == v:
                        continue
                    nz = adj[z]
                    # Check if v is neighbor of z
                    if (nz >> v) & 1:
                        # v is common neighbor of y and z.
                        # So adding (u,v) makes v a common neighbor of u,y,z?
                        # No.
                        # Common neighbors of (u,y,z) are N(u) & N(y) & N(z).
                        # N(u) gains v.
                        # So new intersection = (N(u) & N(y) & N(z)) | ({v} & N(y) & N(z)).
                        # Since v is in N(y) and N(z), the second term is {v}.
                        # So count increases by 1.
                        # We need new count < 3 => old count <= 1.
                        if bit_count(adj[u] & ny & nz) > 1:
                            return False
            
            # Check if u is neighbor of y
            if (ny >> u) & 1:
                for z in range(y + 1, n):
                    if z == u or z == v:
                        continue
                    nz = adj[z]
                    # Check if u is neighbor of z
                    if (nz >> u) & 1:
                        # u is common neighbor of y and z.
                        # Adding (u,v) makes u a common neighbor of v,y,z?
                        # Common neighbors of (v,y,z) are N(v) & N(y) & N(z).
                        # N(v) gains u.
                        # New intersection = (N(v) & N(y) & N(z)) | ({u} & N(y) & N(z)).
                        # Since u is in N(y) and N(z), second term is {u}.
                        # Count increases by 1.
                        # Need old count <= 1.
                        if bit_count(adj[v] & ny & nz) > 1:
                            return False
        
        return True

    def count_edges(adj):
        return sum(bit_count(adj[i]) for i in range(n)) // 2

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    best_edges = 0
    best_adj = None
    
    # Generate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    num_starts = 50
    per_start_time = 2.0
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 5:
            break
        
        # 1. Random Start
        adj = [0] * n
        for (u, v) in all_edges:
            if rng.random() < 0.5:
                add_edge(adj, u, v)
        
        # 2. Clean: Remove invalid edges
        # We don't have a fast "remove" check, so we just iterate and remove if invalid.
        # To check if current graph is valid, we need to check all triples.
        # But we can just try to remove edges until valid?
        # Simpler: Start with empty graph and add greedily?
        # Let's use Greedy Construction for initialization.
        
        adj = [0] * n
        rng.shuffle(all_edges)
        for (u, v) in all_edges:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
        
        cur_edges = count_edges(adj)
        
        # 3. Local Search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Phase 1: Add-only
            rng.shuffle(all_edges)
            for (u, v) in all_edges:
                if (adj[u] >> v) & 1:
                    continue
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    cur_edges += 1
                    improved = True
            
            # Phase 2: Swaps
            if time.time() > deadline - 5:
                break
            
            # Try to find a valid swap
            # Randomly try swaps
            for _ in range(500): # Limit swap attempts per phase
                if time.time() > deadline - 5:
                    break
                # Pick random edge to remove
                # We need to pick an existing edge.
                # Maintain list of existing edges? Or just pick random pair and check if exists.
                # Since n=20, 190 edges. Picking random pair is fast.
                e1_u, e1_v = rng.choice(all_edges)
                if not ((adj[e1_u] >> e1_v) & 1):
                    continue
                
                e2_u, e2_v = rng.choice(all_edges)
                if (e2_u, e2_v) == (e1_u, e1_v) or (e2_u, e2_v) == (e1_v, e1_u):
                    continue
                if (adj[e2_u] >> e2_v) & 1:
                    continue
                
                # Try swap
                remove_edge(adj, e1_u, e1_v)
                if is_valid_add(adj, e2_u, e2_v):
                    add_edge(adj, e2_u, e2_v)
                    improved = True
                    # cur_edges remains same
                else:
                    # Revert
                    add_edge(adj, e1_u, e1_v)
            
            if cur_edges > best_edges:
                best_edges = cur_edges
                best_adj = [a for a in adj]
                
            if not improved:
                break
                
    if best_adj is None:
        # Fallback: return empty or a known small graph
        return []
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
