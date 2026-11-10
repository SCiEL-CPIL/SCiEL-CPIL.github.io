import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 20
    
    # Best known solution so far
    best_edges = 0
    best_adj = [0] * n
    
    # Initialize with the seed solution (5 K4s) to have a valid starting point
    def init_seed():
        adj = [0] * n
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    u = 4 * k + a
                    v = 4 * k + b
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def count_edges(adj):
        return sum(x.bit_count() for x in adj) // 2

    def can_add(adj, u, v):
        # If already connected
        if (adj[u] >> v) & 1:
            return False
            
        # Check common neighbors of u and v
        if (adj[u] & adj[v]).bit_count() > 2:
            return False

        # Check effect on other pairs
        # For each w, if w is neighbor of u, check count(v, w) <= 1
        # if w is neighbor of v, check count(u, w) <= 1
        for w in range(n):
            if w == u or w == v:
                continue
            
            if (adj[u] >> w) & 1:
                # w is in N(u). u is in N(w).
                # Adding (u,v) adds u to N(v).
                # u becomes common neighbor of v and w if u in N(w).
                # u IS in N(w).
                # So count(v, w) increases by 1.
                # Need old count(v, w) <= 1.
                if (adj[v] & adj[w]).bit_count() > 1:
                    return False
            
            if (adj[v] >> w) & 1:
                # w is in N(v). v is in N(w).
                # Adding (u,v) adds v to N(u).
                # v becomes common neighbor of u and w if v in N(w).
                # v IS in N(w).
                # So count(u, w) increases by 1.
                # Need old count(u, w) <= 1.
                if (adj[u] & adj[w]).bit_count() > 1:
                    return False

        return True

    def saturate(adj):
        # Try to add edges until no more can be added
        changed = True
        while changed:
            changed = False
            # Get list of non-edges
            non_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1):
                        non_edges.append((u, v))
                
            if not non_edges:
                break
                
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if can_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    changed = True
        return adj

    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    # Initial Setup
    best_adj = init_seed()
    best_adj = saturate(best_adj)
    best_edges = count_edges(best_adj)

    # Local Search with Perturbation
    # We will iteratively:
    # 1. Take the best known graph
    # 2. Remove 1 random edge (perturbation)
    # 3. Re-saturate
    # 4. If better, update best
    # This allows escaping local optima where no single edge addition is possible.
    
    while time.time() < deadline:
        # Create candidate
        cand_adj = best_adj[:]
        
        # Perturbation: Remove 1 random edge
        # Find a random edge
        edges_list = []
        for u in range(n):
            for v in range(u + 1, n):
                if (cand_adj[u] >> v) & 1:
                    edges_list.append((u, v))
                        
        if edges_list:
            u, v = random.choice(edges_list)
            cand_adj[u] &= ~(1 << v)
            cand_adj[v] &= ~(1 << u)
        else:
            # If no edges, start fresh
            cand_adj = [0] * n

        # Saturate the candidate
        cand_adj = saturate(cand_adj)
        
        cand_edges = count_edges(cand_adj)
        
        if cand_edges > best_edges:
            best_edges = cand_edges
            best_adj = cand_adj[:]
            
        # Occasionally, try a completely fresh random start to escape global basins
        # This is expensive, so do it rarely
        if random.random() < 0.05:
            fresh_adj = [0] * n
            fresh_adj = saturate(fresh_adj)
            fresh_edges = count_edges(fresh_adj)
            if fresh_edges > best_edges:
                best_edges = fresh_edges
                best_adj = fresh_adj[:]

    return get_edges_from_adj(best_adj)
