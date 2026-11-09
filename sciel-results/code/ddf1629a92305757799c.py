"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _creates_c6_fast(adj_mask, n, u, v):
    """Fast check if adding edge (u,v) creates a C6 using bitmasks.
    
    adj_mask is a list of integers, where adj_mask[x] is a bitmask of neighbors of x.
    """
    mask_u = adj_mask[u]
    mask_v = adj_mask[v]
    
    # Exclude u and v from the intermediate nodes
    # We are looking for u-a-b-c-d-v
    # a, b, c, d must be distinct and not u or v
    
    # Iterate over neighbors a of u
    # a cannot be v
    mask_a_candidates = mask_u & ~(1 << v)
    
    while mask_a_candidates:
        # Get lowest set bit
        lsb = mask_a_candidates & -mask_a_candidates
        a = lsb.bit_length() - 1
        mask_a_candidates ^= lsb
        
        # b must be neighbor of a, not u, not v, not a
        mask_b_candidates = adj_mask[a] & ~(1 << u) & ~(1 << v) & ~(1 << a)
        
        while mask_b_candidates:
            lsb = mask_b_candidates & -mask_b_candidates
            b = lsb.bit_length() - 1
            mask_b_candidates ^= lsb
            
            # c must be neighbor of b, not u, not v, not a, not b
            mask_c_candidates = adj_mask[b] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b)
            
            while mask_c_candidates:
                lsb = mask_c_candidates & -mask_c_candidates
                c = lsb.bit_length() - 1
                mask_c_candidates ^= lsb
                
                # d must be neighbor of c, not u, not v, not a, not b, not c
                mask_d_candidates = adj_mask[c] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << c)
                
                # If any d in mask_d_candidates is a neighbor of v, we have a C6
                if mask_d_candidates & mask_v:
                    return True
                    
    return False

def _greedy_build(n, rng, deadline):
    """Build a C6-free graph greedily with a given seed."""
    adj = [set() for _ in range(n)]
    adj_mask = [0] * n
    edges = set()
    
    # Start with 8 disjoint K5s for a good base
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = 5 * k + a, 5 * k + b
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
    
    # Generate candidate edges and sort by a heuristic
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in edges:
                # Heuristic: prefer edges between vertices with lower degrees
                candidates.append((len(adj[u]) + len(adj[v]), rng.random(), u, v))
    
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    for _, _, u, v in candidates:
        if time.time() > deadline:
            break
        if not _creates_c6_fast(adj_mask, n, u, v):
            edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
    
    return edges, adj, adj_mask

def _local_search(edges, adj, adj_mask, n, deadline, rng):
    """Local search: try swap moves (remove one edge, add another)."""
    current_count = len(edges)
    best_count = current_count
    best_edges = edges.copy()
    best_adj = [s.copy() for s in adj]
    best_adj_mask = adj_mask.copy()
    
    no_improve = 0
    max_no_improve = 500 # Increased to allow more exploration with faster checks
    
    # Precompute all possible edges
    all_edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.add((u, v))
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a random edge to remove
        if not edges:
            break
        removed = rng.choice(list(edges))
        ru, rv = removed
        
        # Try to find a replacement edge
        added = None
        
        # Strategy: Try to add an edge that connects two low-degree vertices 
        # or bridges a gap. 
        # First, try random non-edges.
        for _ in range(100):
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (u, v) in edges:
                continue
            if (u, v) == removed:
                continue
            if not _creates_c6_fast(adj_mask, n, u, v):
                added = (u, v)
                break
        
        if added is None:
            # Try more targeted: look for non-edges where both endpoints have low degree
            # This is a heuristic to find "easy" additions
            candidates = []
            for u in range(n):
                deg_u = len(adj[u])
                if deg_u > 10: continue
                for v in adj[u]:
                    pass # This is inefficient, let's stick to random but more iterations if needed
            
            # Just increase random attempts slightly if first batch failed
            for _ in range(50):
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (u, v) in edges:
                    continue
                if (u, v) == removed:
                    continue
                if not _creates_c6_fast(adj_mask, n, u, v):
                    added = (u, v)
                    break
        
        if added is None:
            no_improve += 1
            continue
        
        au, av = added
        
        # Perform the swap
        edges.discard(removed)
        adj[ru].discard(rv)
        adj[rv].discard(ru)
        adj_mask[ru] &= ~(1 << rv)
        adj_mask[rv] &= ~(1 << ru)
        
        edges.add(added)
        adj[au].add(av)
        adj[av].add(au)
        adj_mask[au] |= (1 << av)
        adj_mask[av] |= (1 << au)
        
        new_count = len(edges)
        
        if new_count >= best_count:
            if new_count > best_count:
                best_count = new_count
                best_edges = edges.copy()
                best_adj = [s.copy() for s in adj]
                best_adj_mask = adj_mask.copy()
                no_improve = 0
        else:
            # Revert
            edges.discard(added)
            adj[au].discard(av)
            adj[av].discard(au)
            adj_mask[au] &= ~(1 << av)
            adj_mask[av] &= ~(1 << au)
            
            edges.add(removed)
            adj[ru].add(rv)
            adj[rv].add(ru)
            adj_mask[ru] |= (1 << rv)
            adj_mask[rv] |= (1 << ru)
            
            no_improve += 1
    
    return best_edges, best_adj, best_adj_mask, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n = 40
    best_overall = 0
    best_edges_overall = set()
    
    # Multi-start local search
    # We can afford more starts or longer searches per start due to faster C6 check
    num_starts = 15
    time_per_start = (deadline - start_time) / num_starts
    
    for s in range(num_starts):
        if time.time() > deadline:
            break
        
        seed_start = 42 + s * 1000
        rng = random.Random(seed_start)
        edges, adj, adj_mask = _greedy_build(n, rng, deadline - 10)
        
        if len(edges) > best_overall:
            best_overall = len(edges)
            best_edges_overall = edges.copy()
        
        # Local search with remaining time
        ls_deadline = min(deadline - 2, start_time + time_per_start * (s + 1))
        edges, adj, adj_mask, count = _local_search(edges, adj, adj_mask, n, ls_deadline, rng)
        
        if count > best_overall:
            best_overall = count
            best_edges_overall = edges.copy()
    
    # Final verification: make sure no C6 exists
    final_adj = [set() for _ in range(n)]
    final_adj_mask = [0] * n
    for u, v in best_edges_overall:
        final_adj[u].add(v)
        final_adj[v].add(u)
        final_adj_mask[u] |= (1 << v)
        final_adj_mask[v] |= (1 << u)
    
    # Verify C6-free by checking all potential 6-cycles
    has_c6 = False
    for u in range(n):
        if has_c6:
            break
        for v in final_adj[u]:
            if v <= u:
                continue
            # Temporarily remove edge and check
            final_adj_mask[u] &= ~(1 << v)
            final_adj_mask[v] &= ~(1 << u)
            if _creates_c6_fast(final_adj_mask, n, u, v):
                has_c6 = True
            final_adj_mask[u] |= (1 << v)
            final_adj_mask[v] |= (1 << u)
    
    if has_c6:
        # This shouldn't happen, but fallback to just the K5s
        fallback = set()
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u, v = 5 * k + a, 5 * k + b
                    fallback.add((u, v))
        return [list(e) for e in fallback]
    
    return [list(e) for e in best_edges_overall]

# EVOLVE-BLOCK-END
