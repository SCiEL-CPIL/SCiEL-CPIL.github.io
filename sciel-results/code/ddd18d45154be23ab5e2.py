"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _creates_c6_bitset(adj_bits, n, u, v):
    """Fast check if adding edge (u,v) creates a C6 using bitsets.
    
    A C6 is formed if there exists a simple path of length 5 from u to v.
    Path: u - a - b - c - d - v
    
    We use bitset operations for speed.
    """
    # Get bitsets for neighbors
    nu = adj_bits[u]
    nv = adj_bits[v]
    
    # For each a in N(u) \ {v}:
    #   For each b in N(a) \ {u, v, a}:
    #     For each c in N(b) \ {u, v, a, b}:
    #       Check if N(c) \ {u, v, a, b, c} intersects N(v)
    
    mask_uv = ~(1 << u) & ~(1 << v)
    a_set = nu & mask_uv
    
    while a_set:
        a = (a_set & (-a_set)).bit_length() - 1
        a_set &= a_set - 1
        
        na = adj_bits[a]
        mask_uva = mask_uv & ~(1 << a)
        b_set = na & mask_uva
        
        while b_set:
            b = (b_set & (-b_set)).bit_length() - 1
            b_set &= b_set - 1
            
            nb = adj_bits[b]
            mask_uvab = mask_uva & ~(1 << b)
            c_set = nb & mask_uvab
            
            while c_set:
                c = (c_set & (-c_set)).bit_length() - 1
                c_set &= c_set - 1
                
                nc = adj_bits[c]
                mask_uvabc = mask_uvab & ~(1 << c)
                d_set = nc & mask_uvabc
                
                # Check if any d in d_set is in N(v)
                if d_set & nv:
                    return True
    
    return False

def _creates_c6_fast(adj_bits, n, u, v):
    """Even faster C6 check using precomputed 2-step neighborhoods.
    
    A C6 u-a-b-c-d-v exists iff there's a path of length 5 from u to v.
    We can check: for each a in N(u), for each c in N2(a) \ {u,v,a}, 
    check if N(c) intersects N(v) with appropriate exclusions.
    """
    nu = adj_bits[u]
    nv = adj_bits[v]
    
    mask_uv = ~(1 << u) & ~(1 << v)
    a_set = nu & mask_uv
    
    while a_set:
        a = (a_set & (-a_set)).bit_length() - 1
        a_set &= a_set - 1
        
        na = adj_bits[a]
        mask_uva = mask_uv & ~(1 << a)
        b_set = na & mask_uva
        
        while b_set:
            b = (b_set & (-b_set)).bit_length() - 1
            b_set &= b_set - 1
            
            nb = adj_bits[b]
            mask_uvab = mask_uva & ~(1 << b)
            c_set = nb & mask_uvab
            
            while c_set:
                c = (c_set & (-c_set)).bit_length() - 1
                c_set &= c_set - 1
                
                nc = adj_bits[c]
                mask_uvabc = mask_uvab & ~(1 << c)
                
                # Check if N(c) \ {u,v,a,b,c} intersects N(v)
                if (nc & mask_uvabc) & nv:
                    return True
    
    return False

def _greedy_build(n, rng, deadline):
    """Build a C6-free graph greedily with a given seed."""
    adj = [set() for _ in range(n)]
    adj_bits = [0] * n
    edges = set()
    
    # Start with 8 disjoint K5s for a good base
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = 5 * k + a, 5 * k + b
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
    
    # Generate candidate edges
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in edges:
                candidates.append((u, v))
    
    # Use iterative improvement: repeatedly try to add edges
    # Sort by degree sum for initial ordering
    candidates.sort(key=lambda uv: len(adj[uv[0]]) + len(adj[uv[1]]))
    
    # Multiple passes to allow for changes
    progress = True
    while progress and time.time() < deadline:
        progress = False
        # Re-sort candidates by current degree sum
        candidates.sort(key=lambda uv: len(adj[uv[0]]) + len(adj[uv[1]]))
        
        for u, v in candidates:
            if time.time() > deadline:
                break
            if (u, v) in edges:
                continue
            if not _creates_c6_fast(adj_bits, n, u, v):
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                progress = True
    
    return edges, adj, adj_bits

def _local_search(edges, adj, adj_bits, n, deadline, rng):
    """Local search: try swap moves (remove one edge, add another)."""
    current_count = len(edges)
    best_count = current_count
    best_edges = edges.copy()
    best_adj = [s.copy() for s in adj]
    best_adj_bits = adj_bits[:]
    
    no_improve = 0
    max_no_improve = 300
    
    # Build list of all non-edges for quick access
    all_non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in edges]
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a random edge to remove
        if not edges:
            break
        removed = rng.choice(tuple(edges))
        ru, rv = removed
        
        # Try to find a replacement edge from non-edges
        added = None
        # Try random non-edges, biased toward low-degree vertices
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
            if not _creates_c6_fast(adj_bits, n, u, v):
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
        adj_bits[ru] &= ~(1 << rv)
        adj_bits[rv] &= ~(1 << ru)
        
        edges.add(added)
        adj[au].add(av)
        adj[av].add(au)
        adj_bits[au] |= (1 << av)
        adj_bits[av] |= (1 << au)
        
        new_count = len(edges)
        
        if new_count > best_count:
            best_count = new_count
            best_edges = edges.copy()
            best_adj = [s.copy() for s in adj]
            best_adj_bits = adj_bits[:]
            no_improve = 0
        else:
            # Revert
            edges.discard(added)
            adj[au].discard(av)
            adj[av].discard(au)
            adj_bits[au] &= ~(1 << av)
            adj_bits[av] &= ~(1 << au)
            edges.add(removed)
            adj[ru].add(rv)
            adj[rv].add(ru)
            adj_bits[ru] |= (1 << rv)
            adj_bits[rv] |= (1 << ru)
            no_improve += 1
    
    return best_edges, best_adj, best_adj_bits, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    n = 40
    best_overall = 0
    best_edges_overall = set()
    
    # Multi-start local search
    num_starts = 12
    time_per_start = (deadline - start_time) / num_starts
    
    for s in range(num_starts):
        if time.time() > deadline:
            break
        
        seed_start = 42 + s * 1000
        rng = random.Random(seed_start)
        
        # Greedy build with limited time
        greedy_deadline = min(deadline - 5, start_time + time_per_start * 0.35)
        edges, adj, adj_bits = _greedy_build(n, rng, greedy_deadline)
        
        if len(edges) > best_overall:
            best_overall = len(edges)
            best_edges_overall = edges.copy()
        
        # Local search with remaining time
        ls_deadline = min(deadline - 2, start_time + time_per_start * (s + 1))
        edges, adj, adj_bits, count = _local_search(edges, adj, adj_bits, n, ls_deadline, rng)
        
        if count > best_overall:
            best_overall = count
            best_edges_overall = edges.copy()
    
    # Final verification: make sure no C6 exists
    final_adj = [set() for _ in range(n)]
    final_adj_bits = [0] * n
    for u, v in best_edges_overall:
        final_adj[u].add(v)
        final_adj[v].add(u)
        final_adj_bits[u] |= (1 << v)
        final_adj_bits[v] |= (1 << u)
    
    # Verify C6-free by checking all edges
    has_c6 = False
    for u in range(n):
        if has_c6:
            break
        for v in final_adj[u]:
            if v <= u:
                continue
            # Temporarily remove edge and check
            final_adj[u].discard(v)
            final_adj[v].discard(u)
            final_adj_bits[u] &= ~(1 << v)
            final_adj_bits[v] &= ~(1 << u)
            if _creates_c6_fast(final_adj_bits, n, u, v):
                has_c6 = True
            final_adj[u].add(v)
            final_adj[v].add(u)
            final_adj_bits[u] |= (1 << v)
            final_adj_bits[v] |= (1 << u)
    
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
