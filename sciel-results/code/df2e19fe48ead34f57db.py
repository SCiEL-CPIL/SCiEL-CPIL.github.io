"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 115  # safety margin before 150s kill
    n = 40
    
    # Build adjacency as bitmasks for fast operations
    adj_mask = [0] * n
    
    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    def remove_edge(u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
    
    def degree(u):
        return bin(adj_mask[u]).count('1')
    
    def total_edges():
        return sum(degree(u) for u in range(n)) // 2
    
    def has_c6_fast(u, v):
        """Check if adding edge (u,v) creates a C6 using bitmasks.
        
        Cycle C6 has 6 vertices: u, a, b, c, d, v.
        Edges: (u,a), (a,b), (b,c), (c,d), (d,v), (v,u).
        We need a path of length 5 from u to v in G - (u,v).
        
        Vertices: u, a, b, c, d, v. All distinct.
        
        Algorithm with early termination:
        1. For each a in N(u) \ {v}
        2.   For each b in N(a) \ {u, v, a}
        3.     For each c in N(b) \ {u, v, a, b}
        4.       Check if N(c) ∩ N(v) contains any d distinct from u, a, b, c
        """
        nu = adj_mask[u] & ~(1 << v)
        nv = adj_mask[v] & ~(1 << u)
        
        # Precompute common neighbors of v for faster checks
        a = nu
        while a:
            la = a & (-a)
            a ^= la
            ai = la.bit_length() - 1
            
            # b must be in N(a), not u, v, or ai
            nb = adj_mask[ai] & ~(1 << u) & ~(1 << v) & ~(1 << ai)
            b = nb
            while b:
                lb = b & (-b)
                b ^= lb
                bi = lb.bit_length() - 1
                
                # c must be in N(b), not u, v, ai, or bi
                nc = adj_mask[bi] & ~(1 << u) & ~(1 << v) & ~(1 << ai) & ~(1 << bi)
                c = nc
                while c:
                    lc = c & (-c)
                    c ^= lc
                    ci = lc.bit_length() - 1
                    
                    # d must be in N(ci) ∩ N(v), and distinct from u, ai, bi, ci
                    # Also d must not be v (nv excludes u but d is neighbor of v)
                    # d must not be u, ai, bi, ci
                    nd = adj_mask[ci] & nv & ~(1 << u) & ~(1 << ai) & ~(1 << bi) & ~(1 << ci)
                    if nd:
                        return True
        
        return False
    
    def get_all_edges(mask):
        """Get list of edges from adjacency mask."""
        edges = []
        for u in range(n):
            m = mask[u]
            while m:
                lb = m & (-m)
                m ^= lb
                v = lb.bit_length() - 1
                if u < v:
                    edges.append((u, v))
        return edges
    
    def local_search(initial_mask, time_limit):
        """Local search: try to add edges, if stuck remove random edges and retry."""
        adj_mask[:] = list(initial_mask)
        best_mask = list(adj_mask)
        best_count = total_edges()
        
        while time.time() < time_limit - 2:
            # Try to add any valid edge
            non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) 
                         if not ((1 << j) & adj_mask[i])]
            
            if not non_edges:
                break
            
            # Sort by potential: prefer edges where both endpoints have lower degree
            # This tends to fill in sparse areas first
            non_edges.sort(key=lambda e: (degree(e[0]) + degree(e[1]), random.random()))
            
            added_any = False
            for (u, v) in non_edges:
                if time.time() > time_limit - 1:
                    break
                if (1 << v) & adj_mask[u]:
                    continue
                if has_c6_fast(u, v):
                    continue
                add_edge(u, v)
                added_any = True
                current_count = total_edges()
                if current_count > best_count:
                    best_count = current_count
                    best_mask = list(adj_mask)
                break  # Add one edge at a time for more frequent best updates
            
            if not added_any:
                # Stuck: remove 1-3 random edges to escape local optimum
                current_edges = get_all_edges(adj_mask)
                if not current_edges:
                    break
                
                # Prefer removing edges from high-degree vertices to create more room
                current_edges.sort(key=lambda e: (degree(e[0]) + degree(e[1]), random.random()), reverse=True)
                
                k = random.randint(1, 3)
                edges_to_remove = current_edges[:min(k, len(current_edges))]
                for (u, v) in edges_to_remove:
                    remove_edge(u, v)
        
        return best_mask, best_count
    
    # Initial state: 8 disjoint K5s
    init_adj = [0] * n
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                init_adj[5 * k + a] |= (1 << (5 * k + b))
                init_adj[5 * k + b] |= (1 << (5 * k + a))
    
    best_overall_mask = list(init_adj)
    best_overall_count = total_edges()
    
    # Phase 1: Greedy build from initial state
    t1_end = time.time() + 8
    mask1, count1 = local_search(init_adj, t1_end)
    if count1 > best_overall_count:
        best_overall_count = count1
        best_overall_mask = mask1
    
    # Phase 2: Multiple local search runs with perturbations
    remaining_time = deadline - time.time() - 5
    num_restarts = max(1, int(remaining_time / 8))  # ~8s per restart
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        if r == 0:
            start_mask = list(best_overall_mask)
        else:
            start_mask = list(best_overall_mask)
            current_edges = get_all_edges(start_mask)
            if current_edges:
                # Remove edges from high-degree vertices to create more room
                current_edges.sort(key=lambda e: (degree(e[0]) + degree(e[1]), random.random()), reverse=True)
                edges_to_remove = current_edges[:min(3, len(current_edges))]
                for (u, v) in edges_to_remove:
                    start_mask[u] &= ~(1 << v)
                    start_mask[v] &= ~(1 << u)
        
        t_end = time.time() + min(8, remaining_time - (r * 8))
        if t_end < time.time() + 1:
            break
        mask, count = local_search(start_mask, t_end)
        
        if count > best_overall_count:
            best_overall_count = count
            best_overall_mask = mask

    # Build final edge list from the best state
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (1 << v) & best_overall_mask[u]:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
