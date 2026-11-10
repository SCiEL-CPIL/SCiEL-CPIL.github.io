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
        
        Adding edge (u,v) creates a C6 if there is a path of length 5 from u to v
        in the current graph (excluding the edge u-v itself, which is not present).
        Path: u - a - b - c - d - e - v (6 edges total in the cycle, so 5 in path? No.)
        
        Cycle C6 has 6 vertices: u, a, b, c, d, v.
        Edges: (u,a), (a,b), (b,c), (c,d), (d,v), (v,u).
        If we add (u,v), we need a path of length 4 from u to v in the existing graph?
        Path: u - a - b - c - d - v. Length 4 edges? No, 4 edges means 5 vertices.
        u-a (1), a-b (2), b-c (3), c-d (4), d-v (5). 
        So we need a path of length 5 from u to v?
        Let's trace:
        u is connected to a.
        a is connected to b.
        b is connected to c.
        c is connected to d.
        d is connected to v.
        This is a path of length 4 (edges: u-a, a-b, b-c, c-d, d-v is 5 edges).
        Wait, u-a, a-b, b-c, c-d, d-v. That is 5 edges.
        So we need a path of length 5 from u to v in the graph G - (u,v).
        
        Vertices: u, a, b, c, d, v. All distinct.
        
        Algorithm:
        1. For each a in N(u) \ {v}
        2.   For each b in N(a) \ {u, v, a}
        3.     For each c in N(b) \ {u, v, a, b}
        4.       For each d in N(c) \ {u, v, a, b, c}
        5.         If d is in N(v) \ {u, a, b, c, d}, then C6 exists.
        """
        nu = adj_mask[u] & ~(1 << v)
        nv = adj_mask[v] & ~(1 << u)
        
        a = nu
        while a:
            la = a & (-a)
            a ^= la
            ai = la.bit_length() - 1
            
            nb = adj_mask[ai] & ~(1 << u) & ~(1 << v) & ~(1 << ai)
            b = nb
            while b:
                lb = b & (-b)
                b ^= lb
                bi = lb.bit_length() - 1
                
                nc = adj_mask[bi] & ~(1 << u) & ~(1 << v) & ~(1 << ai) & ~(1 << bi)
                c = nc
                while c:
                    lc = c & (-c)
                    c ^= lc
                    ci = lc.bit_length() - 1
                    
                    # d must be in N(ci) and N(v), and distinct from u, ai, bi, ci
                    # Also d != v (handled by nv excluding u, but d is neighbor of v)
                    # d must not be u, ai, bi, ci
                    nd = adj_mask[ci] & nv & ~(1 << u) & ~(1 << ai) & ~(1 << bi) & ~(1 << ci)
                    if nd:
                        return True
        
        return False

    def local_search(initial_mask, time_limit):
        """Perform local search starting from a given state."""
        # Deep copy initial mask
        adj_mask[:] = list(initial_mask)
        
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        best_adj_mask = list(adj_mask)
        best_count = total_edges()
        
        # Greedy phase
        while time.time() < time_limit:
            random.shuffle(all_pairs)
            added_any = False
            
            for (u, v) in all_pairs:
                if time.time() > time_limit:
                    break
                if (1 << v) & adj_mask[u]:
                    continue
                if not has_c6_fast(u, v):
                    add_edge(u, v)
                    added_any = True
            
            current_count = total_edges()
            if current_count > best_count:
                best_count = current_count
                best_adj_mask = list(adj_mask)
            
            if not added_any:
                # Perturbation: remove 2-3 random edges
                if time.time() > time_limit - 1:
                    break
                
                current_edges = []
                for u in range(n):
                    mask = adj_mask[u]
                    while mask:
                        lb = mask & (-mask)
                        mask ^= lb
                        v = lb.bit_length() - 1
                        if u < v:
                            current_edges.append((u, v))
                
                if not current_edges:
                    break
                
                # Remove 2 random edges
                k = min(2, len(current_edges))
                edges_to_remove = random.sample(current_edges, k)
                for (u, v) in edges_to_remove:
                    remove_edge(u, v)
                
                # Try to re-add
                random.shuffle(all_pairs)
                for (u, v) in all_pairs:
                    if time.time() > time_limit:
                        break
                    if (1 << v) & adj_mask[u]:
                        continue
                    if not has_c6_fast(u, v):
                        add_edge(u, v)
                
                current_count = total_edges()
                if current_count > best_count:
                    best_count = current_count
                    best_adj_mask = list(adj_mask)
        
        return best_adj_mask, best_count

    # Initial state: 8 disjoint K5s
    init_adj = [0] * n
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                init_adj[5 * k + a] |= (1 << (5 * k + b))
                init_adj[5 * k + b] |= (1 << (5 * k + a))
    
    best_overall_mask = list(init_adj)
    best_overall_count = total_edges()
    
    # Run multiple local searches with different random seeds/orderings
    # Split time budget among a few restarts
    num_restarts = 3
    time_per_restart = (deadline - time.time()) / num_restarts - 2
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Slightly perturb the initial state for diversity in later restarts
        if r == 0:
            start_mask = list(init_adj)
        else:
            start_mask = list(best_overall_mask)
            # Remove a few random edges to escape local optimum
            current_edges = []
            for u in range(n):
                mask = start_mask[u]
                while mask:
                    lb = mask & (-mask)
                    mask ^= lb
                    v = lb.bit_length() - 1
                    if u < v:
                        current_edges.append((u, v))
            
            if current_edges:
                edges_to_remove = random.sample(current_edges, min(3, len(current_edges)))
                for (u, v) in edges_to_remove:
                    start_mask[u] &= ~(1 << v)
                    start_mask[v] &= ~(1 << u)
        
        t_end = time.time() + time_per_restart
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
