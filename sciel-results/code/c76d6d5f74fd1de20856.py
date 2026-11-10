"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

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
        """Check if adding edge (u,v) creates a C6 using bitmasks."""
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
    
    def greedy_build(initial_mask, time_limit):
        """Greedy edge addition with C6 checking."""
        adj_mask[:] = list(initial_mask)
        best_mask = list(adj_mask)
        best_count = total_edges()
        
        # Get all non-edge pairs
        non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) 
                     if not ((1 << j) & adj_mask[i])]
        
        random.shuffle(non_edges)
        
        for (u, v) in non_edges:
            if time.time() > time_limit:
                break
            if (1 << v) & adj_mask[u]:
                continue
            if has_c6_fast(u, v):
                continue
            add_edge(u, v)
            current_count = total_edges()
            if current_count > best_count:
                best_count = current_count
                best_mask = list(adj_mask)
        
        return best_mask, best_count
    
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
            
            random.shuffle(non_edges)
            
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
                
                k = random.randint(1, 3)
                edges_to_remove = random.sample(current_edges, min(k, len(current_edges)))
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
    t1_end = time.time() + 5
    mask1, count1 = greedy_build(init_adj, t1_end)
    if count1 > best_overall_count:
        best_overall_count = count1
        best_overall_mask = mask1
    
    # Phase 2: Multiple local search runs with perturbations
    remaining_time = deadline - time.time() - 5
    num_restarts = max(1, int(remaining_time / 10))  # ~10s per restart
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        if r == 0:
            start_mask = list(best_overall_mask)
        else:
            start_mask = list(best_overall_mask)
            current_edges = get_all_edges(start_mask)
            if current_edges:
                edges_to_remove = random.sample(current_edges, min(3, len(current_edges)))
                for (u, v) in edges_to_remove:
                    start_mask[u] &= ~(1 << v)
                    start_mask[v] &= ~(1 << u)
        
        t_end = time.time() + min(10, remaining_time - (r * 10))
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
