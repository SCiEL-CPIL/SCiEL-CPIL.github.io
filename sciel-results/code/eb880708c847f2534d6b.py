import time
import random
import numpy as np

DEADLINE = 120.0
SAFETY_MARGIN = 10.0

def construct():
    start_time = time.time()
    n = 40
    
    # Bitmask-based adjacency
    adj_mask = [0] * n
    
    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    def remove_edge(u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
    
    def has_edge(u, v):
        return (1 << v) & adj_mask[u]
    
    def degree(u):
        return bin(adj_mask[u]).count('1')
    
    def total_edges():
        return sum(degree(u) for u in range(n)) // 2
    
    def has_c6_fast(u, v):
        """Check if adding edge (u,v) creates a C6 using bitmasks.
        
        Adding edge (u,v) creates a C6 if there is a path of length 5 from u to v
        in the current graph: u-a-b-c-d-v with all 6 vertices distinct.
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
                    
                    # d must be in N(ci) ∩ N(v), distinct from u, ai, bi, ci
                    nd = adj_mask[ci] & nv & ~(1 << u) & ~(1 << ai) & ~(1 << bi) & ~(1 << ci)
                    if nd:
                        return True
        
        return False
    
    def get_all_edges():
        edges = []
        for u in range(n):
            mask = adj_mask[u]
            while mask:
                lb = mask & (-mask)
                mask ^= lb
                v = lb.bit_length() - 1
                if u < v:
                    edges.append((u, v))
        return edges
    
    def greedy_fill(adj_mask_ref, all_pairs, time_limit):
        """Try to add as many edges as possible without creating C6."""
        added_any = False
        random.shuffle(all_pairs)
        for (u, v) in all_pairs:
            if time.time() > time_limit:
                break
            if (1 << v) & adj_mask_ref[u]:
                continue
            if not has_c6_fast(u, v):
                add_edge(u, v)
                added_any = True
        return added_any
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Initial state: 8 disjoint K5s (C6-free, 40 edges)
    def init_k5s():
        init_mask = [0] * n
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    init_mask[5 * k + a] |= (1 << (5 * k + b))
                    init_mask[5 * k + b] |= (1 << (5 * k + a))
        return init_mask
    
    best_overall_mask = init_k5s()
    best_overall_count = total_edges()
    
    # Multiple restarts with different strategies
    num_restarts = 5
    elapsed = time.time() - start_time
    time_per_restart = (DEADLINE - SAFETY_MARGIN - elapsed) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        # Set up starting state
        if r == 0:
            start_mask = list(init_k5s())
        elif r == 1:
            # Random sparse start
            start_mask = [0] * n
            random.seed(42 + r)
            for (u, v) in random.sample(all_pairs, min(30, len(all_pairs))):
                if not has_c6_fast(u, v):
                    add_edge(u, v)
            # Reset adj_mask from start_mask
            adj_mask[:] = list(start_mask)
        else:
            # Perturb best found so far
            start_mask = list(best_overall_mask)
            current_edges = get_all_edges()
            if current_edges:
                num_remove = random.randint(3, min(6, len(current_edges)))
                edges_to_remove = random.sample(current_edges, num_remove)
                for (u, v) in edges_to_remove:
                    remove_edge(u, v)
            adj_mask[:] = list(start_mask)
        
        # Local search with aggressive perturbation
        stuck_count = 0
        t_end = time.time() + time_per_restart
        
        while time.time() < t_end:
            added_any = greedy_fill(adj_mask, all_pairs, t_end)
            
            current_count = total_edges()
            if current_count > best_overall_count:
                best_overall_count = current_count
                best_overall_mask = list(adj_mask)
                stuck_count = 0
            
            if added_any:
                continue
            
            # Stuck - try perturbation
            stuck_count += 1
            
            if stuck_count >= 2:
                current_edges = get_all_edges()
                if len(current_edges) < 2:
                    break
                
                # Aggressive multi-edge perturbation: remove 2-5 edges
                num_remove = random.randint(2, min(5, len(current_edges)))
                to_remove = random.sample(current_edges, num_remove)
                
                saved_adj = list(adj_mask)
                
                for (u, v) in to_remove:
                    remove_edge(u, v)
                
                # Greedily add back
                greedy_fill(adj_mask, all_pairs, t_end)
                
                new_count = total_edges()
                if new_count > best_overall_count:
                    best_overall_count = new_count
                    best_overall_mask = list(adj_mask)
                    stuck_count = 0
                else:
                    # Revert
                    adj_mask[:] = list(saved_adj)
                    stuck_count += 1
            
            # If very stuck, try larger perturbation
            if stuck_count >= 8:
                adj_mask[:] = list(best_overall_mask)
                current_edges = get_all_edges()
                if current_edges:
                    num_remove = random.randint(4, min(8, len(current_edges)))
                    to_remove = random.sample(current_edges, num_remove)
                    for (u, v) in to_remove:
                        remove_edge(u, v)
                greedy_fill(adj_mask, all_pairs, t_end)
                new_count = total_edges()
                if new_count > best_overall_count:
                    best_overall_count = new_count
                    best_overall_mask = list(adj_mask)
                stuck_count = 0
    
    # Build final edge list from best state
    edges = []
    for u in range(n):
        mask = best_overall_mask[u]
        while mask:
            lb = mask & (-mask)
            mask ^= lb
            v = lb.bit_length() - 1
            if u < v:
                edges.append([u, v])
    
    return edges
