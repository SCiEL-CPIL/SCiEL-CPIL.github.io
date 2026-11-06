import random
import time
import numpy as np

def construct():
    """
    Constructs a C6-free graph on 40 vertices with a high number of edges.
    Uses randomized greedy with bitset C6-check and local search (edge swaps).
    """
    N = 40
    deadline = time.time() + 110  # 10s safety margin
    
    # Use a single 64-bit integer to represent adjacency for each vertex
    # adj_mask[i] has bit j set if (i,j) is an edge
    best_edges = []
    best_count = 0
    
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    def would_create_c6(adj_mask, u, v):
        """Check if adding edge (u,v) creates a C6. Uses bitset operations."""
        # C6: u-a-x-y-b-v where all 6 vertices distinct
        # a in N(u)\{v}, b in N(v)\{u}, x in N(a)\{u,v,a,b}, y in N(b)\{u,v,a,b,x}
        # and (x,y) is an edge
        mask_u = adj_mask[u] & ~(1 << v)  # neighbors of u excluding v
        mask_v = adj_mask[v] & ~(1 << u)  # neighbors of v excluding u
        
        # Iterate over a in N(u)\{v}
        a_mask = mask_u
        while a_mask:
            lb_a = a_mask & (-a_mask)
            a = lb_a.bit_length() - 1
            a_mask ^= lb_a
            
            # b in N(v)\{u,a}
            b_mask = mask_v & ~(1 << a)
            while b_mask:
                lb_b = b_mask & (-b_mask)
                b = lb_b.bit_length() - 1
                b_mask ^= lb_b
                
                # x in N(a)\{u,v,a,b}
                x_mask = adj_mask[a] & ~((1 << u) | (1 << v) | (1 << a) | (1 << b))
                while x_mask:
                    lb_x = x_mask & (-x_mask)
                    x = lb_x.bit_length() - 1
                    x_mask ^= lb_x
                    
                    # y in N(b)\{u,v,a,b,x}
                    y_mask = adj_mask[b] & ~((1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << x))
                    # Check if any y in y_mask is adjacent to x
                    if (adj_mask[x] & y_mask) != 0:
                        return True
        
        return False
    
    def add_edge(adj_mask, edge_set, u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edge_set.add((min(u,v), max(u,v)))
    
    def remove_edge(adj_mask, edge_set, u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        edge_set.discard((min(u,v), max(u,v)))
    
    def local_search(adj_mask, edge_set, time_limit):
        """Try edge swaps to improve edge count."""
        limit = time.time() + time_limit
        improved = True
        while improved and time.time() < limit:
            improved = False
            # Try removing an edge and adding two others
            edges = list(edge_set)
            random.shuffle(edges)
            for (u1, v1) in edges:
                if time.time() >= limit:
                    break
                # Try to find two non-edges that can replace (u1,v1)
                # First, remove (u1,v1)
                remove_edge(adj_mask, edge_set, u1, v1)
                
                # Collect non-edges
                non_edges = []
                for i in range(N):
                    for j in range(i+1, N):
                        if i == u1 and j == v1:
                            continue
                        if j == u1 and i == v1:
                            continue
                        if not ((adj_mask[i] >> j) & 1):
                            non_edges.append((i, j))
                
                random.shuffle(non_edges)
                found = False
                for (a, b) in non_edges[:50]:
                    if time.time() >= limit:
                        break
                    if would_create_c6(adj_mask, a, b):
                        continue
                    add_edge(adj_mask, edge_set, a, b)
                    found = True
                    
                    # Try to add one more edge
                    for (c, d) in non_edges[:50]:
                        if (c, d) == (a, b):
                            continue
                        if would_create_c6(adj_mask, c, d):
                            continue
                        add_edge(adj_mask, edge_set, c, d)
                        break
                    break
                
                if not found:
                    # Restore (u1,v1)
                    add_edge(adj_mask, edge_set, u1, v1)
        
        return len(edge_set)
    
    # Multiple restarts with different strategies
    strategies = []
    for restart in range(200):
        if time.time() >= deadline:
            break
        
        random.shuffle(all_edges)
        adj_mask = [0] * N
        edge_set = set()
        
        # Greedy phase: add edges in shuffled order
        for u, v in all_edges:
            if time.time() >= deadline:
                break
            if not would_create_c6(adj_mask, u, v):
                add_edge(adj_mask, edge_set, u, v)
        
        # Local search phase
        remaining = deadline - time.time()
        if remaining > 5:
            local_search(adj_mask, edge_set, min(remaining - 5, 5))
        
        current_count = len(edge_set)
        if current_count > best_count:
            best_count = current_count
            best_edges = list(edge_set)
    
    # Return in the required format: list of [u, v]
    return [[u, v] for u, v in best_edges]
