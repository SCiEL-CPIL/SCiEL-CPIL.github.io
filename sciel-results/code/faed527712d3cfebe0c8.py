# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    n = 40
    best_edges = []
    best_count = 0
    
    def run_greedy(shuffle_seed=None):
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
        
        # Start with 8 disjoint K5's (104 edges)
        edges = []
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    edges.append([5 * k + a, 5 * k + b])
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        # Bitset adjacency list
        adj = [0] * n
        for (a, b) in edge_set:
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
            
        def bits(mask):
            while mask:
                lsb = mask & -mask
                yield lsb.bit_length() - 1
                mask ^= lsb

        def has_c6(u, v):
            mask_u = adj[u]
            mask_v = adj[v]
            
            for w in bits(mask_u):
                if w == v:
                    continue
                
                mask_x = adj[w] & ~(1 << u) & ~(1 << v) & ~(1 << w)
                for x in bits(mask_x):
                    mask_y = adj[x] & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x)
                    for y in bits(mask_y):
                        mask_z = adj[y] & mask_v & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x) & ~(1 << y)
                        if mask_z:
                            return True
            return False
        
        all_possible = [(a, b) for a in range(n) for b in range(a + 1, n)]
        random.shuffle(all_possible)
        
        current_edge_count = len(edge_set)
        
        for (u, v) in all_possible:
            if time.time() > deadline:
                break
            if (u, v) in edge_set:
                continue
            
            if not has_c6(u, v):
                edge_set.add((u, v))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edge_count += 1
                
        return [[a, b] for (a, b) in sorted(edge_set)], current_edge_count

    # Run multiple times if time permits
    # First run without seed to use global random state
    edges, count = run_greedy()
    best_edges = edges
    best_count = count
    
    # If we have time, try again with a different seed
    if time.time() < deadline - 10:
        edges2, count2 = run_greedy(42)
        if count2 > best_count:
            best_edges = edges2
            best_count = count2
            
    if time.time() < deadline - 10:
        edges3, count3 = run_greedy(123)
        if count3 > best_count:
            best_edges = edges3
            best_count = count3

    return best_edges
# EVOLVE-BLOCK-END
