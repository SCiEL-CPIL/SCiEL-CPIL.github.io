"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _has_c6(adj, n):
    """
    Check if graph with adjacency bitmasks has a 6-cycle.
    adj: list of int bitmasks.
    n: number of vertices.
    """
    for u in range(n):
        # Neighbors of u
        N_u = adj[u]
        if N_u == 0:
            continue
        
        # Iterate v in N_u where v > u (to avoid checking each cycle multiple times, though strictness isn't required for correctness, just speed)
        # We can just iterate all v in N_u, but checking u-v-w-x-y-z-u implies a specific direction.
        # To be safe and simple, we check all paths of length 5 starting at u and ending at a neighbor of u.
        # However, standard C6 detection:
        # u - v - w - x - y - z - u
        # Constraints:
        # v in N(u)
        # w in N(v) \ {u}
        # x in N(w) \ {u, v}
        # y in N(x) \ {u, v, w}
        # z in N(y) \ {u, v, w, x}
        # u in N(z)
        
        # Optimization: Precompute N(u) \ {u} is just adj[u] since no self-loops.
        
        # To reduce symmetry, we can enforce v > u.
        # But bitmasks make iteration easy.
        # Let's iterate v.
        mask_v = N_u
        while mask_v:
            v = (mask_v & -mask_v).bit_length() - 1
            mask_v &= mask_v - 1
            
            if v <= u:
                continue
                
            # w in N(v) \ {u}
            mask_w = adj[v] & ~(1 << u)
            if mask_w == 0:
                continue
            
            while mask_w:
                w = (mask_w & -mask_w).bit_length() - 1
                mask_w &= mask_w - 1
                
                # x in N(w) \ {u, v}
                mask_x = adj[w] & ~(1 << u) & ~(1 << v)
                if mask_x == 0:
                    continue
                
                while mask_x:
                    x = (mask_x & -mask_x).bit_length() - 1
                    mask_x &= mask_x - 1
                    
                    # y in N(x) \ {u, v, w}
                    mask_y = adj[x] & ~(1 << u) & ~(1 << v) & ~(1 << w)
                    if mask_y == 0:
                        continue
                    
                    while mask_y:
                        y = (mask_y & -mask_y).bit_length() - 1
                        mask_y &= mask_y - 1
                        
                        # z in N(y) \ {u, v, w, x}
                        mask_z = adj[y] & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x)
                        if mask_z == 0:
                            continue
                        
                        while mask_z:
                            z = (mask_z & -mask_z).bit_length() - 1
                            mask_z &= mask_z - 1
                            
                            # Check if u in N(z)
                            if adj[z] & (1 << u):
                                return True
    return False

def _greedy_build(n, seed=None, time_limit=10):
    """
    Build a C6-free graph greedily using bitsets for speed.
    Starts with a dense structure to maximize edges quickly.
    """
    if seed is not None:
        random.seed(seed)
    
    adj = [0] * n
    edge_count = 0
    start_time = time.time()
    
    # Strategy: Start with disjoint cliques (e.g., K5s) to get a high base edge count immediately.
    # 8 * K5 = 8 * 10 = 80 edges.
    # Or 4 * K6 = 4 * 15 = 60 edges.
    # K5 is better for edge density per vertex in small clusters.
    # Let's try 8 K5s.
    for k in range(8):
        start = k * 5
        for a in range(5):
            for b in range(a + 1, 5):
                u = start + a
                v = start + b
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
    
    # Now try to add cross-edges greedily
    # Generate all possible edges not yet present
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not (adj[i] & (1 << j))]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() - start_time > time_limit:
            break
        
        # Tentatively add
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        if _has_c6(adj, n):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            edge_count += 1
            
    return adj, edge_count

def _local_search(adj, n, time_limit=30):
    """
    Improve graph by edge swaps.
    adj: list of bitmasks.
    """
    start_time = time.time()
    best_adj = adj[:]
    best_count = sum(bin(a).count('1') for a in adj) // 2
    
    all_possible = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    while time.time() - start_time < time_limit:
        # Pick a random edge to remove
        # We need to pick an existing edge.
        # Let's pick a random u, then a random neighbor v.
        u = random.randint(0, n - 1)
        if adj[u] == 0:
            continue
        v = (adj[u] & -adj[u]).bit_length() - 1
        
        # Remove edge (u, v)
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        # Try to add a random non-edge
        added = False
        # Try a few random candidates
        for _ in range(100):
            a, b = random.choice(all_possible)
            if adj[a] & (1 << b):
                continue
            
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
            
            if not _has_c6(adj, n):
                added = True
                break
            else:
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
        
        if added:
            new_count = sum(bin(a).count('1') for a in adj) // 2
            if new_count > best_count:
                best_count = new_count
                best_adj = adj[:]
        else:
            # Restore removed edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    return best_adj, best_count

def construct():
    n = 40
    deadline = 140  # Safety margin before 150s limit
    
    best_adj = [0] * n
    best_count = 0
    
    start_time = time.time()
    
    # Try multiple seeds for greedy build
    # The bitset _has_c6 is much faster, so we can try more seeds or longer times
    seeds = [42, 123, 456, 789, 1000, 2024, 3141, 2718, 1234, 5678, 9012, 3456]
    
    for seed in seeds:
        if time.time() - start_time > deadline - 20:
            break
        
        # Allocate remaining time, but cap individual greedy runs to allow for local search
        remaining_for_greedy = min(5, deadline - 20 - (time.time() - start_time))
        if remaining_for_greedy < 0:
            break
            
        adj, count = _greedy_build(n, seed=seed, time_limit=remaining_for_greedy)
        
        if count > best_count:
            best_count = count
            best_adj = adj
    
    # Local search on the best found so far
    if time.time() - start_time < deadline - 5:
        remaining = deadline - 5 - (time.time() - start_time)
        improved_adj, improved_count = _local_search(best_adj, n, time_limit=remaining)
        if improved_count > best_count:
            best_count = improved_count
            best_adj = improved_adj
    
    # Convert bitmasks to edge list
    edges = []
    for u in range(n):
        mask = best_adj[u]
        while mask:
            v = (mask & -mask).bit_length() - 1
            mask &= mask - 1
            if u < v:
                edges.append([u, v])
                
    return edges
# EVOLVE-BLOCK-END
