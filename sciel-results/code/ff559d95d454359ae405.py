"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
from itertools import combinations

N = 18
FULL_MASK = (1 << N) - 1

def _is_k4_free(adj_bits):
    """Check if graph is K4-free using bitmasks."""
    for a in range(N):
        na = adj_bits[a]
        for b in range(a + 1, N):
            if not (na >> b) & 1:
                continue
            common_ab = na & adj_bits[b]
            for c in range(b + 1, N):
                if not (common_ab >> c) & 1:
                    continue
                common_abc = common_ab & adj_bits[c]
                mask_above_c = FULL_MASK ^ ((1 << (c + 1)) - 1)
                if common_abc & mask_above_c:
                    return False
    return True

def _largest_triangle_free(adj_bits):
    """Find largest triangle-free induced subgraph size."""
    # Precompute adjacency lists for faster iteration if needed, 
    # but bitmask is likely faster for small N.
    
    def is_triangle_free(mask):
        m = mask
        while m:
            low_bit = m & (-m)
            i = low_bit.bit_length() - 1
            m ^= low_bit
            ni = adj_bits[i] & mask
            # Check if ni is an independent set
            nn = ni
            while nn:
                low_bit2 = nn & (-nn)
                j = low_bit2.bit_length() - 1
                nn ^= low_bit2
                if adj_bits[j] & ni:
                    return False
        return True

    # Check sizes from N down to 1
    # Optimization: If we find a triangle-free set of size k, 
    # we can stop if we are just checking existence for that size.
    # We want the MAXIMUM size.
    
    for size in range(N, 0, -1):
        found = False
        # Use combinations for clarity, but it's slow.
        # For N=18, size=10, C(18,10) = 43758.
        # This loop might be slow in Python.
        
        # Alternative: Random sampling? No, need exact.
        # Let's stick to combinations but break early.
        
        for subset in combinations(range(N), size):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if is_triangle_free(mask):
                found = True
                break
        if found:
            return size
    return 0

def _bits_to_edges(adj_bits):
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if (adj_bits[i] >> j) & 1:
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    rng = random.Random(12345)
    
    # Phase 1: Greedy Construction
    # Start with empty graph, add edges if K4-free.
    # This yields a maximal K4-free graph.
    
    def greedy_construct(seed_val):
        adj = [0] * N
        all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        rng_g = random.Random(seed_val)
        rng_g.shuffle(all_edges)
        
        for (u, v) in all_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if not _is_k4_free(adj):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        return adj

    best_adj = greedy_construct(42)
    best_score = _largest_triangle_free(best_adj)
    
    # Phase 2: Simulated Annealing
    current_adj = best_adj[:]
    current_score = best_score
    
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    T = 5.0
    T_min = 0.1
    cooling = 0.99
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        move_type = rng.random()
        
        if move_type < 0.3:
            # Add edge
            non_edges = [(u, v) for (u, v) in all_edges if not (current_adj[u] >> v) & 1]
            if not non_edges:
                continue
            u, v = rng.choice(non_edges)
            new_adj = current_adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            if not _is_k4_free(new_adj):
                continue
            new_score = _largest_triangle_free(new_adj)
            delta = new_score - current_score
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_adj = new_adj
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj[:]
                    if best_score <= 8: # Theoretical lower bound is likely around 9-10
                        break
                        
        elif move_type < 0.7:
            # Remove edge
            edges = [(u, v) for (u, v) in all_edges if (current_adj[u] >> v) & 1]
            if not edges:
                continue
            u, v = rng.choice(edges)
            new_adj = current_adj[:]
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            new_score = _largest_triangle_free(new_adj)
            delta = new_score - current_score
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_adj = new_adj
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj[:]
                    if best_score <= 8:
                        break
        else:
            # Swap: remove one, add one
            edges = [(u, v) for (u, v) in all_edges if (current_adj[u] >> v) & 1]
            non_edges = [(u, v) for (u, v) in all_edges if not (current_adj[u] >> v) & 1]
            if not edges or not non_edges:
                continue
            u1, v1 = rng.choice(edges)
            u2, v2 = rng.choice(non_edges)
            
            new_adj = current_adj[:]
            new_adj[u1] &= ~(1 << v1)
            new_adj[v1] &= ~(1 << u1)
            new_adj[u2] |= (1 << v2)
            new_adj[v2] |= (1 << u2)
            
            if not _is_k4_free(new_adj):
                continue
            
            new_score = _largest_triangle_free(new_adj)
            delta = new_score - current_score
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_adj = new_adj
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj[:]
                    if best_score <= 8:
                        break
        
        T *= cooling
        if T < T_min:
            T = T_min

    return _bits_to_edges(best_adj)
# EVOLVE-BLOCK-END
