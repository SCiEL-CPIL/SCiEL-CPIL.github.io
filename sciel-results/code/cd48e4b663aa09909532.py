"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _is_k4_free_fast(n, adj_bits):
    """Check if graph is K4-free using bitmasks. adj_bits[i] is bitmask of neighbors of i."""
    for a in range(n):
        for b in range(a+1, n):
            if not (adj_bits[a] >> b) & 1:
                continue
            common_ab = adj_bits[a] & adj_bits[b]
            for c in range(b+1, n):
                if not (common_ab >> c) & 1:
                    continue
                common_abc = common_ab & adj_bits[c]
                mask_above_c = ((1 << n) - 1) ^ ((1 << (c+1)) - 1)
                if common_abc & mask_above_c:
                    return False
    return True

def _largest_triangle_free_fast(n, adj_bits):
    """Find largest triangle-free induced subgraph."""
    best = 0
    
    for size in range(n, 0, -1):
        found = False
        for subset in combinations(range(n), size):
            # Check if triangle-free
            is_tf = True
            for i in range(len(subset)):
                for j in range(i+1, len(subset)):
                    if (adj_bits[subset[i]] >> subset[j]) & 1:
                        for k in range(j+1, len(subset)):
                            if ((adj_bits[subset[i]] >> subset[k]) & 1) and ((adj_bits[subset[j]] >> subset[k]) & 1):
                                is_tf = False
                                break
                        if not is_tf:
                            break
                if not is_tf:
                    break
            if is_tf:
                found = True
                break
        if found:
            best = size
            break
    
    return best

def _bits_to_edges(n, adj_bits):
    """Convert bitmask adjacency to edge list."""
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (adj_bits[i] >> j) & 1:
                edges.append([i, j])
    return edges

def construct():
    n = 18
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Start with the known construction: complete 3-partite graph K6,6,6
    best_score = 12
    best_adj_bits = [0] * n
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    best_adj_bits[a] |= (1 << b)
                    best_adj_bits[b] |= (1 << a)
    
    rng = random.Random(42)
    
    # Local search: try removing edges from the baseline
    for iteration in range(20000):
        if time.time() > deadline:
            break
        
        adj_bits = best_adj_bits[:]
        
        # Try removing 1-3 random edges
        num_removals = rng.randint(1, 3)
        possible_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj_bits[i] >> j) & 1:
                    possible_edges.append((i, j))
        
        if len(possible_edges) < num_removals:
            break
        
        for _ in range(num_removals):
            idx = rng.randrange(len(possible_edges))
            i, j = possible_edges[idx]
            adj_bits[i] &= ~(1 << j)
            adj_bits[j] &= ~(1 << i)
            possible_edges.pop(idx)
        
        # Check K4-free
        if not _is_k4_free_fast(n, adj_bits):
            continue
        
        # Evaluate
        score = _largest_triangle_free_fast(n, adj_bits)
        if score < best_score:
            best_score = score
            best_adj_bits = adj_bits[:]
            if best_score <= 9:
                break
    
    # Also try random graphs and remove K4s
    for p in [0.5, 0.6, 0.7]:
        for trial in range(2000):
            if time.time() > deadline:
                break
            
            adj_bits = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if rng.random() < p:
                        adj_bits[i] |= (1 << j)
                        adj_bits[j] |= (1 << i)
            
            # Iteratively remove edges to break K4s
            for _ in range(100):
                found_k4 = False
                for a in range(n):
                    for b in range(a+1, n):
                        if not (adj_bits[a] >> b) & 1:
                            continue
                        common_ab = adj_bits[a] & adj_bits[b]
                        for c in range(b+1, n):
                            if not (common_ab >> c) & 1:
                                continue
                            common_abc = common_ab & adj_bits[c]
                            mask_above_c = ((1 << n) - 1) ^ ((1 << (c+1)) - 1)
                            if common_abc & mask_above_c:
                                d_mask = common_abc & mask_above_c
                                d = d_mask.bit_length() - 1
                                adj_bits[a] &= ~(1 << d)
                                adj_bits[d] &= ~(1 << a)
                                found_k4 = True
                                break
                        if found_k4:
                            break
                    if found_k4:
                        break
                if not found_k4:
                    break
            
            if not _is_k4_free_fast(n, adj_bits):
                continue
            
            score = _largest_triangle_free_fast(n, adj_bits)
            if score < best_score:
                best_score = score
                best_adj_bits = adj_bits[:]
                if best_score <= 9:
                    break
        if time.time() > deadline:
            break
        if best_score <= 9:
            break
    
    edges = _bits_to_edges(n, best_adj_bits)
    return edges
# EVOLVE-BLOCK-END
