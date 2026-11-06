"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations
import random
import math

# EVOLVE-BLOCK-START
def _build_bitmask(n, edges):
    """Build adjacency as list of bitmasks for fast triangle detection."""
    mask = [0] * n
    for u, v in edges:
        mask[u] |= (1 << v)
        mask[v] |= (1 << u)
    return mask

def _compute_score_fast(n, mask, deg):
    """Compute max triangle degree sum using bitmask operations."""
    max_score = 0
    for i in range(n):
        mi = mask[i]
        m = mi & ~((1 << (i + 1)) - 1)  # only j > i
        while m:
            j_bit = m & (-m)
            j = j_bit.bit_length() - 1
            m ^= j_bit
            common = mask[i] & mask[j] & ~((1 << (j + 1)) - 1)
            while common:
                k_bit = common & (-common)
                k = k_bit.bit_length() - 1
                common ^= k_bit
                s = deg[i] + deg[j] + deg[k]
                if s > max_score:
                    max_score = s
    return max_score

def _compute_spectral_radius(n, adj_matrix):
    """Compute spectral radius of adjacency matrix using power iteration."""
    if n == 0:
        return 0.0
    v = np.ones(n) / np.sqrt(n)
    for _ in range(200):
        new_v = adj_matrix @ v
        norm = np.linalg.norm(new_v)
        if norm < 1e-12:
            return 0.0
        new_v /= norm
        if np.linalg.norm(new_v - v) < 1e-10:
            break
        v = new_v
    return float(abs(np.dot(v, adj_matrix @ v)))

def _build_adj_matrix(n, edge_set):
    """Build numpy adjacency matrix from edge set."""
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in edge_set:
        adj[u, v] = 1
        adj[v, u] = 1
    return adj

def _local_search(n, edges, max_time=100, use_spectral_guide=True):
    """Local search with edge swaps, optionally guided by spectral radius."""
    start_time = time.time()
    deadline = start_time + max_time
    
    edge_set = set(tuple(sorted(e)) for e in edges)
    mask = _build_bitmask(n, edges)
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    
    current_score = _compute_score_fast(n, mask, deg)
    best_score = current_score
    best_edge_set = set(edge_set)
    
    all_pairs = set(tuple(sorted(p)) for p in combinations(range(n), 2))
    
    # Precompute all non-edges once
    non_edges = list(all_pairs - edge_set)
    
    rng = random.Random(42)
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline - 2:
        if not non_edges:
            break
        
        # Determine which edges to try removing
        edge_list = list(edge_set)
        rng.shuffle(edge_list)
        
        improved = False
        for e_remove in edge_list:
            if time.time() >= deadline - 2:
                break
            u1, v1 = e_remove
            
            # Shuffle non-edges for diversity
            non_edges_shuffled = list(non_edges)
            rng.shuffle(non_edges_shuffled)
            
            for e_add in non_edges_shuffled:
                u2, v2 = e_add
                if (u2, v2) in edge_set:
                    continue
                
                # Tentative swap
                mask[u1] &= ~(1 << v1)
                mask[v1] &= ~(1 << u1)
                mask[u2] |= (1 << v2)
                mask[v2] |= (1 << u2)
                deg[u1] -= 1
                deg[v1] -= 1
                deg[u2] += 1
                deg[v2] += 1
                
                new_score = _compute_score_fast(n, mask, deg)
                
                # Check if improvement or acceptable with spectral guidance
                accept = False
                if new_score < best_score:
                    accept = True
                elif new_score == current_score and use_spectral_guide:
                    # Accept if spectral radius decreases (guides toward quasi-regular)
                    # Only check occasionally to save time
                    if rng.random() < 0.1:
                        adj = _build_adj_matrix(n, edge_set)
                        adj[u1, v1] = 0
                        adj[v1, u1] = 0
                        adj[u2, v2] = 1
                        adj[v2, u2] = 1
                        new_spec = _compute_spectral_radius(n, adj)
                        old_spec = _compute_spectral_radius(n, _build_adj_matrix(n, edge_set))
                        if new_spec < old_spec - 0.01:
                            accept = True
                
                if accept:
                    best_score = min(best_score, new_score)
                    best_edge_set = set(edge_set)
                    best_edge_set.discard(e_remove)
                    best_edge_set.add(tuple(sorted(e_add)))
                    current_score = new_score
                    improved = True
                    edge_set.discard(e_remove)
                    edge_set.add(tuple(sorted(e_add)))
                    non_edges = list(all_pairs - edge_set)
                    no_improve_count = 0
                else:
                    # Revert
                    mask[u1] |= (1 << v1)
                    mask[v1] |= (1 << u1)
                    mask[u2] &= ~(1 << v2)
                    mask[v2] &= ~(1 << u2)
                    deg[u1] += 1
                    deg[v1] += 1
                    deg[u2] -= 1
                    deg[v2] -= 1
                
                if improved:
                    break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
    
    best_edges = [list(e) for e in best_edge_set]
    return best_edges, best_score

def _simulated_annealing(n, edges, max_time=30):
    """Simulated annealing to escape local optima."""
    start_time = time.time()
    deadline = start_time + max_time
    
    edge_set = set(tuple(sorted(e)) for e in edges)
    mask = _build_bitmask(n, edges)
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    
    current_score = _compute_score_fast(n, mask, deg)
    best_score = current_score
    best_edge_set = set(edge_set)
    
    all_pairs = set(tuple(sorted(p)) for p in combinations(range(n), 2))
    non_edges = list(all_pairs - edge_set)
    
    rng = random.Random(123)
    T = 3.0
    T_min = 0.05
    cooling = 0.9995
    
    while time.time() < deadline - 2 and T > T_min:
        if not non_edges:
            break
        
        e_remove = rng.choice(list(edge_set))
        e_add = rng.choice(non_edges)
        
        if e_add in edge_set:
            continue
        
        u1, v1 = e_remove
        u2, v2 = e_add
        
        # Tentative swap
        mask[u1] &= ~(1 << v1)
        mask[v1] &= ~(1 << u1)
        mask[u2] |= (1 << v2)
        mask[v2] |= (1 << u2)
        deg[u1] -= 1
        deg[v1] -= 1
        deg[u2] += 1
        deg[v2] += 1
        
        new_score = _compute_score_fast(n, mask, deg)
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_score = new_score
            edge_set.discard(e_remove)
            edge_set.add(tuple(sorted(e_add)))
            non_edges = list(all_pairs - edge_set)
            
            if current_score < best_score:
                best_score = current_score
                best_edge_set = set(edge_set)
        else:
            mask[u1] |= (1 << v1)
            mask[v1] |= (1 << u1)
            mask[u2] &= ~(1 << v2)
            mask[v2] &= ~(1 << u2)
            deg[u1] += 1
            deg[v1] += 1
            deg[u2] -= 1
            deg[v2] -= 1
        
        T *= cooling
    
    best_edges = [list(e) for e in best_edge_set]
    return best_edges, best_score

def _polish(n, edges, max_time=30):
    """Final polish: exhaustive single-swap search from best state."""
    start_time = time.time()
    deadline = start_time + max_time
    
    edge_set = set(tuple(sorted(e)) for e in edges)
    mask = _build_bitmask(n, edges)
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    
    current_score = _compute_score_fast(n, mask, deg)
    best_score = current_score
    best_edge_set = set(edge_set)
    
    all_pairs = set(tuple(sorted(p)) for p in combinations(range(n), 2))
    non_edges = list(all_pairs - edge_set)
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for e_remove in list(edge_set):
            if time.time() >= deadline - 2:
                break
            u1, v1 = e_remove
            for e_add in non_edges:
                if e_add in edge_set:
                    continue
                u2, v2 = e_add
                
                mask[u1] &= ~(1 << v1)
                mask[v1] &= ~(1 << u1)
                mask[u2] |= (1 << v2)
                mask[v2] |= (1 << u2)
                deg[u1] -= 1
                deg[v1] -= 1
                deg[u2] += 1
                deg[v2] += 1
                
                new_score = _compute_score_fast(n, mask, deg)
                
                if new_score < best_score:
                    best_score = new_score
                    best_edge_set = set(edge_set)
                    best_edge_set.discard(e_remove)
                    best_edge_set.add(tuple(sorted(e_add)))
                    edge_set.discard(e_remove)
                    edge_set.add(tuple(sorted(e_add)))
                    non_edges = list(all_pairs - edge_set)
                    improved = True
                else:
                    mask[u1] |= (1 << v1)
                    mask[v1] |= (1 << u1)
                    mask[u2] &= ~(1 << v2)
                    mask[v2] &= ~(1 << u2)
                    deg[u1] += 1
                    deg[v1] += 1
                    deg[u2] -= 1
                    deg[v2] -= 1
                
                if improved:
                    break
    
    best_edges = [list(e) for e in best_edge_set]
    return best_edges, best_score

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    total_deadline = 145  # safety margin before 150s limit
    
    # Seed 1: K_{6,10} + 5 matching edges in part B
    def make_seed1():
        edges = []
        for u in range(6):
            for v in range(6, 16):
                edges.append([u, v])
        internal_edges = [(6,7), (8,9), (10,11), (12,13), (14,15)]
        for u, v in internal_edges:
            edges.append([u, v])
        return edges
    
    # Seed 2: K_{8,8} + 1 internal edge
    def make_seed2():
        edges = [[u, v] for u in range(8) for v in range(8, 16)]
        edges.append([0, 1])
        return edges
    
    # Seed 3: K_{6,10} + 5 edges forming a path in part B
    def make_seed3():
        edges = []
        for u in range(6):
            for v in range(6, 16):
                edges.append([u, v])
        # Path: 6-7-8-9-10 (4 edges) + one more: 12-13
        internal_edges = [(6,7), (7,8), (8,9), (9,10), (12,13)]
        for u, v in internal_edges:
            edges.append([u, v])
        return edges
    
    # Evaluate all seeds
    seeds = [make_seed1(), make_seed2(), make_seed3()]
    best_edges = seeds[0]
    best_score = 999
    
    for seed in seeds:
        es = set(tuple(sorted(e)) for e in seed)
        mask = _build_bitmask(n, seed)
        deg = [0] * n
        for u, v in seed:
            deg[u] += 1
            deg[v] += 1
        s = _compute_score_fast(n, mask, deg)
        if s < best_score:
            best_score = s
            best_edges = seed
    
    # Phase 1: Local search with spectral guidance
    elapsed = time.time() - start_time
    search_time = min(90, max(30, total_deadline - 20 - elapsed))
    best_edges, best_score = _local_search(n, best_edges, max_time=search_time, use_spectral_guide=True)
    
    # Phase 2: Simulated annealing
    elapsed = time.time() - start_time
    sa_time = min(30, max(10, total_deadline - 15 - elapsed))
    best_edges, sa_score = _simulated_annealing(n, best_edges, max_time=sa_time)
    if sa_score < best_score:
        best_score = sa_score
    
    # Phase 3: Final polish
    elapsed = time.time() - start_time
    polish_time = min(20, max(5, total_deadline - 5 - elapsed))
    best_edges, final_score = _polish(n, best_edges, max_time=polish_time)
    if final_score < best_score:
        best_score = final_score
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
