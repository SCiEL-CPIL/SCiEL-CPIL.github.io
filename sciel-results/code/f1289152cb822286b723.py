import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    n = 16
    target_edges = 65
    
    # Precompute all possible edges as (u, v) with u < v
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    all_possible_set = set(all_possible_edges)
    
    def build_adj_mask(edge_set):
        """Build bitmask adjacency and degree array."""
        adj_mask = [0] * n
        deg = [0] * n
        for u, v in edge_set:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj_mask, deg
    
    def score_from_mask(adj_mask, deg):
        """Compute max degree sum over triangles using bitmasks."""
        max_sum = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj_mask[i] & (1 << j):
                    common = adj_mask[i] & adj_mask[j]
                    # Only consider k > j to avoid duplicates
                    common &= ~((1 << (j+1)) - 1)
                    while common:
                        lsb = common & (-common)
                        k = lsb.bit_length() - 1
                        s = deg[i] + deg[j] + deg[k]
                        if s > max_sum:
                            max_sum = s
                        common ^= lsb
        return max_sum
    
    def score_edge_set(edge_set):
        """Score an edge set."""
        adj_mask, deg = build_adj_mask(edge_set)
        return score_from_mask(adj_mask, deg)
    
    # Start with K_{8,8} + edge (0,1)
    initial_edges = set()
    for u in range(8):
        for v in range(8, 16):
            initial_edges.add((u, v))
    initial_edges.add((0, 1))
    
    best_edges = initial_edges
    best_score = score_edge_set(best_edges)
    best_adj_mask, best_deg = build_adj_mask(best_edges)
    
    # Local search with bitmask-accelerated scoring
    for iteration in range(200000):
        if time.time() > deadline:
            break
        
        # Try random swaps
        edge_list = list(best_edges)
        current_edge = random.choice(edge_list)
        possible_additions = [e for e in all_possible_edges if e not in best_edges]
        
        if not possible_additions:
            break
            
        new_edge = random.choice(possible_additions)
        
        # Try the swap
        new_edge_set = set(best_edges)
        new_edge_set.discard(current_edge)
        new_edge_set.add(new_edge)
        
        if len(new_edge_set) != target_edges:
            continue
        
        # Fast scoring using bitmasks
        new_adj_mask, new_deg = build_adj_mask(new_edge_set)
        new_score = score_from_mask(new_adj_mask, new_deg)
        
        if new_score < best_score:
            best_score = new_score
            best_edges = new_edge_set
            best_adj_mask = new_adj_mask
            best_deg = new_deg
        elif new_score == best_score and random.random() < 0.2:
            # Accept equal score to escape local optima
            best_edges = new_edge_set
            best_adj_mask = new_adj_mask
            best_deg = new_deg
    
    # Final polish: try all single-edge swaps from current best
    if time.time() < deadline - 15:
        improved = True
        while improved:
            improved = False
            edge_list = list(best_edges)
            for current_edge in edge_list:
                if time.time() > deadline:
                    break
                for new_edge in all_possible_edges:
                    if new_edge in best_edges:
                        continue
                    new_edge_set = set(best_edges)
                    new_edge_set.discard(current_edge)
                    new_edge_set.add(new_edge)
                    if len(new_edge_set) != target_edges:
                        continue
                    new_adj_mask, new_deg = build_adj_mask(new_edge_set)
                    new_score = score_from_mask(new_adj_mask, new_deg)
                    if new_score < best_score:
                        best_score = new_score
                        best_edges = new_edge_set
                        best_adj_mask = new_adj_mask
                        best_deg = new_deg
                        improved = True
    
    # Format edges as [[u,v], ...]
    result_edges = [[min(u,v), max(u,v)] for u, v in best_edges]
    return {'n': n, 'edges': result_edges}
