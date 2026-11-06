"""Optimized Q3-free graph maximization on 16 vertices using bitmask representation and simulated annealing."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 16
    total_edges = n * (n - 1) // 2  # 120

    # Edge index mapping
    edge_idx = {}
    idx_edge = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[(i, j)] = len(idx_edge)
            idx_edge.append((i, j))

    # Q3 edges (12 edges)
    q3_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]

    # Precompute all 8-vertex subsets and for each, map Q3 edge positions to edge indices
    all_8sets = list(combinations(range(n), 8))
    
    # For each 8-subset, store the 12 edge indices (in canonical order) that correspond to Q3 edges
    # Q3 has 8 vertices labeled 0-7, with 12 edges
    q3_edge_positions = []
    for vs in all_8sets:
        pos_map = {}
        for i in range(8):
            pos_map[vs[i]] = i
        eidxs = []
        for a, b in q3_edges:
            # Map Q3 vertex labels to actual vertex indices
            va, vb = vs[a], vs[b]
            ei = edge_idx.get((min(va, vb), max(va, vb)))
            if ei is not None:
                eidxs.append(ei)
            else:
                eidxs.append(-1)  # shouldn't happen
        q3_edge_positions.append(eidxs)

    # Precompute for faster containment check: for each 8-subset, the bitmask of 12 Q3 edge indices
    # We'll represent the graph as a Python int bitmask of 120 bits

    def count_violations(mask):
        """Count number of Q3 embeddings where all 12 edges are present."""
        count = 0
        for eidxs in q3_edge_positions:
            # Check if all 12 edges are present in mask
            all_present = True
            for ei in eidxs:
                if not (mask >> ei) & 1:
                    all_present = False
                    break
            if all_present:
                count += 1
        return count

    def is_q3_free(mask):
        return count_violations(mask) == 0

    # Greedy construction to get a good starting point
    # Start with empty graph, add edges greedily
    current_mask = 0
    # Try to add edges in a smart order - prioritize creating dense regions
    # Use a heuristic: try all edges, add if it doesn't create Q3
    
    # Better approach: start from a known good structure and add edges
    # K3,13 structure: 3 vertices connected to all other 13, no edges among the 13 or among the 3
    # Actually let's try a different approach: simulated annealing from a random dense graph
    
    random.seed(42)
    
    # Multiple restarts with different strategies
    best_mask = 0
    best_edges = 0
    
    # Strategy 1: Greedy add from empty
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        random.shuffle(idx_edge)
        mask = 0
        for ei in idx_edge:
            new_mask = mask | (1 << ei)
            if is_q3_free(new_mask):
                mask = new_mask
        ecount = bin(mask).count('1')
        if ecount > best_edges:
            best_edges = ecount
            best_mask = mask
    
    # Strategy 2: Simulated annealing from dense random starts
    T_start = 2.0
    T_min = 0.05
    cooldown = 0.999
    
    for trial in range(20):
        if time.time() > deadline - 15:
            break
        
        # Random dense start
        p = 0.6 + 0.1 * random.random()
        mask = 0
        for ei in range(total_edges):
            if random.random() < p:
                mask |= (1 << ei)
        
        cur_violations = count_violations(mask)
        cur_edges = bin(mask).count('1')
        
        T = T_start
        best_in_run = mask if cur_violations == 0 else 0
        best_edges_in_run = cur_edges if cur_violations == 0 else 0
        
        while T > T_min and time.time() < deadline - 15:
            # Pick a random edge to flip
            ei = random.randrange(total_edges)
            new_mask = mask ^ (1 << ei)
            new_violations = count_violations(new_mask)
            new_edges = bin(new_mask).count('1')
            
            # Objective: maximize edges, minimize violations (heavy penalty)
            old_score = cur_edges - 10000 * cur_violations
            new_score = new_edges - 10000 * new_violations
            delta = new_score - old_score
            
            if delta >= 0 or random.random() < (2.718281828 ** (delta / max(T, 1e-10))):
                mask = new_mask
                cur_violations = new_violations
                cur_edges = new_edges
                
                if cur_violations == 0 and cur_edges > best_edges_in_run:
                    best_edges_in_run = cur_edges
                    best_in_run = mask
            
            T *= cooldown
        
        if best_edges_in_run > best_edges:
            best_edges = best_edges_in_run
            best_mask = best_in_run
    
    # Strategy 3: Local search / hill climbing on the best found
    if time.time() < deadline - 5:
        mask = best_mask
        cur_violations = count_violations(mask)
        cur_edges = bin(mask).count('1')
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            # Try adding each missing edge
            for ei in range(total_edges):
                if (mask >> ei) & 1:
                    continue
                new_mask = mask | (1 << ei)
                if count_violations(new_mask) == 0:
                    mask = new_mask
                    cur_edges += 1
                    improved = True
        
        if cur_edges > best_edges:
            best_edges = cur_edges
            best_mask = mask
    
    # Final verification and extraction
    result = []
    for ei in range(total_edges):
        if (best_mask >> ei) & 1:
            u, v = idx_edge[ei]
            result.append([u, v])
    
    return result
