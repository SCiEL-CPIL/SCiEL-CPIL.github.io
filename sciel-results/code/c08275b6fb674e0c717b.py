import numpy as np
import time
import sys
from itertools import combinations, product

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Generate candidate vectors with squared norm 4
    # Type A: four +/-1 entries (weight 4)
    # Type B: one +/-2 entry (weight 1, value 2)
    
    all_vecs = []
    
    # Type A: 4 non-zero entries of +/-1
    supports_4 = list(combinations(range(D), 4))
    for s in supports_4:
        for signs in product([1, -1], repeat=4):
            v = [0] * D
            for i, si in zip(s, signs):
                v[i] = si
            all_vecs.append(tuple(v))
    
    # Type B: one entry of +/-2
    for i in range(D):
        for s in [2, -2]:
            v = [0] * D
            v[i] = s
            all_vecs.append(tuple(v))
    
    n = len(all_vecs)
    
    # Build compatibility graph using bitsets
    # Two vectors are compatible if their dot product <= 2
    # (since we need distance >= max_norm, and max_norm_sq = 4, so dist_sq >= 4)
    # dist_sq = |u|^2 + |v|^2 - 2*u.v = 4 + 4 - 2*u.v = 8 - 2*u.v
    # We need dist_sq >= 4 => 8 - 2*u.v >= 4 => 2*u.v <= 4 => u.v <= 2
    
    adj = [0] * n  # adj[i] has bit j set if i and j are compatible (edge in compatibility graph)
    
    # For maximum clique, we want a set of mutually compatible vectors
    # So we search for maximum clique in the compatibility graph
    
    # Precompute dot products efficiently
    # Convert to numpy array for vectorized dot products
    vecs_np = np.array(all_vecs, dtype=np.int64)
    
    # Build adjacency by checking dot products
    # n is at most 5280 + 22 = 5302
    # n^2/2 = ~14 million pairs, which is feasible with numpy
    
    # Compute all pairwise dot products in chunks to avoid memory issues
    chunk_size = 512
    for i in range(0, n, chunk_size):
        if time.time() > deadline:
            break
        i_end = min(i + chunk_size, n)
        chunk = vecs_np[i:i_end]  # (chunk_len, D)
        # Dot with all vectors
        dots = chunk @ vecs_np.T  # (chunk_len, n)
        
        for local_i in range(i_end - i):
            global_i = i + local_i
            row = dots[local_i]
            # Compatible if dot <= 2
            compatible = row <= 2
            # Exclude self
            compatible[global_i] = False
            # Set bits
            for j in range(n):
                if compatible[j]:
                    adj[global_i] |= (1 << j)
    
    # Now find maximum clique using branch and bound with greedy coloring
    # Greedy coloring for upper bound
    
    def greedy_coloring(vertices, adj_list):
        """Color the subgraph induced by vertices. Returns list of colors and max color."""
        color = {}
        for v in vertices:
            # Find smallest color not used by colored neighbors
            used = set()
            for u in adj_list[v]:
                if u in color:
                    used.add(color[u])
            c = 0
            while c in used:
                c += 1
            color[v] = c
        max_color = max(color.values()) + 1 if color else 0
        return color, max_color
    
    # Maximum clique search
    best_clique = set()
    
    def expand(clique, candidates, adj_bits):
        """Branch and bound maximum clique search."""
        nonlocal best_clique
        if time.time() > deadline:
            return
        
        if not candidates:
            if len(clique) > len(best_clique):
                best_clique = set(clique)
            return
        
        # Get upper bound: len(clique) + len(candidates)
        if len(clique) + len(candidates) <= len(best_clique):
            return
        
        # Order candidates by degree in candidate subgraph (descending)
        cand_list = list(candidates)
        cand_set = set(candidates)
        
        # Greedy coloring for upper bound
        # Color the candidate subgraph
        color_map = {}
        max_color = 0
        for v in cand_list:
            used_colors = set()
            for u in cand_list:
                if u == v:
                    continue
                if u in color_map:
                    # Check if v and u are adjacent
                    if (adj_bits[v] >> u) & 1:
                        used_colors.add(color_map[u])
            c = 0
            while c in used_colors:
                c += 1
            color_map[v] = c
            if c > max_color:
                max_color = c
        max_color += 1
        
        if len(clique) + max_color <= len(best_clique):
            return
        
        # Process vertices in reverse color order
        for v in reversed(cand_list):
            if time.time() > deadline:
                return
            if v not in cand_set:
                continue
            
            # Upper bound with this vertex's color
            if len(clique) + color_map[v] <= len(best_clique):
                continue
            
            # Include v
            new_clique = clique | {v}
            # New candidates: neighbors of v in current candidates
            new_candidates = set()
            for u in cand_set:
                if u == v:
                    continue
                if (adj_bits[v] >> u) & 1:
                    new_candidates.add(u)
            
            expand(new_clique, new_candidates, adj_bits)
            
            # Exclude v
            cand_set.discard(v)
            
            # Update upper bound
            remaining_max_color = max((color_map[u] for u in cand_set), default=0) + 1
            if len(clique) + remaining_max_color <= len(best_clique):
                return
    
    # Start with all vertices
    all_vertices = set(range(n))
    expand(set(), all_vertices, adj)
    
    if time.time() > deadline:
        # Use best found so far
        pass
    
    if not best_clique:
        # Fallback: return a single vector
        return np.array([all_vecs[0]], dtype=np.int64)
    
    # Convert clique to array
    res = np.array([all_vecs[i] for i in sorted(best_clique)], dtype=np.int64)
    return res
