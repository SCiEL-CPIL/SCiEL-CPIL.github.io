"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

D = 11

# EVOLVE-BLOCK-START

def _build_supports():
    """Generate all 4-element subsets of {0,...,10} as bitmasks."""
    supports = []
    indices = range(D)
    for comb in combinations(indices, 4):
        mask = 0
        for idx in comb:
            mask |= (1 << idx)
        supports.append(mask)
    return supports

def _max_independent_set(adj, n, deadline=None):
    """
    Find a large independent set in a graph defined by adjacency lists.
    Uses a branch-and-bound approach with greedy coloring upper bound.
    """
    best_set = set()
    
    # Convert adjacency to sets for fast lookup
    adj_sets = [set(neighbors) for neighbors in adj]
    
    # Greedy coloring for upper bound
    def greedy_coloring(vertices):
        """
        Returns a list of colors for the given vertices.
        The maximum color used is an upper bound on the size of the MIS in the subgraph induced by these vertices.
        """
        if not vertices:
            return 0, {}
        
        colors = {}
        v_list = list(vertices)
        # Order vertices by degree in the subgraph (descending)
        # This helps coloring be tighter
        degrees = {}
        for v in v_list:
            deg = 0
            for u in v_list:
                if u != v and u in adj_sets[v]:
                    deg += 1
            degrees[v] = deg
            
        v_list.sort(key=lambda x: degrees[x], reverse=True)
        
        for v in v_list:
            # Find smallest available color
            used_colors = set()
            for u in v_list:
                if u != v and u in adj_sets[v] and u in colors:
                    used_colors.add(colors[u])
            
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            
        if colors:
            return max(colors.values()) + 1, colors
        else:
            return 0, colors

    # Branch and bound
    def bb(vertices, current_set, depth=0):
        nonlocal best_set
        
        if deadline and time.time() > deadline:
            return
            
        if not vertices:
            if len(current_set) > len(best_set):
                best_set = set(current_set)
            return
            
        # Upper bound
        ub, _ = greedy_coloring(vertices)
        
        if len(current_set) + ub <= len(best_set):
            return
            
        # Choose a vertex to branch on
        # Heuristic: pick vertex with highest degree in the remaining subgraph
        v = max(vertices, key=lambda x: len(adj_sets[x].intersection(vertices)))
        
        # Branch 1: Include v
        new_current = set(current_set)
        new_current.add(v)
        # Remove v and its neighbors from consideration
        new_vertices = vertices - adj_sets[v] - {v}
        bb(new_vertices, new_current, depth + 1)
        
        # Branch 2: Exclude v
        new_vertices = vertices - {v}
        bb(new_vertices, current_set, depth + 1)

    # Initialize with a greedy solution
    alive = set(range(n))
    greedy_set = set()
    while alive:
        # Pick node with min degree
        v = min(alive, key=lambda u: len(adj_sets[u].intersection(alive)))
        greedy_set.add(v)
        alive.difference_update(adj_sets[v])
        alive.remove(v)
    
    best_set = greedy_set
    all_vertices = set(range(n))
    bb(all_vertices, set(), 0)
    
    return best_set

def construct():
    deadline = time.time() + 110
    
    # 1. Generate supports
    supports = _build_supports()
    n_sup = len(supports)
    
    # 2. Build conflict graph
    # Conflict if intersection size is 3
    adj = [set() for _ in range(n_sup)]
    for i in range(n_sup):
        for j in range(i + 1, n_sup):
            inter = supports[i] & supports[j]
            if inter.bit_count() == 3:
                adj[i].add(j)
                adj[j].add(i)
    
    # 3. Find Maximum Independent Set (MIS) of supports
    # This ensures that any two selected supports share at most 2 indices.
    # This is the key structural change: using a proper B&B MIS solver instead of WalkSAT/Heuristics.
    mis_supports = _max_independent_set(adj, n_sup, deadline - 5)
    
    # 4. Generate vectors for the MIS supports
    # For each support, we can add all 16 sign variations.
    # As analyzed, all sign variations of a single support are mutually compatible.
    # And since supports in the MIS share at most 2 indices, vectors from different supports are also compatible.
    # Let's verify compatibility between different supports S1, S2 with |S1 \cap S2| <= 2.
    # Let u in S1, v in S2.
    # u.v = sum_{k in S1 \cap S2} u_k v_k.
    # Max possible dot product: if |S1 \cap S2| = 2, max dot is 2 (if signs match).
    # Dist^2 = ||u||^2 + ||v||^2 - 2u.v = 4 + 4 - 2(2) = 4.
    # This is >= max_norm^2 (which is 4). So it's valid.
    # If |S1 \cap S2| = 1, max dot is 1. Dist^2 = 8 - 2 = 6. Valid.
    # If |S1 \cap S2| = 0, max dot is 0. Dist^2 = 8. Valid.
    
    vectors = []
    for idx in mis_supports:
        mask = supports[idx]
        inds = [i for i in range(D) if (mask >> i) & 1]
        
        # Add all 16 sign variations
        for sign_mask in range(16):
            v = np.zeros(D, dtype=np.int64)
            for j, idx_val in enumerate(inds):
                if sign_mask & (1 << j):
                    v[idx_val] = 1
                else:
                    v[idx_val] = -1
            vectors.append(v)
            
    S4 = np.array(vectors, dtype=np.int64)
    
    # 5. Try to add Shell 2 vectors (weight 2)
    # Shell 2 vectors are of the form e_i - e_j. Norm^2 = 2.
    # We need to check compatibility with S4 and among themselves.
    
    shell2_vecs = []
    for i in range(D):
        for j in range(i+1, D):
            v1 = np.zeros(D, dtype=np.int64)
            v1[i] = 1; v1[j] = -1
            shell2_vecs.append(v1)
            v2 = np.zeros(D, dtype=np.int64)
            v2[i] = -1; v2[j] = 1
            shell2_vecs.append(v2)
    shell2 = np.array(shell2_vecs, dtype=np.int64)
    
    if len(S4) > 0:
        # Check compatibility of shell2 with S4
        # Norm shell2 = sqrt(2). Norm S4 = 2.
        # Dist^2 >= 4.
        # 2 + 4 - 2u.v >= 4 => u.v <= 1.
        
        dots_24 = shell2 @ S4.T
        max_ip_24 = dots_24.max(axis=1)
        valid_2_mask = max_ip_24 <= 1
        S2_candidates = shell2[valid_2_mask]
        
        n2 = len(S2_candidates)
        if n2 > 0:
            # Now select a subset of S2_candidates that are mutually compatible.
            # Norm S2 = sqrt(2).
            # Dist^2 >= 4.
            # 2 + 2 - 2u.v >= 4 => u.v <= 0.
            
            dots_22 = S2_candidates @ S2_candidates.T
            # We want a subset where off-diagonal entries are <= 0.
            # This is a Maximum Independent Set on the graph where edge exists if dot > 0.
            
            adj2 = [set() for _ in range(n2)]
            for i in range(n2):
                for j in range(i+1, n2):
                    if dots_22[i, j] > 0:
                        adj2[i].add(j)
                        adj2[j].add(i)
                        
            # Greedy MIS for S2
            alive2 = set(range(n2))
            selected2 = set()
            while alive2:
                v_idx = min(alive2, key=lambda u: len(adj2[u].intersection(alive2)))
                selected2.add(v_idx)
                alive2.difference_update(adj2[v_idx])
                alive2.remove(v_idx)
                
            S2 = S2_candidates[list(selected2)]
            result = np.vstack([S4, S2])
        else:
            result = S4
    else:
        result = S4
        
    return result

# EVOLVE-BLOCK-END
