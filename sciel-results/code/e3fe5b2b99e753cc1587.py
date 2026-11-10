# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import math


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a maximum-size square-difference-free set modulo 205 using an exact
    maximum independent set solver via branch-and-bound with bitset operations.
    
    The graph G has vertices 0..204, with an edge between a and b if (a-b) mod 205
    is a nonzero quadratic residue mod 205. We seek a maximum independent set in G.
    
    Returns:
        np.ndarray of shape (s,), dtype int64, containing the residues of the
        maximum (or near-maximum) independent set.
    """
    m = 205
    
    # Compute forbidden differences: nonzero quadratic residues mod 205
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Build adjacency list and bitsets
    # adj[i] is a Python int where bit j is set if i and j are adjacent (forbidden pair)
    adj = [0] * m
    for a in range(m):
        for b in range(a + 1, m):
            d = (a - b) % m
            if d in forbidden:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
    
    # All vertices bitmask
    all_vertices = (1 << m) - 1
    
    # Precompute degree for ordering
    degrees = [bin(adj[i]).count('1') for i in range(m)]
    
    # Order vertices by descending degree for better pruning
    order = sorted(range(m), key=lambda x: -degrees[x])
    
    # Map from ordered index to original vertex
    # We'll work in the original vertex space but use ordering for branching
    
    best_set = []
    best_size = 0
    
    # Branch and bound for maximum independent set
    # State: current independent set (list), candidates mask (bitset of vertices that can still be added)
    
    def max_clique_size(subset_mask):
        """
        Compute a lower bound on the clique number of the induced subgraph on subset_mask.
        This is an upper bound on the independent set size in the complement.
        Actually, we want an upper bound on the size of an independent set in the
        subgraph induced by subset_mask.
        
        A simple upper bound: |subset_mask| (trivial)
        Better: greedy coloring of the subgraph gives an upper bound on clique number
        in the complement graph, which equals the independence number in the original.
        
        Let's use a simple greedy approach: repeatedly pick a vertex, remove it and its neighbors.
        The number of picks is a lower bound on the independence number.
        For an upper bound, we can use: size of subset, or a greedy coloring bound.
        
        For simplicity and speed, let's use: min(|subset|, 1 + max over v in subset of (1 + max_clique_size(subset & ~adj[v] & ~(1<<v))))
        But that's the exact recursion. Let's just use |subset| as a trivial upper bound
        and rely on the branch-and-bound structure.
        
        Actually, a better upper bound: the size of a maximum matching + something, but that's complex.
        Let's use: the number of vertices in subset_mask divided by the minimum degree + 1, roughly.
        
        For now, let's use a simple greedy coloring upper bound on the complement graph.
        In the complement graph, an independent set is a clique.
        The chromatic number of the complement graph is an upper bound on the clique number.
        Greedy coloring gives an upper bound on chromatic number.
        """
        # Convert mask to list of vertices
        verts = []
        mm = subset_mask
        while mm:
            # Find lowest set bit
            lsb = mm & (-mm)
            v = lsb.bit_length() - 1
            verts.append(v)
            mm ^= lsb
        
        if not verts:
            return 0
        
        # Greedy coloring of the complement graph induced by verts
        # In the complement, two vertices are adjacent if they are NOT adjacent in original
        # i.e., if (a-b) mod m is NOT in forbidden (and a != b)
        # For coloring, we assign colors such that adjacent-in-complement have different colors.
        
        color_of = {}
        max_color = 0
        for v in verts:
            # Find available colors
            used_colors = set()
            for u in verts:
                if u == v:
                    continue
                # Check if u and v are adjacent in complement
                # They are adjacent in complement if NOT adjacent in original
                # adj[v] has bit u set if adjacent in original
                if not (adj[v] >> u) & 1:
                    # Adjacent in complement
                    if u in color_of:
                        used_colors.add(color_of[u])
            # Assign smallest available color
            c = 0
            while c in used_colors:
                c += 1
            color_of[v] = c
            max_color = max(max_color, c + 1)
        
        return max_color
    
    def branch_and_bound(current_set, candidates_mask, depth):
        nonlocal best_set, best_size
        
        # Upper bound: current size + upper bound on independent set in candidates
        if candidates_mask == 0:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
        
        # Trivial upper bound
        cand_count = bin(candidates_mask).count('1')
        if len(current_set) + cand_count <= best_size:
            return
        
        # Better upper bound using greedy coloring
        ub = max_clique_size(candidates_mask)
        if len(current_set) + ub <= best_size:
            return
        
        # Choose a vertex to branch on
        # Pick the vertex in candidates with highest degree within candidates
        mm = candidates_mask
        best_v = -1
        best_deg = -1
        while mm:
            lsb = mm & (-mm)
            v = lsb.bit_length() - 1
            # Degree of v within candidates
            deg = bin(candidates_mask & adj[v]).count('1')
            if deg > best_deg:
                best_deg = deg
                best_v = v
            mm ^= lsb
        
        if best_v == -1:
            return
        
        # Branch 1: Include best_v
        new_candidates = candidates_mask & ~adj[best_v] & ~(1 << best_v)
        current_set.append(best_v)
        branch_and_bound(current_set, new_candidates, depth + 1)
        current_set.pop()
        
        # Branch 2: Exclude best_v
        branch_and_bound(current_set, candidates_mask & ~(1 << best_v), depth + 1)
    
    # Initialize with an empty set and all vertices as candidates
    branch_and_bound([], all_vertices, 0)
    
    # If the exact search is too slow or finds nothing, fall back to greedy
    if best_size == 0:
        selected = []
        for candidate in range(m):
            ok = True
            for s in selected:
                d1 = (candidate - s) % m
                d2 = (s - candidate) % m
                if d1 in forbidden or d2 in forbidden:
                    ok = False
                    break
            if ok:
                selected.append(candidate)
        best_set = selected
    
    return np.array(best_set, dtype=np.int64)


# EVOLVE-BLOCK-END
