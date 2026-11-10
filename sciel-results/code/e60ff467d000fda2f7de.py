# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large square-difference-free set modulo 205.
    
    We model this as a maximum independent set problem on the graph where
    vertices are residues 0..204 and edges connect pairs with forbidden differences.
    We use a bitmask-based branch-and-bound approach with greedy coloring upper bounds
    to prune the search space effectively.
    
    Returns:
        np.ndarray of distinct residues in 0..204 with no two differing by a nonzero square mod 205.
    """
    m = 205
    
    # Compute forbidden differences: nonzero quadratic residues mod 205
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Build adjacency: adj[i] is a set of j such that (i-j) mod m is in forbidden
    # Since forbidden is symmetric (if d is forbidden, m-d is also forbidden),
    # the graph is undirected.
    adj = [set() for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            d = (i - j) % m
            if d in forbidden:
                adj[i].add(j)
                adj[j].add(i)
    
    # Convert adjacency to bitmasks for fast operations
    # adj_mask[i] is an integer bitmask where bit j is set if i and j are adjacent
    adj_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_mask[i] = mask
    
    # We want a maximum independent set in this graph.
    # Equivalent to maximum clique in the complement graph.
    # In the complement graph, vertices i and j are adjacent if they are NOT forbidden-differing.
    # comp_adj_mask[i] has bit j set if i and j are compatible (can both be in the set).
    
    comp_adj_mask = [0] * m
    for i in range(m):
        # All other vertices except i
        all_mask = (1 << m) - 1
        all_mask &= ~(1 << i)
        # Remove neighbors in original graph
        comp_adj_mask[i] = all_mask & ~adj_mask[i]
    
    # Branch and bound for maximum independent set
    # We'll use a recursive approach with bitmask representation of candidate vertices
    
    best_set = []  # The best independent set found so far
    best_size = 0
    
    # Greedy coloring for upper bound:
    # Given a set of candidates (bitmask), greedily color them to get an upper bound on clique size
    # In the complement graph, we want max clique, so we color the complement of the subgraph induced by candidates
    # Actually, for maximum independent set in G, it's maximum clique in complement G'.
    # Greedy coloring of G' gives upper bound on clique number.
    
    def greedy_color_upper_bound(candidates_mask):
        """
        Greedily color the complement graph induced by candidates_mask.
        Returns the number of colors used, which is an upper bound on the clique size
        (i.e., the independent set size in the original graph).
        """
        # Extract list of vertices
        verts = []
        temp = candidates_mask
        while temp:
            low_bit = temp & (-temp)
            v = low_bit.bit_length() - 1
            verts.append(v)
            temp ^= low_bit
        
        if not verts:
            return 0
        
        # Greedy coloring: assign smallest possible color to each vertex
        # In the complement graph, two vertices are adjacent if they are compatible
        # For coloring, we need to color such that adjacent vertices have different colors
        # In complement graph G', edge (u,v) exists if u,v are compatible in original
        # We want to color G' so that compatible vertices get different colors
        # Upper bound on clique = number of colors
        
        color_of = {}
        color_count = 0
        for v in verts:
            # Find colors used by neighbors of v in G'
            neighbor_colors = set()
            # Neighbors of v in G' are those u in candidates such that u is compatible with v
            # i.e., u is in comp_adj_mask[v] and u is in candidates
            neighbors_in_candidates = comp_adj_mask[v] & candidates_mask
            temp_neigh = neighbors_in_candidates
            while temp_neigh:
                low_bit = temp_neigh & (-temp_neigh)
                u = low_bit.bit_length() - 1
                temp_neigh ^= low_bit
                if u in color_of:
                    neighbor_colors.add(color_of[u])
            
            # Assign smallest available color
            c = 0
            while c in neighbor_colors:
                c += 1
            color_of[v] = c
            if c + 1 > color_count:
                color_count = c + 1
        
        return color_count
    
    # Branch and bound search
    def branch_and_bound(candidates_mask, current_set):
        nonlocal best_set, best_size
        
        if not candidates_mask:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
        
        # Upper bound: greedy coloring
        ub = greedy_color_upper_bound(candidates_mask)
        if len(current_set) + ub <= best_size:
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick vertex with highest degree in the induced subgraph
        # For maximum clique in complement, pick vertex with high degree in complement
        temp = candidates_mask
        best_v = -1
        best_degree = -1
        while temp:
            low_bit = temp & (-temp)
            v = low_bit.bit_length() - 1
            temp ^= low_bit
            degree = bin(comp_adj_mask[v] & candidates_mask).count('1')
            if degree > best_degree:
                best_degree = degree
                best_v = v
        
        if best_v == -1:
            return
        
        # Branch 1: include best_v
        new_candidates = candidates_mask & comp_adj_mask[best_v]
        # Remove best_v from candidates
        new_candidates &= ~(1 << best_v)
        branch_and_bound(new_candidates, current_set + [best_v])
        
        # Branch 2: exclude best_v
        new_candidates = candidates_mask & ~(1 << best_v)
        branch_and_bound(new_candidates, current_set)
    
    # Initial candidates: all vertices
    all_candidates = (1 << m) - 1
    branch_and_bound(all_candidates, [])
    
    return np.array(best_set, dtype=np.int64)


# EVOLVE-BLOCK-END
