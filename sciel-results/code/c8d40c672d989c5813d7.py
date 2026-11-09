"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

def construct():
    n = 20
    max_colors = 20
    
    # Precompute all 4-cycles in K_n
    # A 4-cycle is a set of 4 distinct vertices {a,b,c,d} with a specific cyclic order
    # For 4 vertices a<b<c<d, the 3 distinct 4-cycles are:
    #   a-b-c-d-a (edges: ab, bc, cd, da)
    #   a-b-d-c-a (edges: ab, bd, dc, ca)
    #   a-c-b-d-a (edges: ac, cb, bd, da)
    
    four_cycles = []  # list of (e1, e2, e3, e4) where each ei is (u,v) with u<v
    vertex_pairs = list(combinations(range(n), 2))
    pair_to_idx = {p: i for i, p in enumerate(vertex_pairs)}
    
    for a, b, c, d in combinations(range(n), 4):
        # Cycle 1: a-b-c-d-a
        e1 = (a, b)
        e2 = (b, c)
        e3 = (c, d)
        e4 = (d, a)
        four_cycles.append((pair_to_idx[e1], pair_to_idx[e2], pair_to_idx[e3], pair_to_idx[e4]))
        
        # Cycle 2: a-b-d-c-a
        e1 = (a, b)
        e2 = (b, d)
        e3 = (d, c)
        e4 = (c, a)
        four_cycles.append((pair_to_idx[e1], pair_to_idx[e2], pair_to_idx[e3], pair_to_idx[e4]))
        
        # Cycle 3: a-c-b-d-a
        e1 = (a, c)
        e2 = (c, b)
        e3 = (b, d)
        e4 = (d, a)
        four_cycles.append((pair_to_idx[e1], pair_to_idx[e2], pair_to_idx[e3], pair_to_idx[e4]))
    
    num_cycles = len(four_cycles)
    num_edges = len(vertex_pairs)
    
    # Build incidence matrix: for each cycle, which 4 edges it uses
    # cycle_edges[cycle_idx] = [edge_idx1, edge_idx2, edge_idx3, edge_idx4]
    cycle_edges = np.array(four_cycles, dtype=np.int32)
    
    # For each edge, which cycles does it belong to?
    edge_cycles = [[] for _ in range(num_edges)]
    for ci in range(num_cycles):
        for ei in cycle_edges[ci]:
            edge_cycles[ei].append(ci)
    
    # Convert to numpy arrays for speed
    edge_cycle_counts = np.array([len(c) for c in edge_cycles], dtype=np.int32)
    max_cycles_per_edge = max(len(c) for c in edge_cycles)
    
    # Pad edge_cycles to fixed size
    edge_cycles_padded = np.full((num_edges, max_cycles_per_edge), -1, dtype=np.int32)
    for ei in range(num_edges):
        for k, ci in enumerate(edge_cycles[ei]):
            edge_cycles_padded[ei, k] = ci
    
    # State: color_of_edge[e] = color (0..max_colors-1) or -1 if edge not present
    color_of_edge = np.full(num_edges, -1, dtype=np.int8)
    edge_present = np.zeros(num_edges, dtype=bool)
    
    # For each edge, track which colors are forbidden due to 4-cycle constraints
    # forbidden[e, c] = True if adding edge e with color c would create a non-rainbow 4-cycle
    forbidden = np.zeros((num_edges, max_colors), dtype=bool)
    
    def update_forbidden_after_add(edge_idx, color):
        """After adding edge edge_idx with color, update forbidden colors for other edges."""
        edge_present[edge_idx] = True
        color_of_edge[edge_idx] = color
        
        # For each cycle containing edge_idx, check the other 3 edges
        for ci in edge_cycles[edge_idx]:
            cycle = cycle_edges[ci]
            other_edges = [e for e in cycle if e != edge_idx]
            
            # Count how many of the other 3 edges are present
            present_count = 0
            present_colors = []
            for e in other_edges:
                if edge_present[e]:
                    present_count += 1
                    present_colors.append(color_of_edge[e])
            
            if present_count == 3:
                # All 4 edges present - this cycle must be rainbow
                # Check if it is (it should be, since we validated before adding)
                all_colors = [color] + present_colors
                assert len(set(all_colors)) == 4, f"Invalid 4-cycle created: {all_colors}"
            elif present_count == 2:
                # Two other edges present. If we add the 4th edge later, we need all 4 distinct.
                # For each of the 2 missing edges, if we add it with color c, we need
                # c != color_of_edge[edge_idx] and c != the two present colors
                # Actually, we need all 4 colors distinct, so c must differ from all 3 existing colors
                existing_colors = set([color] + present_colors)
                for e in other_edges:
                    if not edge_present[e]:
                        for c in existing_colors:
                            forbidden[e, c] = True
            elif present_count == 1:
                # One other edge present. If we add both remaining edges, need all 4 distinct.
                # For now, just note that the two missing edges can't share a color with each other
                # or with the two existing colors. But this is a constraint between two future edges,
                # which is harder to track. We'll handle this at addition time.
                pass
    
    def can_add_edge(edge_idx, color):
        """Check if we can add edge edge_idx with given color."""
        if edge_present[edge_idx]:
            return False
        
        # Check all cycles containing this edge
        for ci in edge_cycles[edge_idx]:
            cycle = cycle_edges[ci]
            other_edges = [e for e in cycle if e != edge_idx]
            
            # Collect colors of present edges in this cycle
            present_colors = [color]  # the new edge's color
            for e in other_edges:
                if edge_present[e]:
                    present_colors.append(color_of_edge[e])
            
            # If all 4 edges are present (3 others + new), all colors must be distinct
            num_present_others = sum(1 for e in other_edges if edge_present[e])
            if num_present_others == 3:
                if len(set(present_colors)) != 4:
                    return False
            elif num_present_others == 2:
                # Two others present. The 4th edge (not yet added) will need a color distinct
                # from all 4. This doesn't block us now, but we should mark constraints.
                # For now, just check that the 3 present colors (including new) are distinct
                if len(set(present_colors)) != 3:
                    return False
            elif num_present_others == 1:
                # One other present. Check that new color != that other's color
                if len(set(present_colors)) != 2:
                    return False
            # num_present_others == 0: no constraint yet
        
        return True
    
    # Greedy edge addition with random ordering
    import random
    random.seed(123)
    
    edge_order = list(range(num_edges))
    random.shuffle(edge_order)
    
    edges_added = 0
    for edge_idx in edge_order:
        # Try each color
        colors_to_try = list(range(max_colors))
        random.shuffle(colors_to_try)
        for c in colors_to_try:
            if not forbidden[edge_idx, c]:
                if can_add_edge(edge_idx, c):
                    update_forbidden_after_add(edge_idx, c)
                    edges_added += 1
                    break
    
    # Build result
    result = []
    for ei, (u, v) in enumerate(vertex_pairs):
        if edge_present[ei]:
            result.append([u, v, int(color_of_edge[ei])])
    
    return result
# EVOLVE-BLOCK-END
