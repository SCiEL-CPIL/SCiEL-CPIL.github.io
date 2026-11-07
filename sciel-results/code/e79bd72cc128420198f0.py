"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Build edge lookup: map (u,v) -> index in edges list
    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i
    
    # Identify the structure of prism graph edges:
    # Two m-cycles: vertices 0..m-1 (top) and m..2m-1 (bottom)
    # Matching edges: (i, i+m) for i in 0..m-1
    #
    # We'll fix a spanning tree to +1 and only search over non-tree edges.
    # 
    # A spanning tree for the prism graph:
    # - Fix matching edges (i, i+m) for i = 0..m-1 to +1 (these connect the two cycles)
    # - Fix one cycle, say the top cycle edges (i, i+1 mod m), to +1
    # - The remaining edges are the bottom cycle edges (m+i, m+i+1 mod m)
    #
    # For a 2-lift to be connected, we need the signed cycle conditions to work out.
    # The gauge freedom: fixing signs of a spanning tree to +1 means we only need
    # to assign signs to the remaining (chord) edges.
    #
    # Let's identify which edges form a spanning tree:
    # - All matching edges (i, i+m): m edges
    # - Top cycle edges: (0,1), (1,2), ..., (m-2, m-1): m-1 edges
    # Total tree edges: m + (m-1) = 2m - 1 = n - 1. Good, that's a spanning tree.
    #
    # Non-tree edges (chords):
    # - Bottom cycle edges: (m, m+1), (m+1, m+2), ..., (m+m-2, m+m-1): m-1 edges
    # - Plus the "closing" edge of the top cycle: (m-1, 0): 1 edge
    # Total chords: m edges
    #
    # Wait, let me recount. Total edges in prism graph = 3m (two cycles of m edges each + m matching edges).
    # Spanning tree has 2m-1 edges. So chords = 3m - (2m-1) = m+1 edges.
    #
    # Let me re-identify:
    # Tree edges:
    #   Matching: (i, i+m) for i=0..m-1: m edges
    #   Top cycle: (i, i+1) for i=0..m-2: m-1 edges (not including (m-1, 0))
    #   Total: 2m-1 edges
    #
    # Chord edges:
    #   Top cycle closing: (m-1, 0): 1 edge
    #   Bottom cycle: (m+i, m+i+1) for i=0..m-1: m edges
    #   Total: m+1 edges
    #
    # So we need to enumerate 2^(m+1) sign assignments for the chords.
    # For m=9, that's 2^10 = 1024 assignments. Very fast.
    
    # Find indices of chord edges
    chord_indices = []
    tree_edge_set = set()
    
    for i, (u, v) in enumerate(edges):
        is_matching = abs(u - v) == m
        if is_matching:
            tree_edge_set.add(i)
        else:
            # It's a cycle edge
            top = min(u, v) < m
            # Check if it's in the top cycle (excluding the closing edge)
            if top:
                # Top cycle edge: (i, i+1) mod m
                # This is a tree edge unless it's (m-1, 0)
                if (u == m-1 and v == 0) or (u == 0 and v == m-1):
                    chord_indices.append(i)
                else:
                    tree_edge_set.add(i)
            else:
                # Bottom cycle edge - all are chords
                chord_indices.append(i)
    
    num_chords = len(chord_indices)
    
    # For each chord assignment, build the full sign vector and check validity + spectral radius
    best_radius = float('inf')
    best_signs = None
    
    # Precompute: for each chord assignment, the sign vector
    for mask in range(1 << num_chords):
        if time.time() > deadline:
            break
        
        signs = np.ones(num_edges, dtype=np.int8)  # default +1 for tree edges
        
        for j, idx in enumerate(chord_indices):
            if mask & (1 << j):
                signs[idx] = -1
            else:
                signs[idx] = +1
        
        # Check validity using check_lift
        try:
            radius, details = check_lift(graph, signs)
        except Exception:
            continue
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: try the baseline
        from autoresearch_suite.science_tcs_expansion import lift_baseline
        return lift_baseline(graph, budget)
    
    # If we have time, do a local search to potentially improve
    if time.time() < deadline - 5.0:
        # Local search: flip individual signs and check if it improves
        current_signs = best_signs.copy()
        current_radius = best_radius
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(num_edges):
                if time.time() > deadline - 2.0:
                    break
                # Try flipping sign i
                trial_signs = current_signs.copy()
                trial_signs[i] = -trial_signs[i]
                try:
                    trial_radius, _ = check_lift(graph, trial_signs)
                    if trial_radius < current_radius - 1e-10:
                        current_signs = trial_signs
                        current_radius = trial_radius
                        improved = True
                except Exception:
                    pass
        best_signs = current_signs
    
    return best_signs
# EVOLVE-BLOCK-END
