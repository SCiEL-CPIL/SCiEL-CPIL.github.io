"""Spectral graph lifts: exhaustive gauge-fixed search with optimized radius computation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.science_tcs_expansion import check_lift

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 90s kill (60s task limit)
    
    m = graph['n'] // 2
    edges = graph['edges']
    num_edges = len(edges)
    n = 2 * m
    
    # Identify edge types
    bottom_cycle_edges = []  # edges within bottom cycle (0..m-1)
    top_cycle_edges = []     # edges within top cycle (m..2m-1)
    matching_edges = []      # matching edges (i, i+m)
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            bottom_cycle_edges.append(idx)
        elif u >= m and v >= m:
            top_cycle_edges.append(idx)
        else:
            matching_edges.append(idx)
    
    # Build spanning tree for gauge fixing
    # Tree: all m matching edges + m-1 bottom cycle edges
    # Chords: 1 bottom cycle edge + m top cycle edges
    tree_edge_set = set()
    for i in range(m):
        tree_edge_set.add((i, i + m))
        tree_edge_set.add((i + m, i))
    for i in range(m - 1):
        tree_edge_set.add((i, i + 1))
        tree_edge_set.add((i + 1, i))
    
    chord_indices = []
    chord_edge_pairs = []
    for idx, (u, v) in enumerate(edges):
        if (u, v) not in tree_edge_set:
            chord_indices.append(idx)
            chord_edge_pairs.append((u, v))
    
    num_chords = len(chord_indices)
    
    # Pre-build base adjacency with all +1
    base_adj = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        base_adj[u, v] = 1.0
        base_adj[v, u] = 1.0
    
    # Precompute which edges are chords for fast sign assignment
    chord_mask = np.zeros(num_edges, dtype=bool)
    for idx in chord_indices:
        chord_mask[idx] = True
    
    # Cache for check_lift results (optional, but let's just compute directly)
    # We'll compute spectral radius directly using eigvalsh on the signed adjacency matrix
    
    def compute_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        A = base_adj.copy()
        # Apply signs
        for idx in chord_indices:
            s = signs[idx]
            u, v = edges[idx]
            A[u, v] = s
            A[v, u] = s
        try:
            eigvals = np.linalg.eigvalsh(A)
            return float(np.max(np.abs(eigvals)))
        except Exception:
            return float('inf')
    
    def is_connected(signs):
        """Check if the lift is connected using check_lift."""
        try:
            radius, details = check_lift(graph, signs)
            return True, radius
        except Exception:
            return False, float('inf')
    
    best_signs = None
    best_radius = float('inf')
    
    # Enumerate all 2^num_chords assignments
    # For m=4: 5 chords -> 32 assignments
    # For m=5: 6 chords -> 64 assignments
    # For m=7: 8 chords -> 256 assignments
    # For m=9: 10 chords -> 1024 assignments
    total_assignments = 1 << num_chords
    
    for assignment in range(total_assignments):
        if time.time() > deadline:
            break
        
        # Build signs vector: tree edges are +1, chord edges depend on assignment
        signs = np.ones(num_edges, dtype=np.int8)
        for j, eidx in enumerate(chord_indices):
            if assignment & (1 << j):
                signs[eidx] = -1
        
        # Check validity and compute radius
        valid, radius = is_connected(signs)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: all +1
        best_signs = np.ones(num_edges, dtype=np.int8)
        best_radius = compute_radius(best_signs)
    
    # Local search refinement: try flipping individual edges (not just chords)
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for eidx in range(num_edges):
                trial_signs = best_signs.copy()
                trial_signs[eidx] = -trial_signs[eidx]
                valid, radius = is_connected(trial_signs)
                if valid and radius < best_radius - 1e-12:
                    best_radius = radius
                    best_signs = trial_signs
                    improved = True
    
    # Second pass: try flipping pairs of edges
    if time.time() < deadline - 10.0:
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            for i in range(num_edges):
                for j in range(i + 1, num_edges):
                    if time.time() > deadline - 10.0:
                        break
                    trial_signs = best_signs.copy()
                    trial_signs[i] = -trial_signs[i]
                    trial_signs[j] = -trial_signs[j]
                    valid, radius = is_connected(trial_signs)
                    if valid and radius < best_radius - 1e-12:
                        best_radius = radius
                        best_signs = trial_signs
                        improved = True
                        break
                if time.time() > deadline - 10.0:
                    break
    
    return best_signs
# EVOLVE-BLOCK-END
