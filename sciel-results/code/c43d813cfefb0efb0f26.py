"""Few edges in 5-uniform hypergraphs without Property B (m(5)).

Strategy: Fast vectorized greedy set cover to find a non-2-colorable 5-uniform
hypergraph with minimum edges, followed by local search reduction.

The problem is a set cover: universe = all 2^n colorings, each 5-edge covers
the colorings where it's monochromatic. We need to cover all colorings.
"""

import time
import random
import sys
from itertools import combinations
import numpy as np

sys.setrecursionlimit(100000)


def _is_2_colorable(n, edges):
    """Exact check (backtracking + unit propagation) for a proper 2-colouring.
    
    Returns True iff the hypergraph IS 2-colorable (i.e., has a proper 2-coloring).
    """
    if not edges:
        return True
    
    m = len(edges)
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    all_vars = (1 << n) - 1
    
    def backtrack(assign, unassigned):
        # Unit propagation
        changed = True
        while changed:
            changed = False
            for mask in edge_masks:
                assigned_in_edge = mask & ~unassigned
                unassigned_in_edge = mask & unassigned
                
                if unassigned_in_edge == 0:
                    # All vertices in this edge are assigned.
                    if (assign & mask) == mask or (assign & mask) == 0:
                        return None  # contradiction: monochromatic edge found
                else:
                    assigned_zero = assigned_in_edge & ~assign
                    assigned_one = assigned_in_edge & assign
                    unassigned_count = bin(unassigned_in_edge).count('1')
                    
                    # Clause 1: at least one vertex in edge is 1
                    if assigned_one == 0 and unassigned_count == 1:
                        v = unassigned_in_edge.bit_length() - 1
                        assign |= (1 << v)
                        unassigned &= ~(1 << v)
                        changed = True
                        continue
                    
                    # Clause 2: at least one vertex in edge is 0
                    if assigned_zero == 0 and unassigned_count == 1:
                        v = unassigned_in_edge.bit_length() - 1
                        assign &= ~(1 << v)
                        unassigned &= ~(1 << v)
                        changed = True
                        continue
        
        if unassigned == 0:
            for mask in edge_masks:
                if (assign & mask) == mask or (assign & mask) == 0:
                    return None
            return assign
        
        # Branch on an unassigned variable
        v = unassigned.bit_length() - 1
        res = backtrack(assign | (1 << v), unassigned & ~(1 << v))
        if res is not None:
            return res
        return backtrack(assign & ~(1 << v), unassigned & ~(1 << v))
    
    return backtrack(0, all_vars) is not None


def _complete_5_graph(n):
    """Return the complete 5-uniform hypergraph on n vertices."""
    return [list(e) for e in combinations(range(n), 5)]


def _compute_edge_cover_sets_fast(n, all_edges):
    """Vectorized computation of which colorings each edge makes monochromatic.
    
    Returns: numpy array of shape (num_edges, num_colorings) with 1 if edge is
    monochromatic under that coloring, else 0.
    """
    num_edges = len(all_edges)
    num_colorings = 1 << n
    
    # Convert all_edges to a numpy array for vectorization
    edge_arr = np.array(all_edges, dtype=np.int32)  # shape (num_edges, 5)
    
    # All colorings as a numpy array
    colorings = np.arange(num_colorings, dtype=np.int32)
    
    # For each edge and each coloring, check if monochromatic
    # Edge e with vertices v0..v4 is monochromatic under coloring c if
    # all bits at positions v0..v4 in c are the same.
    
    # Extract bits for each vertex position
    # bits[e, c, k] = (coloring[c] >> edge_arr[e,k]) & 1
    # Shape: (num_edges, num_colorings, 5)
    
    # Vectorized: for each vertex position k in the edge
    bits = np.zeros((num_edges, num_colorings, 5), dtype=np.int8)
    for k in range(5):
        # For each edge, get the vertex index
        # For each coloring, extract that bit
        # bits[:, :, k] = (colorings >> edge_arr[:, k]) & 1
        # But edge_arr[:, k] has different values for different edges
        # So we need: bits[e, c, k] = (colorings[c] >> edge_arr[e, k]) & 1
        
        # Reshape for broadcasting
        # colorings has shape (num_colorings,)
        # edge_arr[:, k] has shape (num_edges,)
        # We want (num_edges, num_colorings)
        
        # (colorings[None, :] >> edge_arr[:, k][:, None]) & 1
        bits[:, :, k] = (colorings[None, :] >> edge_arr[:, k][:, None]) & 1
    
    # Edge is monochromatic if all 5 bits are 0 or all 5 bits are 1
    # all_zero = (bits.sum(axis=2) == 0)
    # all_one = (bits.sum(axis=2) == 5)
    bit_sums = bits.sum(axis=2)
    mono = ((bit_sums == 0) | (bit_sums == 5)).astype(np.int8)
    
    return mono  # shape (num_edges, num_colorings)


def _greedy_set_cover_fast(n, all_edges, mono_matrix, deadline, seed=42):
    """Fast greedy set cover using vectorized operations.
    
    mono_matrix: (num_edges, num_colorings) binary matrix
    Returns list of edge indices or None.
    """
    rng = random.Random(seed)
    num_edges = len(all_edges)
    num_colorings = 1 << n
    
    # Uncovered colorings as a boolean vector
    uncovered = np.ones(num_colorings, dtype=bool)
    selected = []
    
    while uncovered.any():
        if time.time() > deadline:
            return None
        
        # Compute coverage of each edge on uncovered colorings
        # coverage[e] = number of uncovered colorings that edge e covers
        # = mono_matrix[e] & uncovered, then sum
        coverage = (mono_matrix[:, uncovered] if False else  # can't index like this easily
                    (mono_matrix * uncovered[None, :]).sum(axis=1))
        
        # Find best edge
        best_idx = int(np.argmax(coverage))
        if coverage[best_idx] == 0:
            return None  # No progress possible
        
        selected.append(best_idx)
        uncovered[mono_matrix[best_idx] > 0] = False
    
    return selected


def _local_search_reduce(n, edges, mono_matrix, deadline):
    """Try to remove edges while maintaining non-2-colorability.
    
    An edge can be removed if its monochromatic colorings are all covered by other edges.
    """
    selected = list(range(len(edges)))
    num_edges = len(edges)
    num_colorings = 1 << n
    
    # Covered colorings for each selected edge
    # covered_mask[c] = number of selected edges that cover coloring c
    cover_count = np.zeros(num_colorings, dtype=np.int32)
    for e in selected:
        cover_count[mono_matrix[e] > 0] += 1
    
    # Try removing each edge
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            e = selected[i]
            # If we remove edge e, colorings only covered by e become uncovered
            # Those are colorings where cover_count == 1 and mono_matrix[e] > 0
            removable = True
            for c in range(num_colorings):
                if mono_matrix[e, c] > 0 and cover_count[c] == 1:
                    removable = False
                    break
            
            if removable:
                # Remove edge e
                cover_count[mono_matrix[e] > 0] -= 1
                selected.pop(i)
                changed = True
                break  # Restart from beginning after a removal
    
    return [edges[i] for i in selected]


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    n = 11
    all_edges = list(combinations(range(n), 5))
    num_edges = len(all_edges)
    
    # Fallback: complete 5-graph on 9 vertices
    fallback = {'n': 9, 'edges': _complete_5_graph(9)}
    
    # Compute monochromatic coverage matrix
    try:
        mono_matrix = _compute_edge_cover_sets_fast(n, all_edges)
    except Exception:
        return fallback
    
    # Fast greedy set cover
    selected_indices = _greedy_set_cover_fast(n, all_edges, mono_matrix, deadline)
    
    if selected_indices is not None:
        result_edges = [list(all_edges[i]) for i in selected_indices]
        
        # Verify non-2-colorability
        if not _is_2_colorable(n, result_edges):
            # Try local search reduction
            try:
                reduced = _local_search_reduce(n, result_edges, mono_matrix, deadline)
                if reduced and not _is_2_colorable(n, reduced):
                    return {'n': n, 'edges': reduced}
            except Exception:
                pass
            return {'n': n, 'edges': result_edges}
    
    # Try with different seeds (greedy is deterministic here, but let's try n=10 too)
    for trial_n in [10, 12]:
        if time.time() > deadline:
            break
        trial_edges = list(combinations(range(trial_n), 5))
        try:
            trial_mono = _compute_edge_cover_sets_fast(trial_n, trial_edges)
            trial_selected = _greedy_set_cover_fast(trial_n, trial_edges, trial_mono, deadline)
            if trial_selected is not None:
                trial_result = [list(trial_edges[i]) for i in trial_selected]
                if not _is_2_colorable(trial_n, trial_result):
                    try:
                        reduced = _local_search_reduce(trial_n, trial_result, trial_mono, deadline)
                        if reduced and not _is_2_colorable(trial_n, reduced):
                            return {'n': trial_n, 'edges': reduced}
                    except Exception:
                        pass
                    return {'n': trial_n, 'edges': trial_result}
        except Exception:
            pass
    
    # Return fallback
    return fallback
