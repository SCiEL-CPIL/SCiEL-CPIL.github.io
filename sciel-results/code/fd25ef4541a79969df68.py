"""Few edges in 5-uniform hypergraphs without Property B (m(5)).

Strategy: Use MILP (scipy.optimize.milp) to find a minimum edge set that makes
every 2-coloring have at least one monochromatic edge. The set cover formulation
over all 2^n colorings is the primary approach. We try multiple vertex counts
and use local search to reduce edge count after MILP.
"""

import time
import random
from itertools import combinations
import numpy as np


def _is_2_colorable(n, edges):
    """Exact check (backtracking + unit propagation) for a proper 2-colouring.
    
    Returns True iff the hypergraph IS 2-colorable (i.e., has a proper 2-coloring).
    """
    if not edges:
        return True
    
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    all_vars = (1 << n) - 1
    
    def backtrack(assign, unassigned):
        changed = True
        while changed:
            changed = False
            for mask in edge_masks:
                assigned_in_edge = mask & ~unassigned
                unassigned_in_edge = mask & unassigned
                
                if unassigned_in_edge == 0:
                    if (assign & mask) == mask or (assign & mask) == 0:
                        return None
                else:
                    assigned_zero = assigned_in_edge & ~assign
                    assigned_one = assigned_in_edge & assign
                    unassigned_count = bin(unassigned_in_edge).count('1')
                    
                    if assigned_one == 0 and unassigned_count == 1:
                        v = unassigned_in_edge.bit_length() - 1
                        assign |= (1 << v)
                        unassigned &= ~(1 << v)
                        changed = True
                        continue
                    
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
        
        # Choose variable with highest degree in unassigned edges
        v = unassigned.bit_length() - 1
        res = backtrack(assign | (1 << v), unassigned & ~(1 << v))
        if res is not None:
            return res
        return backtrack(assign & ~(1 << v), unassigned & ~(1 << v))
    
    return backtrack(0, all_vars) is not None


def _build_constraint_matrix(n):
    """Build the constraint matrix A where A[i,j]=1 iff edge j is monochromatic
    under coloring i. Also return the list of all 5-edges."""
    all_edges = list(combinations(range(n), 5))
    num_colorings = 1 << n
    num_edges = len(all_edges)
    
    edge_masks = []
    for e in all_edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    A = np.zeros((num_colorings, num_edges), dtype=np.int8)
    for i in range(num_colorings):
        for j in range(num_edges):
            mask = edge_masks[j]
            if (i & mask) == 0 or (i & mask) == mask:
                A[i, j] = 1
    
    return all_edges, A


def _try_milp(n, deadline=None):
    """Try to solve the set cover ILP using scipy.optimize.milp."""
    if deadline is None:
        deadline = time.time() + 130.0
    
    all_edges, A = _build_constraint_matrix(n)
    num_edges = len(all_edges)
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        c = np.ones(num_edges, dtype=np.float64)
        constraint = LinearConstraint(A, 1, np.inf)
        bounds = Bounds(0, 1)
        integrality = np.ones(num_edges, dtype=np.int8)
        
        res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if res.status == 0 and res.x is not None:
            x = np.round(res.x).astype(int)
            selected_indices = np.where(x > 0)[0]
            result = [list(all_edges[i]) for i in selected_indices]
            
            if not _is_2_colorable(n, result):
                return result
    except Exception:
        pass
    
    return None


def _local_search_reduce(n, edges, deadline=None, max_iters=1000):
    """Try to remove edges one at a time while maintaining non-2-colorability."""
    if deadline is None:
        deadline = time.time() + 130.0
    
    edges = list(edges)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(edges)):
            if time.time() > deadline:
                break
            trial = edges[:i] + edges[i+1:]
            if not _is_2_colorable(n, trial):
                edges = trial
                improved = True
                break
    
    return edges


def _greedy_set_cover(n, deadline=None):
    """Greedy set cover approach to find a non-2-colorable 5-uniform hypergraph."""
    if deadline is None:
        deadline = time.time() + 130.0
    
    all_edges, A = _build_constraint_matrix(n)
    num_colorings = 1 << n
    num_edges = len(all_edges)
    
    # Convert A columns to Python int bitsets for fast coverage computation
    edge_bitsets = []
    for j in range(num_edges):
        col = A[:, j]
        bitset = 0
        for i in range(num_colorings):
            if col[i]:
                bitset |= (1 << i)
        edge_bitsets.append(bitset)
    
    universe = (1 << num_colorings) - 1
    covered = 0
    selected_edges = []
    uncovered = universe
    
    # Sort edges by coverage size (descending)
    edge_indices = sorted(range(num_edges), key=lambda i: bin(edge_bitsets[i]).count('1'), reverse=True)
    selected_set = set()
    
    while uncovered != 0 and time.time() < deadline:
        best_idx = -1
        best_count = 0
        for idx in edge_indices:
            if idx in selected_set:
                continue
            coverage = edge_bitsets[idx] & uncovered
            count = bin(coverage).count('1')
            if count > best_count:
                best_count = count
                best_idx = idx
        
        if best_idx == -1 or best_count == 0:
            break
        
        selected_edges.append(best_idx)
        selected_set.add(best_idx)
        covered |= edge_bitsets[best_idx]
        uncovered = universe & ~covered
    
    if uncovered == 0:
        result = [list(all_edges[i]) for i in selected_edges]
        return result
    else:
        return None


def _complete_5_graph(n):
    """Return the complete 5-uniform hypergraph on n vertices."""
    return [list(e) for e in combinations(range(n), 5)]


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Fallback: complete 5-graph on 9 vertices (126 edges, definitely non-2-colorable)
    fallback_n = 9
    fallback_edges = _complete_5_graph(fallback_n)
    best_result = {'n': fallback_n, 'edges': fallback_edges}
    best_count = len(fallback_edges)
    
    # Try MILP for different n values, starting with smaller n
    for n in [11, 12, 10, 13]:
        if time.time() > deadline:
            break
        
        result = _try_milp(n, deadline=deadline)
        if result is not None:
            # Try to reduce further with local search
            reduced = _local_search_reduce(n, result, deadline=deadline)
            if len(reduced) < best_count:
                best_count = len(reduced)
                best_result = {'n': n, 'edges': reduced}
    
    # Try greedy set cover as backup
    for n in [11, 12, 10]:
        if time.time() > deadline:
            break
        
        result = _greedy_set_cover(n, deadline=deadline)
        if result is not None:
            if not _is_2_colorable(n, result):
                reduced = _local_search_reduce(n, result, deadline=deadline)
                if len(reduced) < best_count:
                    best_count = len(reduced)
                    best_result = {'n': n, 'edges': reduced}
    
    return best_result
