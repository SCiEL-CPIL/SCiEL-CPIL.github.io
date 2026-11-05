"""
Greedy Set Cover on the dual space (colorings) to find a minimal 5-uniform
hypergraph without Property B.

Strategy:
- Enumerate all 2^16 colorings of 16 vertices (2^16 = 65536, feasible).
- For each coloring, compute the set of 5-subsets (edges) that are monochromatic.
- Greedily select edges that "break" (make non-monochromatic) the most remaining
  valid colorings.
- Verify with exact backtracking + unit propagation that the resulting hypergraph
  is not 2-colorable.
- If needed, try different vertex counts or add a final verification/polish.

The known best is 51 edges. We aim to find a construction with as few edges as possible.
"""

import time
import sys
from itertools import combinations

# Deadline guard
_START = time.time()
TIME_LIMIT = 110.0  # seconds, safety margin before 120s evaluator kill


def _time_exceeded():
    return (time.time() - _START) > TIME_LIMIT


def _all_colorings(n):
    """Yield all 2^n colorings as bitmasks (0/1 per vertex)."""
    for mask in range(1 << n):
        yield mask


def _is_mono_5(mask, edge, n):
    """Check if all 5 vertices in edge have the same color in mask."""
    c = (mask >> edge[0]) & 1
    for v in edge[1:]:
        if ((mask >> v) & 1) != c:
            return False
    return True


def _mono_edges_for_coloring(mask, n):
    """Return the set of 5-subsets that are monochromatic under mask."""
    # Group vertices by color
    color0 = []
    color1 = []
    for v in range(n):
        if (mask >> v) & 1:
            color1.append(v)
        else:
            color0.append(v)
    
    mono = set()
    for group in (color0, color1):
        if len(group) >= 5:
            for e in combinations(group, 5):
                mono.add(e)
    return mono


def _backtrack_check(n, edges):
    """
    Exact check: is the hypergraph 2-colorable?
    Returns True if 2-colorable (i.e., there exists a coloring where no edge is monochromatic).
    Uses backtracking with unit propagation.
    """
    # Build adjacency: for each vertex, list of edges containing it
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    # For unit propagation: if 4 vertices of an edge have the same color,
    # the 5th must be the opposite color (to avoid monochromatic edge).
    
    def propagate(assignment, queue):
        """
        assignment: list of -1 (unassigned), 0, or 1 for each vertex.
        queue: list of vertex indices to process.
        Returns (success, assignment, queue) where success=False means contradiction.
        """
        while queue:
            v = queue.pop()
            if assignment[v] != -1:
                continue
            # Check all edges containing v
            forced = -1
            for ei in vertex_edges[v]:
                e = edges[ei]
                other_colors = []
                for u in e:
                    if u == v:
                        continue
                    if assignment[u] == -1:
                        other_colors = None
                        break
                    other_colors.append(assignment[u])
                if other_colors is None:
                    continue
                # All 4 other vertices are assigned
                if len(set(other_colors)) == 1:
                    # All same color -> v must be opposite
                    required = 1 - other_colors[0]
                    if forced == -1:
                        forced = required
                    elif forced != required:
                        return False, assignment, queue
            if forced != -1:
                assignment[v] = forced
                # Propagate to other vertices in edges containing v
                for ei in vertex_edges[v]:
                    for u in edges[ei]:
                        if u != v and assignment[u] == -1:
                            queue.append(u)
        return True, assignment, queue
    
    def backtrack(assignment, unassigned):
        """
        assignment: list of -1, 0, 1.
        unassigned: set of unassigned vertex indices.
        Returns True if a valid coloring exists.
        """
        if not unassigned:
            # All assigned; verify no edge is monochromatic
            for e in edges:
                colors = [assignment[v] for v in e]
                if len(set(colors)) == 1:
                    return False
            return True
        
        # Pick vertex with most constraints (most edges)
        v = max(unassigned, key=lambda x: len(vertex_edges[x]))
        unassigned.discard(v)
        
        for color in (0, 1):
            new_assignment = assignment[:]
            new_assignment[v] = color
            
            # Check immediate conflicts: any edge where all other 4 are same color
            conflict = False
            for ei in vertex_edges[v]:
                e = edges[ei]
                others = [u for u in e if u != v]
                if all(new_assignment[u] != -1 for u in others):
                    if len(set(new_assignment[u] for u in others)) == 1:
                        if new_assignment[others[0]] == color:
                            conflict = True
                            break
            if conflict:
                continue
            
            # Unit propagation
            queue = []
            for ei in vertex_edges[v]:
                for u in edges[ei]:
                    if u != v and new_assignment[u] == -1:
                        queue.append(u)
            
            success, new_assignment, queue = propagate(new_assignment, queue)
            if not success:
                continue
            
            new_unassigned = set(u for u in unassigned if new_assignment[u] == -1)
            
            if backtrack(new_assignment, new_unassigned):
                return True
        
        unassigned.add(v)
        return False
    
    assignment = [-1] * n
    unassigned = set(range(n))
    return backtrack(assignment, unassigned)


def construct():
    global _START
    _START = time.time()
    
    best_result = None
    best_edges = 201  # upper bound
    
    # Try n=16 first (2^16 = 65536 colorings, manageable)
    for n in [16, 15, 17]:
        if _time_exceeded():
            break
        
        # Enumerate all colorings and their monochromatic edge sets
        # For each coloring, we need the set of 5-subsets that are mono.
        # Total possible 5-subsets: C(n,5)
        # n=16: C(16,5) = 4368
        # n=15: C(15,5) = 3003
        # n=17: C(17,5) = 6188
        
        all_edges = list(combinations(range(n), 5))
        edge_index = {e: i for i, e in enumerate(all_edges)}
        num_edges = len(all_edges)
        
        # For each coloring, store which edges are monochromatic as a bitset or set
        # To save memory, we'll process in batches or use a different approach.
        
        # Greedy set cover approach:
        # Start with all 2^n colorings as "unbroken".
        # Repeatedly pick the edge that breaks the most remaining colorings.
        
        # For each edge, precompute which colorings make it monochromatic.
        # An edge e is monochromatic in coloring c if all 5 vertices have same color.
        # For a fixed edge, the number of colorings where it's mono = 2^(n-5) * 2 = 2^(n-4)
        # (choose the common color: 2 ways, assign remaining n-5 vertices freely)
        
        # Instead of storing all, we'll iterate:
        # remaining_colorings: set of coloring masks
        # For each candidate edge, count how many remaining colorings have it mono.
        
        # This is expensive if done naively. Let's use a smarter approach.
        
        # Alternative: represent the set of remaining colorings compactly.
        # Since we start with all 2^n colorings, and each edge removes a subset,
        # we can track this as a list of (edge, removed_colorings) but that's large.
        
        # Better: iterate over all colorings for each candidate edge selection.
        # With 65536 colorings and ~4368 edges, and maybe ~50 iterations,
        # that's 50 * 4368 * 65536 ≈ 14 billion operations. Too slow.
        
        # Let's use a different strategy: precompute for each coloring its mono edges,
        # then for each edge, count how many colorings have it in their mono set.
        
        # Memory: 65536 colorings * up to ~200 mono edges each. Manageable.
        
        coloring_mono_edges = []
        for mask in range(1 << n):
            if _time_exceeded():
                break
            mono = set()
            color0 = []
            color1 = []
            for v in range(n):
                if (mask >> v) & 1:
                    color1.append(v)
                else:
                    color0.append(v)
            for group in (color0, color1):
                if len(group) >= 5:
                    for e in combinations(group, 5):
                        mono.add(e)
            coloring_mono_edges.append(mono)
        
        if _time_exceeded():
            continue
        
        num_colorings = 1 << n
        
        # remaining: boolean array, True if coloring is still "valid" (not broken yet)
        remaining = [True] * num_colorings
        num_remaining = num_colorings
        
        # For each edge, maintain count of how many remaining colorings have it mono
        edge_count = [0] * num_edges
        for i, mono in enumerate(coloring_mono_edges):
            for e in mono:
                edge_count[edge_index[e]] += 1
        
        selected_edges = []
        
        while num_remaining > 0 and not _time_exceeded():
            # Find edge with max count
            best_ei = -1
            best_count = -1
            for ei in range(num_edges):
                if edge_count[ei] > best_count:
                    best_count = edge_count[ei]
                    best_ei = ei
            
            if best_count <= 0:
                break  # Should not happen if num_remaining > 0
            
            selected_edges.append(all_edges[best_ei])
            
            # Remove all colorings where this edge is mono
            e = all_edges[best_ei]
            for i in range(num_colorings):
                if remaining[i] and e in coloring_mono_edges[i]:
                    remaining[i] = False
                    num_remaining -= 1
                    # Update edge counts for other mono edges in this coloring
                    for other_e in coloring_mono_edges[i]:
                        if other_e != e:
                            edge_count[edge_index[other_e]] -= 1
            
            # Update count for the selected edge itself
            edge_count[best_ei] = 0  # All remaining colorings with this edge are gone
        
        if num_remaining == 0 and not _time_exceeded():
            # Verify with exact backtracking
            if not _backtrack_check(n, selected_edges):
                # Not 2-colorable! This is our answer.
                if len(selected_edges) < best_edges:
                    best_edges = len(selected_edges)
                    best_result = {'n': n, 'edges': [list(e) for e in selected_edges]}
            else:
                # It is 2-colorable, which contradicts our set cover. 
                # This shouldn't happen if our set cover is correct, but let's be safe.
                pass
    
    # If we didn't find anything, fall back to a known construction or the complete graph
    if best_result is not None:
        return best_result
    
    # Fallback: complete 5-uniform hypergraph on 9 vertices (63 edges)
    from itertools import combinations as comb
    return {'n': 9, 'edges': [list(e) for e in comb(range(9), 5)]}
