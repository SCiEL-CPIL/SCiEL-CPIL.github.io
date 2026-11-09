"""Few edges in 5-uniform hypergraphs without Property B (m(5)).

Strategy: Use a hardcoded 51-edge 5-uniform hypergraph on 11 vertices that is
known to be non-2-colorable (the best known construction for m(5) <= 51).
We verify non-2-colourability with an exact backtracking + unit propagation
checker (equivalent to the task's required check).

The construction is based on taking a non-2-colorable 3-uniform hypergraph
(the Fano plane has 7 edges on 7 vertices) and lifting it to 5-uniform by
adding two shared vertices. However, as analyzed, this simple lift may be
2-colorable (color all Fano vertices one color, shared vertices the other).

Instead, we use a more robust construction: the complete 5-graph on 9 vertices
has 126 edges, which is too many. We need a better approach.

Actually, let's use a known non-2-colorable 5-uniform hypergraph with 51 edges.
The best known m(5) construction uses 51 edges. Let me construct one carefully.

Approach: Use a 5-uniform hypergraph derived from the Petersen graph or similar
small non-2-colorable structures, then add edges to ensure non-2-colorability
while keeping the count at or below 51.

Let me use a systematic approach: start with vertices 0..10 (n=11) and build
edges that make the hypergraph non-2-colorable. A key insight: if we can find
a set of 5-edges such that no matter how we 2-color the 11 vertices, at least
one edge becomes monochromatic.

I'll use the following construction based on known results:
- Take the 11 vertices.
- Use edges that form a "covering" of all possible monochromatic patterns.

Actually, let me just hardcode a verified 51-edge construction. I'll generate
it using a greedy/ILP-like approach in code, but since we need it fast, I'll
embed a pre-computed one.

Let me use a different approach: the complement of a 2-coloring must hit every
edge. So we need a family of 5-sets such that every 2-coloring of [11] makes
at least one set monochromatic.

Here's a cleaner construction: Use the 11 vertices as the projective line over
GF(11) or similar. But let me just use a well-known small non-2-colorable
5-uniform hypergraph.

I'll construct it as follows:
- n = 11 vertices
- Use 51 edges derived from a known optimal construction

Since I can't easily verify a hardcoded list, let me use a construction that
I can verify in code: start with the Fano plane idea but fix it properly.

The issue with the Fano lift: adding two fixed vertices v, w to each Fano edge
{a,b,c} gives {a,b,c,v,w}. If we color a,b,c as 1 and v,w as 0, every edge
has both colors. So it IS 2-colorable. Bad.

Fix: Instead of adding fixed vertices, add vertices that vary. Or use a different
non-2-colorable base.

Let me use the following: a 5-uniform hypergraph on 11 vertices where we take
all edges of a 5-partite structure... no.

Let me try: n=9, complete 5-graph has C(9,5)=126 edges. Too many.
n=10, C(10,5)=252. More.

I need a sparse construction. Let me use the following approach in code:
1. Start with n=11 vertices.
2. Use a randomized greedy search to find a non-2-colorable 5-uniform hypergraph
   with few edges, using the exact checker to verify.
3. If found within time, return it.
4. Fallback: complete 5-graph on 9 vertices (126 edges, definitely non-2-colorable
   since any 2-coloring of 9 vertices has a color class of size >= 5, which forms
   a monochromatic edge).

Wait, that's a great insight! For K_n^5 (complete 5-graph), if n >= 10, then any
2-coloring has a color class of size >= 5 (pigeonhole: 10 vertices, 2 colors,
one color has >= 5). So K_10^5 is non-2-colorable with C(10,5)=252 edges.
But K_9^5: 9 vertices, 2 colors, one color has >= 5 (since ceil(9/2)=5). So
K_9^5 is also non-2-colorable! C(9,5)=126 edges.

Can we do better? We need at most 51 edges ideally. Let me search.

I'll implement a fast search: try random 5-uniform hypergraphs on n=11 with
increasing edge counts, checking non-2-colorability with a fast bitset solver.
"""

import time
import random
from itertools import combinations


def _is_2_colorable(n, edges):
    """Exact check (backtracking + unit propagation) for a proper 2-colouring.
    
    Returns True iff the hypergraph IS 2-colorable (i.e., has a proper 2-coloring).
    """
    if not edges:
        return True
    
    m = len(edges)
    # For each edge, store the bitmask of its vertices
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    all_vars = (1 << n) - 1
    import sys
    sys.setrecursionlimit(100000)
    
    def backtrack(assign, unassigned):
        # Check for contradictions and propagate
        changed = True
        while changed:
            changed = False
            for mask in edge_masks:
                # Edge is monochromatic if all its vertices have the same value.
                # Edge is satisfied (not monochromatic) if it has both 0 and 1.
                # Edge is falsified (monochromatic) if all assigned vertices in it have the same value.
                
                # For this edge, which vertices are assigned?
                assigned_in_edge = mask & ~unassigned
                unassigned_in_edge = mask & unassigned
                
                if unassigned_in_edge == 0:
                    # All vertices in this edge are assigned.
                    # Check if monochromatic: all 0 or all 1.
                    # All 1: assign has all bits in mask
                    # All 0: assign has no bits in mask
                    if (assign & mask) == mask or (assign & mask) == 0:
                        return None  # contradiction: monochromatic edge found
                    # Otherwise, this edge is satisfied (has both colors)
                else:
                    # Some vertices unassigned. Check if forced to be monochromatic.
                    # If all assigned vertices in this edge are 1, and we need to avoid monochromatic,
                    # then at least one unassigned vertex must be 0.
                    # If all assigned vertices in this edge are 0, and we need to avoid monochromatic,
                    # then at least one unassigned vertex must be 1.
                    # This is NOT unit propagation in the usual SAT sense for this problem.
                    # The constraint is: edge must NOT be monochromatic.
                    # This is a complex constraint, not a simple CNF clause.
                    # 
                    # Actually, "not monochromatic" for a set of size 5 means:
                    # (at least one is 0) AND (at least one is 1)
                    # = (NOT (all 1)) AND (NOT (all 0))
                    # = (OR of NOT x_v for v in edge) AND (OR of x_v for v in edge)
                    # So it IS a CNF with two clauses per edge!
                    
                    # Clause 1: OR of x_v for v in edge (at least one is 1)
                    # Clause 2: OR of NOT x_v for v in edge (at least one is 0)
                    # 
                    # For unit propagation:
                    # Clause 1 (at least one is 1): if all but one vertex in edge are assigned 0,
                    #   the remaining one must be 1.
                    # Clause 2 (at least one is 0): if all but one vertex in edge are assigned 1,
                    #   the remaining one must be 0.
                    
                    # Clause 1: at least one vertex in edge is 1
                    # Count how many vertices in edge are assigned 0.
                    assigned_zero = assigned_in_edge & ~assign
                    assigned_one = assigned_in_edge & assign
                    unassigned_count = bin(unassigned_in_edge).count('1')
                    
                    # If all assigned vertices in edge are 0, and unassigned_count >= 1:
                    # We need at least one 1. If unassigned_count == 1, that one must be 1.
                    if assigned_one == 0 and unassigned_count == 1:
                        v = unassigned_in_edge.bit_length() - 1
                        assign |= (1 << v)
                        unassigned &= ~(1 << v)
                        changed = True
                        continue
                    
                    # Clause 2: at least one vertex in edge is 0
                    # If all assigned vertices in edge are 1, and unassigned_count >= 1:
                    # We need at least one 0. If unassigned_count == 1, that one must be 0.
                    if assigned_zero == 0 and unassigned_count == 1:
                        v = unassigned_in_edge.bit_length() - 1
                        assign &= ~(1 << v)
                        unassigned &= ~(1 << v)
                        changed = True
                        continue
        
        if unassigned == 0:
            # All assigned, check no edge is monochromatic
            for mask in edge_masks:
                if (assign & mask) == mask or (assign & mask) == 0:
                    return None
            return assign
        
        # Branch on an unassigned variable
        v = unassigned.bit_length() - 1
        # Try True first
        res = backtrack(assign | (1 << v), unassigned & ~(1 << v))
        if res is not None:
            return res
        # Try False
        return backtrack(assign & ~(1 << v), unassigned & ~(1 << v))
    
    return backtrack(0, all_vars) is not None


def _complete_5_graph(n):
    """Return the complete 5-uniform hypergraph on n vertices."""
    from itertools import combinations
    return [list(e) for e in combinations(range(n), 5)]


def _greedy_set_cover(n, target_edges=None, deadline=None, seed=42):
    """
    Greedy set cover approach to find a non-2-colorable 5-uniform hypergraph.
    
    Model: Each 2-coloring of n vertices can be represented as a bitmask (n bits).
    A 5-edge (5-set) is monochromatic under coloring c if all 5 vertices have the
    same bit value. We want to select edges such that every possible coloring
    has at least one monochromatic edge.
    
    This is a set cover problem:
    - Universe: all 2^n colorings
    - For each 5-edge e, the set of colorings where e is monochromatic
    - Find minimum set of edges covering all colorings
    
    Greedy: repeatedly pick the edge that covers the most uncovered colorings.
    """
    rng = random.Random(seed)
    
    if deadline is None:
        deadline = time.time() + 130.0
    
    # All possible 5-edges
    all_edges = list(combinations(range(n), 5))
    
    # For each edge, precompute which colorings make it monochromatic
    # A coloring is a bitmask of n bits. Edge is monochromatic if all 5 bits are 0 or all 5 bits are 1.
    # For edge with vertices v0..v4, monochromatic-0 means bits v0..v4 are all 0.
    # Monochromatic-1 means bits v0..v4 are all 1.
    
    # Represent colorings as integers 0..2^n - 1
    # For each edge, compute the set of colorings where it's monochromatic as a bitset (Python int)
    
    # Optimization: For n=11, 2^11 = 2048 colorings. We can use Python ints as bitsets.
    
    # For each edge, compute mask0 (colorings where edge is all-0) and mask1 (colorings where edge is all-1)
    edge_coloring_sets = []
    for edge in all_edges:
        # mask0: bitset of colorings where all vertices in edge are 0
        # mask1: bitset of colorings where all vertices in edge are 1
        mask0 = 0
        mask1 = 0
        for coloring in range(1 << n):
            all_zero = True
            all_one = True
            for v in edge:
                bit = (coloring >> v) & 1
                if bit != 0:
                    all_zero = False
                if bit != 1:
                    all_one = False
            if all_zero:
                mask0 |= (1 << coloring)
            if all_one:
                mask1 |= (1 << coloring)
        
        edge_coloring_sets.append(mask0 | mask1)
    
    # Universe: all colorings
    universe = (1 << (1 << n)) - 1
    
    # Greedy set cover
    covered = 0
    selected_edges = []
    uncovered = universe
    
    # Sort edges by coverage size (descending) for initial ordering
    edge_indices = sorted(range(len(all_edges)), key=lambda i: bin(edge_coloring_sets[i]).count('1'), reverse=True)
    
    while uncovered != 0:
        if time.time() > deadline:
            break
        
        # Find edge with maximum coverage of uncovered colorings
        best_idx = -1
        best_count = 0
        for idx in edge_indices:
            if idx in selected_edges:
                continue
            coverage = edge_coloring_sets[idx] & uncovered
            count = bin(coverage).count('1')
            if count > best_count:
                best_count = count
                best_idx = idx
            if count == 0:
                continue
        
        if best_idx == -1 or best_count == 0:
            # No edge can cover more uncovered colorings
            break
        
        selected_edges.append(best_idx)
        covered |= edge_coloring_sets[best_idx]
        uncovered = universe & ~covered
        
        # If we've covered all colorings, we have a non-2-colorable hypergraph
        if uncovered == 0:
            break
    
    if uncovered == 0:
        # We found a complete cover
        result = [list(all_edges[i]) for i in selected_edges]
        return result
    else:
        return None


def construct():
    import time
    deadline = time.time() + 135.0  # safety margin before 150s kill
    
    # Fallback: complete 5-graph on 9 vertices (126 edges, definitely non-2-colorable)
    fallback_n = 9
    fallback_edges = _complete_5_graph(fallback_n)
    
    # Try greedy set cover on n=11
    n = 11
    result = _greedy_set_cover(n, deadline=deadline, seed=42)
    
    if result is not None:
        # Verify it's actually non-2-colorable
        if not _is_2_colorable(n, result):
            return {'n': n, 'edges': result}
    
    # Try a few more seeds
    for seed in [123, 456, 789, 101]:
        if time.time() > deadline:
            break
        result = _greedy_set_cover(n, deadline=deadline, seed=seed)
        if result is not None:
            if not _is_2_colorable(n, result):
                return {'n': n, 'edges': result}
    
    # Try randomized local search: start with random edges, add edges to cover remaining colorings
    if time.time() < deadline:
        rng = random.Random(42)
        all_edges = list(combinations(range(n), 5))
        
        # Precompute edge-coloring sets
        edge_coloring_sets = []
        for edge in all_edges:
            mask0 = 0
            mask1 = 0
            for coloring in range(1 << n):
                all_zero = True
                all_one = True
                for v in edge:
                    bit = (coloring >> v) & 1
                    if bit != 0:
                        all_zero = False
                    if bit != 1:
                        all_one = False
                if all_zero:
                    mask0 |= (1 << coloring)
                if all_one:
                    mask1 |= (1 << coloring)
            edge_coloring_sets.append(mask0 | mask1)
        
        universe = (1 << (1 << n)) - 1
        
        # Randomized greedy with different orderings
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            rng = random.Random(42 + attempt * 1000)
            rng.shuffle(all_edges)
            edge_indices = list(range(len(all_edges)))
            rng.shuffle(edge_indices)
            
            covered = 0
            selected_edges = []
            uncovered = universe
            
            while uncovered != 0:
                if time.time() > deadline:
                    break
                
                best_idx = -1
                best_count = 0
                for idx in edge_indices:
                    if idx in selected_edges:
                        continue
                    coverage = edge_coloring_sets[idx] & uncovered
                    count = bin(coverage).count('1')
                    if count > best_count:
                        best_count = count
                        best_idx = idx
                
                if best_idx == -1 or best_count == 0:
                    break
                
                selected_edges.append(best_idx)
                covered |= edge_coloring_sets[best_idx]
                uncovered = universe & ~covered
                
                if uncovered == 0:
                    break
            
            if uncovered == 0:
                result = [list(all_edges[i]) for i in selected_edges]
                if not _is_2_colorable(n, result):
                    return {'n': n, 'edges': result}
    
    # Return fallback
    return {'n': fallback_n, 'edges': fallback_edges}
