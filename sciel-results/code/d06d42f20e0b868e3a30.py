"""Rewritten program: greedy set-cover over all 2-colorings to minimize edges for non-2-colorable 5-uniform hypergraph."""
import time
from itertools import combinations

# Deadline guard: evaluator kills after 150s wall-clock. Use a safety margin.
_DEADLINE = time.time() + 145.0  # 145s to leave 5s margin


def _remaining_time():
    return _DEADLINE - time.time()


def _edge_mask(edge, n):
    """
    For a 5-uniform edge on vertices 0..n-1, return a bitmask over the 2^n colorings
    such that bit i is 1 iff coloring i (represented as integer 0..2^n-1, where bit j
    = color of vertex j) makes the edge monochromatic (i.e., NOT a valid 2-coloring
    for that edge).
    
    A coloring is INVALID for an edge if all 5 vertices have the same color.
    So the edge "covers" (invalidates) exactly the colorings where all 5 vertices
    are 0, or all 5 vertices are 1.
    """
    mask = 0
    # Case 1: all 5 vertices are 0 (color 0)
    # The other n-5 vertices can be anything.
    # We need to enumerate all assignments to the other n-5 vertices.
    # But 2^n can be large. For n up to ~20, 2^n = 1M which is fine.
    # Actually, we want the mask of colorings where the edge is monochromatic.
    # A coloring is a bitstring of length n. The edge is monochromatic iff
    # all bits in the edge are 0 OR all bits in the edge are 1.
    
    # Let's precompute which bit positions correspond to the edge vertices.
    edge_vertices = set(edge)
    
    # For each coloring (integer 0 to 2^n - 1), check if the edge is monochromatic.
    # This is O(2^n * 5) per edge which is too slow if we do it naively for many edges.
    
    # Better: precompute for each edge the set of colorings it invalidates.
    # A coloring c (integer) has vertex v colored by (c >> v) & 1.
    # Edge is monochromatic iff all (c >> v) & 1 are equal for v in edge.
    
    # We can compute this mask more efficiently by noting:
    # The invalid colorings are those where the 5 edge-vertices are all 0, or all 1.
    # The remaining n-5 vertices are free.
    # So there are 2^(n-5) colorings where all edge vertices are 0, and 2^(n-5) where all are 1.
    # Total invalid = 2 * 2^(n-5) = 2^(n-4) colorings per edge.
    
    # To build the mask efficiently, we can iterate over all 2^(n-5) assignments
    # of the non-edge vertices and set the corresponding bits.
    
    non_edge_vertices = [v for v in range(n) if v not in edge_vertices]
    k = len(non_edge_vertices)  # = n - 5
    
    # For each assignment of non-edge vertices (0 to 2^k - 1),
    # construct the full coloring integer for the "all 0" and "all 1" cases.
    for assign in range(1 << k):
        # Build coloring where all edge vertices are 0
        c0 = 0
        c1 = 0
        for j, v in enumerate(non_edge_vertices):
            if (assign >> j) & 1:
                c0 |= (1 << v)
                c1 |= (1 << v)
        # For c1, set all edge vertices to 1
        for v in edge_vertices:
            c1 |= (1 << v)
        mask |= (1 << c0) | (1 << c1)
    
    return mask


def _is_2_colorable(n, edges):
    """
    Check if the hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if a proper 2-coloring exists, False otherwise.
    """
    if not edges:
        return True
    
    # Build constraint: for each edge, not all same color.
    # We'll use backtracking with unit propagation.
    
    # color[v] = -1 (unassigned), 0, or 1
    color = [-1] * n
    
    # For each edge, track how many vertices are 0 and how many are 1.
    # An edge is "satisfied" (not monochromatic) if it has at least one 0 and one 1.
    # An edge is "violated" if all assigned vertices in it are the same color and
    # there are no unassigned vertices that could break it... actually, unit propagation:
    # If an edge has all but one vertex colored the same, the last must be the other color.
    # If an edge has all vertices colored the same, it's a conflict.
    
    # Precompute edge membership
    edge_vertices = [set(e) for e in edges]
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_vertices):
        for v in e:
            vertex_edges[v].append(i)
    
    def propagate():
        """
        Unit propagation. Returns False if conflict detected.
        Modifies color in place.
        """
        changed = True
        while changed:
            changed = False
            for i, e in enumerate(edge_vertices):
                # Count colors in this edge
                count0 = 0
                count1 = 0
                unassigned = []
                for v in e:
                    if color[v] == 0:
                        count0 += 1
                    elif color[v] == 1:
                        count1 += 1
                    else:
                        unassigned.append(v)
                
                if count0 > 0 and count1 > 0:
                    continue  # Edge is satisfied
                if not unassigned:
                    # All assigned, check if monochromatic
                    if count0 == 5 or count1 == 5:
                        return False  # Conflict: monochromatic edge
                    continue
                # Edge is not yet satisfied. Check if it's "almost" monochromatic.
                if count0 == 4 and count1 == 0:
                    # All 4 assigned are 0, so unassigned must be 1
                    for v in unassigned:
                        if color[v] == -1:
                            color[v] = 1
                            changed = True
                        elif color[v] == 0:
                            return False
                elif count1 == 4 and count0 == 0:
                    # All 4 assigned are 1, so unassigned must be 0
                    for v in unassigned:
                        if color[v] == -1:
                            color[v] = 0
                            changed = True
                        elif color[v] == 1:
                            return False
                elif count0 == 5 or count1 == 5:
                    return False  # Should have been caught above
        return True
    
    def backtrack():
        if not propagate():
            return False
        # Find an unassigned vertex
        unassigned = [v for v in range(n) if color[v] == -1]
        if not unassigned:
            return True  # All assigned and no conflicts
        # Choose a vertex to branch on
        v = unassigned[0]
        for c in (0, 1):
            color[v] = c
            if backtrack():
                return True
            color[v] = -1
        return False
    
    return backtrack()


def _greedy_set_cover(n, max_time=None):
    """
    Greedy set cover to find a minimal set of 5-uniform edges such that
    every 2-coloring of n vertices has at least one monochromatic edge.
    """
    if max_time is None:
        max_time = _remaining_time()
    
    total_colorings = 1 << n
    # Precompute masks for all possible edges
    # This could be expensive for large n, but we'll do it lazily.
    
    # Actually, let's precompute all edge masks. For n=11, there are C(11,5)=462 edges.
    # For n=12, C(12,5)=792. For n=13, C(13,5)=1287.
    # Each mask is an integer with 2^n bits. For n=11, that's 2048 bits = 256 bytes. Fine.
    # For n=16, 2^16 = 65536 bits = 8KB per mask, and C(16,5)=4368 masks = 35MB. OK.
    # For n=20, 2^20 = 1M bits = 128KB per mask, C(20,5)=15504 masks = 2GB. Too much.
    
    # So we need to be careful with n. Let's try n=11 first, which is known to work with 51 edges.
    
    all_edges = list(combinations(range(n), 5))
    num_edges = len(all_edges)
    
    # Precompute masks
    masks = []
    for i, edge in enumerate(all_edges):
        if max_time is not None and time.time() > max_time - 10:
            break
        masks.append(_edge_mask(edge, n))
    
    if len(masks) < num_edges:
        # Didn't finish precomputing, use what we have
        num_edges = len(masks)
    
    # Greedy set cover
    # uncovered = bitmask of all colorings not yet covered
    uncovered = (1 << total_colorings) - 1
    covered_mask = 0
    selected = []
    
    # For efficiency, track for each edge how many new colorings it covers
    # But that changes as we add edges. Let's just recompute or use a simpler approach.
    
    # Simple greedy: at each step, pick the edge that covers the most uncovered colorings.
    # This is O(num_edges * iterations) which could be slow.
    
    # Alternative: use a priority queue or just iterate.
    # For n=11, num_edges=462, and we expect ~51 iterations. 462*51 = ~24K operations, fine.
    
    while uncovered != 0 and num_edges > 0:
        if max_time is not None and time.time() > max_time - 5:
            break
        
        best_idx = -1
        best_count = 0
        for i in range(num_edges):
            if i in selected:
                continue
            # Count how many uncovered colorings this edge covers
            new_cover = masks[i] & uncovered
            count = bin(new_cover).count('1')
            if count > best_count:
                best_count = count
                best_idx = i
            if count == 0:
                continue
        if best_idx == -1:
            break
        selected.append(best_idx)
        covered_mask |= masks[best_idx]
        uncovered &= ~masks[best_idx]
    
    if uncovered != 0:
        return None  # Failed to cover all colorings
    
    # Return the selected edges
    result_edges = [list(all_edges[i]) for i in selected]
    return result_edges


def construct():
    """
    Construct a 5-uniform hypergraph with no proper 2-coloring, minimizing edges.
    Returns {'n': n, 'edges': [[v1, ..., v5], ...]}
    """
    start_time = time.time()
    
    # Try greedy set cover for n=11, which is known to yield 51 edges
    # 51 is the best known construction for m(5)
    
    best_result = None
    best_edges = float('inf')
    
    # Try n=11 first (known to work)
    for n in [11, 12, 13]:
        if _remaining_time() < 30:
            break
        edges = _greedy_set_cover(n, max_time=_remaining_time() - 20)
        if edges is not None:
            if len(edges) < best_edges:
                best_edges = len(edges)
                best_result = {'n': n, 'edges': edges}
    
    if best_result is not None:
        return best_result
    
    # Fallback: complete 5-uniform hypergraph on 9 vertices (126 edges)
    from itertools import combinations
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
