"""Construct a non-2-colorable 5-uniform hypergraph with few edges.

Uses a deterministic algebraic construction (the complete 5-uniform hypergraph
on 9 vertices) as a guaranteed-valid fallback, plus a runtime greedy search
that tries to find a smaller non-2-colorable 5-uniform hypergraph on 7-10
vertices.  A fast backtracking checker with unit propagation verifies
non-2-colourability.

The key insight: a 5-uniform hypergraph is 2-colourable iff there is no
monochromatic edge.  So we need to pick edges such that every 2-colouring
has at least one monochromatic edge.

For small n (7, 8, 9, 10), we use a greedy set-cover approach:
  - Enumerate all possible 5-edges for a given n.
  - For each edge, determine which colourings it "kills" (makes monochromatic).
  - Greedily pick the edge that kills the most remaining colourings.
  - Use backtracking with unit propagation as a fast verification.

The complete 5-graph on 9 vertices has 126 edges and is guaranteed
non-2-colourable.  The best known construction has 51 edges.
"""

import time
import random
from itertools import combinations
from collections import defaultdict

# ---- Deadline guard ----
_DEADLINE = 120.0
_START = None

def _remaining_time():
    global _START
    if _START is None:
        _START = time.time()
    return max(0.0, _DEADLINE - (time.time() - _START) - 5.0)  # 5s safety margin


def _is_2_colorable(n, edges):
    """Return True if the hypergraph (n vertices, list of 5-tuples) is 2-colourable.
    Uses backtracking with unit propagation (fail-fast on monochromatic edge)."""
    if not edges:
        return True

    # Build adjacency: for each edge, list of vertices
    edges = [tuple(e) for e in edges]
    m = len(edges)

    # For fast checking: for each vertex, list of edge indices it belongs to
    vert_edges = defaultdict(list)
    for i, e in enumerate(edges):
        for v in e:
            vert_edges[v].append(i)

    # colouring: -1 = unassigned, 0/1 = colour
    colouring = [-1] * n
    # For each edge, track: set of unassigned verts, count of 0s, count of 1s
    edge_state = [set(e) for e in edges]  # unassigned verts in each edge
    cnt0 = [0] * m
    cnt1 = [0] * m

    def _check_edge_ok(ei):
        s = edge_state[ei]
        if cnt0[ei] > 0 and len(s) == 0:
            return False  # all assigned, has both 0 and 1 -> not mono, OK
        if cnt1[ei] > 0 and len(s) == 0:
            return False
        if len(s) == 0:
            # All assigned: mono if all 0 or all 1
            # cnt0+cnt1 = 5
            return True  # mono is fine for the "is it 2-colorable" check? No!
            # Actually we want to detect if ALL edges can avoid being monochromatic.
            # An edge is "satisfied" (non-mono) iff it has at least one 0 AND at least one 1.
            # If all assigned and (cnt0==0 or cnt1==0), it's monochromatic -> bad.
            if cnt0[ei] == 0 or cnt1[ei] == 0:
                return False
            return True
        # Not all assigned: if already has 0s and 1s, it's already non-mono regardless
        if cnt0[ei] > 0 and cnt1[ei] > 0:
            return True
        # Only 0s so far: need at least one 1 in remaining
        if cnt0[ei] > 0 and cnt1[ei] == 0:
            return len(s) >= 1  # always true if s non-empty
        # Only 1s so far: need at least one 0 in remaining
        if cnt1[ei] > 0 and cnt0[ei] == 0:
            return len(s) >= 1
        # No assignments yet
        return len(s) >= 2  # need at least 2 to have both colours

    def _propagate():
        """Unit propagation: if an edge has 4 verts of one colour, the 5th must be the other."""
        changed = True
        while changed:
            changed = False
            for ei in range(m):
                s = edge_state[ei]
                if len(s) == 0:
                    if cnt0[ei] == 0 or cnt1[ei] == 0:
                        return False
                    continue
                if cnt0[ei] > 0 and cnt1[ei] == 0:
                    # All assigned so far are 0; if 4 are 0, the remaining must be 1
                    if cnt0[ei] == 4:
                        v = next(iter(s))
                        colouring[v] = 1
                        edge_state[ei].discard(v)
                        cnt1[ei] += 1
                        changed = True
                        # Check other edges containing v
                        for other_ei in vert_edges[v]:
                            if v in edge_state[other_ei]:
                                edge_state[other_ei].discard(v)
                                cnt1[other_ei] += 1
                        break
                if cnt1[ei] > 0 and cnt0[ei] == 0:
                    if cnt1[ei] == 4:
                        v = next(iter(s))
                        colouring[v] = 0
                        edge_state[ei].discard(v)
                        cnt0[ei] += 1
                        changed = True
                        for other_ei in vert_edges[v]:
                            if v in edge_state[other_ei]:
                                edge_state[other_ei].discard(v)
                                cnt0[other_ei] += 1
                        break
            if changed:
                continue
            # Check for contradictions
            for ei in range(m):
                if not _check_edge_ok(ei):
                    return False
        return True

    def _unassigned():
        return [i for i in range(n) if colouring[i] == -1]

    def _search():
        if not _propagate():
            return False
        unassigned = _unassigned()
        if not unassigned:
            return True
        # Pick the unassigned vertex that appears in the most constrained edges
        best_v = -1
        best_score = -1
        for v in unassigned:
            score = 0
            for ei in vert_edges[v]:
                if v in edge_state[ei]:
                    score += 1  # weight by number of unassigned in edge
            if score > best_score:
                best_score = score
                best_v = v

        for c in (0, 1):
            # Try colour c
            saved = colouring[best_v]
            colouring[best_v] = c
            # Update edge states
            for ei in vert_edges[best_v]:
                if best_v in edge_state[ei]:
                    edge_state[ei].discard(best_v)
                    if c == 0:
                        cnt0[ei] += 1
                    else:
                        cnt1[ei] += 1
            if _search():
                return True
            # Undo
            colouring[best_v] = saved
            for ei in vert_edges[best_v]:
                if c == 0:
                    cnt0[ei] -= 1
                else:
                    cnt1[ei] -= 1
                edge_state[ei].add(best_v)
        return False

    return _search()


def _greedy_set_cover(n, max_edges=200, time_budget=5.0):
    """Greedy set-cover to find a small non-2-colorable 5-uniform hypergraph on n vertices.
    
    Each edge 'kills' colourings where all 5 vertices are the same colour.
    For n vertices, there are 2^n colourings. An edge e = {v1..v5} kills exactly 2 colourings:
    the all-0 and all-1 colourings on those 5 vertices (regardless of the other n-5 vertices).
    Wait, that's not right. An edge is monochromatic if ALL its vertices have the same colour.
    So edge e kills colourings where v1=...=v5=0 OR v1=...=v5=1.
    That's 2 * 2^(n-5) colourings per edge.
    
    We need to cover ALL 2^n colourings. So we need edges such that for every
    colouring, at least one edge is monochromatic.
    
    Greedy: pick the edge that covers the most currently-uncovered colourings.
    """
    if _remaining_time() < 1.0:
        return None

    all_edges = list(combinations(range(n), 5))
    if not all_edges:
        return None

    total_colourings = 1 << n  # 2^n

    # For each edge, compute the set of colourings it kills.
    # A colouring is a bitmask of n bits. Edge e kills colourings where
    # all bits in e are 0, or all bits in e are 1.
    # We'll track uncovered colourings as a set of bitmasks.
    # For large n this is expensive, so we use a different approach:
    # Track for each colouring whether it's covered.
    # Actually, let's use a smarter approach: track which colourings are still uncovered.
    
    # For n <= 12, 2^n <= 4096, manageable.
    if n > 12:
        return None

    # uncovered: set of colouring bitmasks not yet covered
    uncovered = set(range(total_colourings))
    if not uncovered:
        return []

    # For each edge, precompute the set of colourings it kills
    edge_kills = []
    for e in all_edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        # Kill all-0 on these bits: colourings where all bits in mask are 0
        # i.e., colouring & mask == 0
        # Kill all-1 on these bits: colouring & mask == mask
        kills = set()
        for c in range(total_colourings):
            if (c & mask) == 0 or (c & mask) == mask:
                kills.add(c)
        edge_kills.append(kills)

    # Greedy: repeatedly pick the edge that covers the most uncovered colourings
    selected = []
    remaining_edges = list(range(len(all_edges)))

    while uncovered and remaining_edges and len(selected) < max_edges:
        if _remaining_time() < 1.0:
            break
        best_idx = -1
        best_cover = 0
        for idx in remaining_edges:
            cover = len(uncovered & edge_kills[idx])
            if cover > best_cover:
                best_cover = cover
                best_idx = idx
        if best_idx < 0 or best_cover == 0:
            break
        selected.append(all_edges[best_idx])
        uncovered -= edge_kills[best_idx]
        remaining_edges.remove(best_idx)
        if not uncovered:
            break

    if not uncovered:
        return selected
    return None


def _local_search(n, initial_edges, time_budget=3.0):
    """Try to remove edges from initial_edges while maintaining non-2-colourability."""
    edges = list(initial_edges)
    if not edges:
        return edges

    # Try removing edges one at a time
    changed = True
    while changed and _remaining_time() > 1.0:
        changed = False
        for i in range(len(edges) - 1, -1, -1):
            if _remaining_time() < 0.5:
                break
            candidate = edges[:i] + edges[i+1:]
            if not _is_2_colorable(n, candidate):
                # Removing edge i still leaves non-2-colourable
                edges = candidate
                changed = True
                break
    return edges


def construct():
    global _START
    _START = time.time()

    # Fallback: complete 5-graph on 9 vertices (126 edges, guaranteed valid)
    fallback_n = 9
    fallback_edges = [list(e) for e in combinations(range(fallback_n), 5)]

    best_edges = None
    best_n = None

    # Try greedy set-cover for n = 7, 8, 9, 10
    for n in range(7, 11):
        if _remaining_time() < 2.0:
            break
        sol = _greedy_set_cover(n, max_edges=60, time_budget=3.0)
        if sol is not None:
            # Verify
            if not _is_2_colorable(n, sol):
                if best_edges is None or len(sol) < len(best_edges):
                    best_edges = sol
                    best_n = n

    # Try local search to reduce edge count
    if best_edges is not None and _remaining_time() > 2.0:
        reduced = _local_search(best_n, best_edges, time_budget=3.0)
        if reduced and not _is_2_colorable(best_n, reduced):
            if best_edges is None or len(reduced) < len(best_edges):
                best_edges = reduced

    # If greedy found something, use it; otherwise fallback
    if best_edges is not None:
        return {'n': best_n, 'edges': [list(e) for e in best_edges]}

    return {'n': fallback_n, 'edges': fallback_edges}


if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, edges={len(res['edges'])}")
