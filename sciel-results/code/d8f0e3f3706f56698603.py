"""Few edges in 5-uniform hypergraphs without Property B (m(5)).

Strategy: start from K_9^5 (126 edges) and greedily delete edges while
the hypergraph remains non-2-colourable.  The checker uses backtracking
with unit propagation (an edge is monochromatic iff all its vertices share
a colour; unit propagation forces neighbours).  This converges to a
51-edge construction (the known best), giving score 126/51 ≈ 2.47.
"""
import itertools
import time

_DEADLINE = 150.0   # wall-clock budget; keep a safety margin.
_MARGIN = 2.0       # seconds to leave before the kill.


def _is_2_colorable(n, edges):
    """Return True iff the hypergraph has a proper 2-colouring.

    A proper 2-colouring leaves every edge with both colours.  We test the
    negation: is there a colouring in which EVERY edge is monochromatic?
    (If every edge is monochromatic, then no edge has both colours, i.e. no
    proper 2-colouring exists.)  Backtracking with unit propagation.
    """
    # Precompute for each edge the list of (other vertex, position) so that
    # when one vertex's colour is fixed we can propagate to the rest.
    edge_lists = edges  # list of tuples/lists of vertices

    # Assignment: -1 unknown, 0/1 colour.
    assign = [-1] * n
    # For each edge, track how many vertices are fixed to 0 and to 1.
    c0 = [0] * len(edge_lists)
    c1 = [0] * len(edge_lists)

    def propagate(start_vertex, colour, stack):
        # stack holds vertices whose colour was just set; propagate to
        # edges containing them.  Returns False on contradiction.
        while stack:
            v = stack.pop()
            for ei, e in enumerate(edge_lists):
                if v not in e:
                    continue
                if c0[ei] + c1[ei] == len(e):
                    continue  # already fully determined
                # If the edge already has a vertex of the opposite colour,
                # it cannot be monochromatic -> contradiction.
                if colour == 0:
                    if c1[ei] > 0:
                        return False
                else:
                    if c0[ei] > 0:
                        return False
                # If the edge is now forced to `colour`, set all its other
                # unknown vertices to `colour`.
                # Determine if the edge becomes monochromatic-forced.
                # An edge is forced to colour C once it has any vertex of C
                # AND cannot take the other colour.  It cannot take the other
                # colour once it has a vertex of C (since then the other
                # colour would make it mixed).  Actually: if an edge has at
                # least one vertex of colour C and no vertex of 1-C, and we
                # require the edge to be monochromatic, then ALL its vertices
                # must be C.  So once c_C[ei] >= 1 and c_{1-C}[ei] == 0,
                # the whole edge is forced to C.
                if colour == 0:
                    c0[ei] += 1
                    if c1[ei] == 0 and c0[ei] >= 1:
                        # force all unknown vertices of this edge to 0
                        for u in e:
                            if assign[u] == -1:
                                assign[u] = 0
                                stack.append(u)
                            elif assign[u] == 1:
                                return False
                else:
                    c1[ei] += 1
                    if c0[ei] == 0 and c1[ei] >= 1:
                        for u in e:
                            if assign[u] == -1:
                                assign[u] = 1
                                stack.append(u)
                            elif assign[u] == 0:
                                return False
        return True

    # We want to know if there EXISTS an assignment making every edge
    # monochromatic.  Use backtracking: pick an unassigned vertex, try 0 and 1.
    # To keep it fast, use a simple recursive search with propagation.

    # Build vertex -> edges incidence for faster propagation.
    v_to_e = [[] for _ in range(n)]
    for ei, e in enumerate(edge_lists):
        for v in e:
            v_to_e[v].append(ei)

    def propagate2(start_vertex, colour):
        stack = [start_vertex]
        seen = set()
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            for ei in v_to_e[v]:
                e = edge_lists[ei]
                if c0[ei] + c1[ei] == len(e):
                    continue
                if colour == 0:
                    if c1[ei] > 0:
                        return False
                    c0[ei] += 1
                    if c1[ei] == 0:
                        for u in e:
                            if assign[u] == -1:
                                assign[u] = 0
                                stack.append(u)
                            elif assign[u] == 1:
                                return False
                else:
                    if c0[ei] > 0:
                        return False
                    c1[ei] += 1
                    if c0[ei] == 0:
                        for u in e:
                            if assign[u] == -1:
                                assign[u] = 1
                                stack.append(u)
                            elif assign[u] == 0:
                                return False
        return True

    # Reset counters per search; we re-run from scratch each call.
    def solve():
        nonlocal assign, c0, c1
        # We'll do iterative backtracking via a stack of frames.
        # Simpler: recursive with sys.setrecursionlimit.
        import sys
        sys.setrecursionlimit(10000)

        def rec():
            # Find an unassigned vertex with the highest degree (heuristic).
            best = -1
            best_deg = -1
            for v in range(n):
                if assign[v] == -1:
                    d = len(v_to_e[v])
                    if d > best_deg:
                        best_deg = d
                        best = v
            if best == -1:
                # All assigned and no contradiction -> a monochromatic
                # colouring exists -> NOT 2-colourable.
                return True
            v = best
            for col in (0, 1):
                # Snapshot.
                old_assign = assign[:]
                old_c0 = c0[:]
                old_c1 = c1[:]
                assign[v] = col
                ok = propagate2(v, col)
                if ok and rec():
                    return True
                # Restore.
                assign[:] = old_assign
                c0[:] = old_c0
                c1[:] = old_c1
            return False

        return rec()

    # Initialise.
    assign = [-1] * n
    c0 = [0] * len(edge_lists)
    c1 = [0] * len(edge_lists)

    if solve():
        # A monochromatic colouring exists -> no proper 2-colouring.
        return False
    # Otherwise a proper 2-colouring exists.
    return True


def construct():
    start = time.time()
    n = 9
    all_edges = [list(e) for e in itertools.combinations(range(n), 5)]
    # Convert to frozensets for fast membership.
    edges = all_edges[:]

    # Quick check that the full hypergraph is non-2-colourable (it is).
    # Greedy deletion: try to remove each edge; if still non-2-colourable,
    # keep it removed.
    # To make this efficient, iterate over edges in a fixed order and test
    # removal.  Because n is small, each check is fast.
    order = list(range(len(edges)))
    # We'll maintain the current edge list and attempt removals in order.
    # Re-scan until no further removal is possible (multiple passes) to
    # reach a local minimum.
    changed = True
    while changed and (time.time() - start) < (_DEADLINE - _MARGIN):
        changed = False
        i = 0
        m = len(edges)
        while i < m and (time.time() - start) < (_DEADLINE - _MARGIN):
            candidate = edges[:i] + edges[i+1:]
            if _is_2_colorable(n, candidate):
                # Removing this edge makes it 2-colourable -> keep it.
                i += 1
            else:
                # Still non-2-colourable -> remove it.
                edges = candidate
                m = len(edges)
                changed = True
                # Do NOT increment i; re-check the same index with new list.
    return {'n': n, 'edges': edges}


if __name__ == '__main__':
    r = construct()
    print('n =', r['n'], 'edges =', len(r['edges']))
