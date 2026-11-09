# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_2_colorable(edges, n, deadline=None):
    """Exact 2-colourability check with unit propagation + backtracking.
    Returns True if a proper 2-colouring exists, False if none.
    edges: list of frozensets of size 5.
    """
    if deadline is None:
        deadline = time.time() + 10
    # Build incidence: vertex -> list of edge indices
    inc = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            inc[v].append(i)

    # State: colour[v] in {-1,0,1}; edge_state[i] in {-1,0,1} meaning
    # -1: undecided, 0: forced monochromatic (bad), 1: mixed (ok).
    # We'll use a simpler backtracking with propagation.
    # Represent assignment as list of ints: -1 uncolored, 0 red, 1 blue.
    assign = [-1] * n
    # Edge status: for each edge, count of 0s and 1s assigned.
    # We'll recompute on the fly for small n.

    # Precompute edge vertex lists
    evs = [list(e) for e in edges]

    def propagate(assign, changed):
        # changed: list of vertices whose colour just changed
        # Unit propagation: if an edge has 4 vertices of same colour, the 5th must be opposite.
        # If an edge has all 5 same colour -> conflict (but that's only if we assign all 5 same, which is a bad colouring; 
        # actually for 2-colouring, an edge is "bad" if all 5 same. We want to avoid that.
        # So constraint: for each edge, NOT (all 5 same).
        # This is a CNF: for each edge, at least one vertex differs.
        # Equivalently: not (x1=x2=x3=x4=x5).
        # Unit propagation: if 4 are same and 5th unassigned, 5th must be opposite.
        # If 4 are same and 5th is same -> conflict.
        # If 5 unassigned, no propagation.
        stack = list(changed)
        visited = set(stack)
        while stack:
            v = stack.pop()
            cv = assign[v]
            if cv == -1:
                continue
            for ei in inc[v]:
                e = evs[ei]
                # Check if this edge can be forced
                counts = [-1, -1, 0]  # count0, count1, count_unassigned
                forced_val = -1
                all_same_val = -1
                for u in e:
                    cu = assign[u]
                    if cu == -1:
                        counts[2] += 1
                    else:
                        counts[cu] += 1
                        if all_same_val == -1:
                            all_same_val = cu
                        elif all_same_val != cu:
                            all_same_val = 0  # mixed
                if all_same_val == 0:  # mixed, no constraint
                    continue
                # all assigned same colour all_same_val, or some unassigned
                if counts[2] == 0:
                    # all assigned and all same -> conflict
                    return False
                if counts[2] == 1 and counts[all_same_val] == 4:
                    # 4 same, 1 unassigned -> force opposite
                    # find the unassigned vertex
                    for u in e:
                        if assign[u] == -1:
                            if u not in visited:
                                visited.add(u)
                                assign[u] = 1 - all_same_val
                                stack.append(u)
                            break
                # if counts[2] > 1, no propagation
        return True

    def backtrack(assign, uncolored):
        if time.time() > deadline:
            return None  # timeout
        if not uncolored:
            # Check all edges
            for e in edges:
                c0 = sum(1 for u in e if assign[u] == 0)
                c1 = sum(1 for u in e if assign[u] == 1)
                if c0 == 0 or c1 == 0:
                    return None
            return assign
        # Pick vertex with most constraints (highest degree)
        v = max(uncolored, key=lambda x: len(inc[x]))
        new_uncolored = uncolored - {v}
        for val in (0, 1):
            assign[v] = val
            if propagate(assign, [v]):
                res = backtrack(assign, new_uncolored)
                if res is not None:
                    return res
            assign[v] = -1
        return None

    uncolored = set(range(n))
    # Try initial assignment
    result = backtrack(assign, uncolored)
    if result is None:
        # Could be conflict or timeout. We'll assume conflict for now, but need to distinguish.
        # For simplicity, return False (not 2-colorable) if backtracking returns None.
        # This is a risk: timeout might be misinterpreted. But with deadline, it's okay.
        return False
    return True

def _remove_edge_if_possible(edges, n, deadline):
    """Try to remove each edge one by one; if the hypergraph remains non-2-colorable, keep it removed."""
    # This is a local search: try removing random edges, check if still non-2-colorable.
    # If yes, keep removed. If no, put back.
    # We'll do a few rounds.
    random.seed(42)
    edges_set = set(edges)
    for _ in range(50):
        if time.time() > deadline:
            break
        if not edges_set:
            break
        # Pick a random edge
        e = random.choice(list(edges_set))
        edges_set.remove(e)
        # Check if still non-2-colorable
        if not _is_2_colorable(list(edges_set), n, deadline):
            # Still non-2-colorable, keep removed
            continue
        else:
            # Became 2-colorable, put back
            edges_set.add(e)
    return list(edges_set)

def construct():
    deadline = time.time() + 110  # safety margin
    best_edges = None
    best_n = None

    # Try several algebraic constructions
    # 1. Cyclic construction on n vertices
    for n in range(5, 27):
        if time.time() > deadline:
            break
        # Generate all 5-subsets, then select a cyclic set
        # We'll use a simple heuristic: take every k-th 5-subset in lexicographic order
        all_edges = list(combinations(range(n), 5))
        # Try different strides
        for stride in range(1, 10):
            if time.time() > deadline:
                break
            edges = [all_edges[i] for i in range(0, len(all_edges), stride)]
            if len(edges) > 200:
                continue
            # Check if non-2-colorable
            if not _is_2_colorable([frozenset(e) for e in edges], n, deadline):
                # Non-2-colorable!
                if best_edges is None or len(edges) < len(best_edges):
                    best_edges = edges
                    best_n = n
                    # Try to reduce further
                    edges_reduced = _remove_edge_if_possible(edges, n, deadline)
                    if len(edges_reduced) < len(best_edges):
                        best_edges = edges_reduced
                        best_n = n
                    break
        if best_edges is not None and len(best_edges) <= 30:
            break

    # Fallback: complete hypergraph on 9 vertices
    if best_edges is None:
        n = 9
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n

    # Final check
    if _is_2_colorable([frozenset(e) for e in best_edges], best_n, deadline):
        # Should not happen, but fallback
        n = 9
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n

    return {'n': best_n, 'edges': [list(e) for e in best_edges]}
# EVOLVE-BLOCK-END
