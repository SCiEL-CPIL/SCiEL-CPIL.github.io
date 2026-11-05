"""Search for a small non-2-colorable 5-uniform hypergraph."""
import time
import random
from itertools import combinations

DEADLINE = 145.0
START = time.time()


def _is_non_2colorable(n, edges, time_limit=2.0):
    """Backtracking with unit propagation. Returns True if NOT 2-colorable."""
    m = len(edges)
    if m == 0:
        return False

    # Precompute vertex -> edge indices
    vert_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vert_edges[v].append(i)

    # State: assign[v] = -1 unassigned, 0 or 1
    # c0[e] = count of color 0 in edge e, c1[e] = count of color 1
    assign = [-1] * n
    c0 = [0] * m
    c1 = [0] * m

    def propagate(assign, c0, c1):
        """Returns False if contradiction (monochromatic edge). Modifies in place."""
        changed = True
        while changed:
            changed = False
            for e in range(m):
                if c0[e] == 5 or c1[e] == 5:
                    return False
                # If 4 of one color, force the 5th
                if c0[e] == 4 and c1[e] == 0:
                    for v in edges[e]:
                        if assign[v] == -1:
                            assign[v] = 1
                            c1[e] += 1
                            for e2 in vert_edges[v]:
                                if e2 != e:
                                    c1[e2] += 1
                            changed = True
                            break
                elif c1[e] == 4 and c0[e] == 0:
                    for v in edges[e]:
                        if assign[v] == -1:
                            assign[v] = 0
                            c0[e] += 1
                            for e2 in vert_edges[v]:
                                if e2 != e:
                                    c0[e2] += 1
                            changed = True
                            break
        return True

    def backtrack(assign, c0, c1):
        # Find an unassigned vertex with minimum unassigned count in its edges
        best_v = -1
        best_score = float('inf')
        for v in range(n):
            if assign[v] != -1:
                continue
            min_unassigned = 5
            for e in vert_edges[v]:
                unassigned_in_e = 5 - c0[e] - c1[e]
                if unassigned_in_e < min_unassigned:
                    min_unassigned = unassigned_in_e
            if min_unassigned < best_score:
                best_score = min_unassigned
                best_v = v
                if best_score == 1:
                    break

        if best_v == -1:
            # All assigned, no contradiction found => valid 2-coloring exists
            return False

        for color in (0, 1):
            new_assign = assign[:]
            new_c0 = c0[:]
            new_c1 = c1[:]
            new_assign[best_v] = color
            for e in vert_edges[best_v]:
                if color == 0:
                    new_c0[e] += 1
                else:
                    new_c1[e] += 1
            if not propagate(new_assign, new_c0, new_c1):
                continue
            if backtrack(new_assign, new_c0, new_c1):
                return True
        return True

    if not propagate(assign, c0, c1):
        return True
    return backtrack(assign, c0, c1)


def _try_build(seed, n_vertices, max_edges=60):
    """Build a random hypergraph incrementally, stop when non-2-colorable."""
    rng = random.Random(seed)
    all_possible = list(combinations(range(n_vertices), 5))
    rng.shuffle(all_possible)

    edges = []
    for e in all_possible:
        if len(edges) >= max_edges:
            break
        if time.time() > DEADLINE - 5:
            break
        edges.append(list(e))
        if _is_non_2colorable(n_vertices, edges):
            return n_vertices, edges
    return None


def construct():
    best = None
    best_edges = float('inf')
    seed = 0

    # Try various vertex counts
    for n_v in [10, 11, 12, 13, 14, 15, 16]:
        while time.time() < DEADLINE - 5:
            seed += 1
            result = _try_build(seed, n_v)
            if result is not None:
                n, edges = result
                if len(edges) < best_edges:
                    best_edges = len(edges)
                    best = {'n': n, 'edges': edges}
                    if best_edges <= 51:
                        break
        if best_edges <= 51:
            break

    if best is None:
        # Fallback: complete 5-uniform on 9 vertices (126 edges)
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    return best
