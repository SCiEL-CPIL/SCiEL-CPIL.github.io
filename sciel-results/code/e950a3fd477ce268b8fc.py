"""Search for a small non-2-colorable 5-uniform hypergraph."""
import random
import time
from itertools import combinations

DEADLINE = 150.0
START = time.time()

def _is_non_2colorable(n, edges):
    """Backtracking with unit propagation. Returns True if NOT 2-colorable."""
    m = len(edges)
    if m == 0:
        return False
    # adjacency: for each vertex, list of edge indices containing it
    vert_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vert_edges[v].append(i)
    # For unit propagation: if an edge has 4 vertices same color, force the 5th.
    # Represent assignment: -1 unassigned, 0/1 color.
    assign = [-1] * n
    # edge color counts: for each edge, count of 0s and 1s among assigned vertices
    e_count0 = [0] * m
    e_count1 = [0] * m

    def propagate(assign, e_count0, e_count1, queue):
        # queue: list of (edge_idx, vertex) where vertex was just assigned
        # We'll use a simple approach: repeatedly scan for unit edges.
        # For efficiency, maintain a list of edges that are "almost monochromatic".
        changed = True
        while changed:
            changed = False
            for ei in range(m):
                c0, c1 = e_count0[ei], e_count1[ei]
                # unassigned count in this edge
                unassigned = 5 - c0 - c1
                if c0 == 5 or c1 == 5:
                    # monochromatic edge -> contradiction
                    return False
                if c0 == 4 and unassigned == 1:
                    # force the unassigned vertex to color 1
                    for v in edges[ei]:
                        if assign[v] == -1:
                            assign[v] = 1
                            e_count1[ei] += 1
                            for ej in vert_edges[v]:
                                if ej != ei:
                                    e_count1[ej] += 1
                            changed = True
                            break
                elif c1 == 4 and unassigned == 1:
                    # force the unassigned vertex to color 0
                    for v in edges[ei]:
                        if assign[v] == -1:
                            assign[v] = 0
                            e_count0[ei] += 1
                            for ej in vert_edges[v]:
                                if ej != ei:
                                    e_count0[ej] += 1
                            changed = True
                            break
        return True

    def backtrack(assign, e_count0, e_count1):
        # Check if all assigned consistently (unit propagation already done)
        # Find an unassigned vertex to branch on
        # Heuristic: pick vertex in an edge with most assigned vertices
        best_v = -1
        best_score = -1
        for v in range(n):
            if assign[v] != -1:
                continue
            # count how many edges are "tight" for this vertex
            score = 0
            for ei in vert_edges[v]:
                c0, c1 = e_count0[ei], e_count1[ei]
                unassigned = 5 - c0 - c1
                if unassigned == 1:
                    score += 10
                elif unassigned == 2:
                    score += 5
                elif unassigned == 3:
                    score += 1
            if score > best_score:
                best_score = score
                best_v = v
        if best_v == -1:
            # all assigned, check no monochromatic edge
            for ei in range(m):
                if e_count0[ei] == 5 or e_count1[ei] == 5:
                    return True  # contradiction found
            return False  # valid 2-coloring exists
        # Try both colors
        for color in (0, 1):
            new_assign = assign[:]
            new_c0 = e_count0[:]
            new_c1 = e_count1[:]
            new_assign[best_v] = color
            if color == 0:
                new_c0[ei] = new_c0[ei] + 1  # placeholder, fix below
            # Redo properly
            new_assign = assign[:]
            new_c0 = e_count0[:]
            new_c1 = e_count1[:]
            new_assign[best_v] = color
            for ei in vert_edges[best_v]:
                if color == 0:
                    new_c0[ei] += 1
                else:
                    new_c1[ei] += 1
            if not propagate(new_assign, new_c0, new_c1, None):
                continue
            # After propagation, check for monochromatic edges
            mono = False
            for ei in range(m):
                if new_c0[ei] == 5 or new_c1[ei] == 5:
                    mono = True
                    break
            if mono:
                continue
            if backtrack(new_assign, new_c0, new_c1):
                return True
        return False

    # Initial propagation
    if not propagate(assign, e_count0, e_count1, None):
        return True
    for ei in range(m):
        if e_count0[ei] == 5 or e_count1[ei] == 5:
            return True
    return backtrack(assign, e_count0, e_count1)


def _try_build(seed, max_edges=60, max_vertices=26):
    """Build a random hypergraph incrementally, stop when non-2-colorable."""
    rng = random.Random(seed)
    # Start with a small number of vertices, add edges
    # We'll pick edges from a pool of candidate edges on up to max_vertices vertices
    # Strategy: use a moderate number of vertices (e.g., 12-18) and add random edges
    n_vertices = rng.randint(10, 18)
    all_possible = list(combinations(range(n_vertices), 5))
    rng.shuffle(all_possible)
    edges = []
    for e in all_possible:
        if len(edges) >= max_edges:
            break
        edges.append(list(e))
        if _is_non_2colorable(n_vertices, edges):
            return n_vertices, edges
    return None


def construct():
    best = None
    best_edges = float('inf')
    seed = 0
    while time.time() < DEADLINE - 10:
        seed += 1
        result = _try_build(seed)
        if result is not None:
            n, edges = result
            if len(edges) < best_edges:
                best_edges = len(edges)
                best = {'n': n, 'edges': edges}
                if best_edges <= 51:
                    break
        if seed % 50 == 0:
            # occasionally try different vertex counts via different strategy
            pass
    if best is None:
        # Fallback: complete 5-uniform on 9 vertices
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    return best
