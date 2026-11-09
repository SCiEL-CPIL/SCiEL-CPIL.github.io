"""Combined: Tomita greedy coloring upper bound + Bron-Kerbosch with degeneracy ordering."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _degeneracy_order(adj):
    n = P['n']
    remaining = list(range(n))
    remaining_set = set(remaining)
    order = []
    while remaining:
        min_deg = min(len(adj[v] & remaining_set) for v in remaining)
        v = next(v for v in remaining if len(adj[v] & remaining_set) == min_deg)
        order.append(v)
        remaining.remove(v)
        remaining_set.discard(v)
    return order


def _color_sort(adj, candidates):
    """Greedy coloring of the candidate set; returns (color, order) where
    order[i] is the vertex whose color is color[i]. The maximum color value is
    an upper bound on the clique size within candidates."""
    if not candidates:
        return [], []
    # Order candidates by degree (descending) within the candidate set for a tighter bound
    cand_list = sorted(candidates, key=lambda v: -len(adj[v] & candidates))
    color_of = {}
    order = []
    max_color = 0
    # color classes: list of sets
    classes = []
    for v in cand_list:
        # Find the smallest color class v can join (no neighbor already in it)
        placed = False
        for ci, cls in enumerate(classes):
            if not (adj[v] & cls):
                cls.add(v)
                color_of[v] = ci + 1
                order.append(v)
                placed = True
                break
        if not placed:
            new_cls = {v}
            classes.append(new_cls)
            color_of[v] = len(classes)
            order.append(v)
            if len(classes) > max_color:
                max_color = len(classes)
    # color array aligned with order
    colors = [color_of[v] for v in order]
    return colors, order


def _expand(adj, R, P_set, X, best, best_size, deadline, node_counter):
    """Bron-Kerbosch with Tomita coloring bound. Mutates P_set and X; R is a set."""
    node_counter[0] += 1
    if time.time() > deadline:
        return best, best_size

    if not P_set:
        if not X:
            if len(R) > best_size:
                best = list(R)
                best_size = len(R)
        return best, best_size

    # Prune: even using every candidate we can't beat best
    if len(R) + len(P_set) <= best_size:
        return best, best_size

    # Coloring upper bound
    colors, order = _color_sort(adj, P_set)
    if not order:
        return best, best_size
    if len(R) + colors[-1] <= best_size:
        return best, best_size

    # Process vertices in color order (reverse of assignment = higher color first)
    # We iterate in the order list; the color of order[i] is colors[i].
    # Tomita: iterate vertices in the order produced, and at step i the bound is
    # len(R) + colors[i] (since colors[i] is the max color among order[i:] ... actually
    # colors is non-decreasing in order). We iterate in reverse order so that the
    # remaining set shrinks and the bound tightens.
    # Simpler correct approach: iterate order in the given (non-decreasing color) order,
    # and at each step the upper bound for the remaining candidates is colors[i].
    for i in range(len(order) - 1, -1, -1):
        v = order[i]
        if v not in P_set:
            continue
        if time.time() > deadline:
            return best, best_size
        # Bound: R plus at most colors[i] more (max color in order[i:])
        if len(R) + colors[i] <= best_size:
            break
        new_R = R | {v}
        new_P = P_set & adj[v]
        new_X = X & adj[v]
        best, best_size = _expand(adj, new_R, new_P, new_X, best, best_size, deadline, node_counter)
        P_set.discard(v)
        X.add(v)
    return best, best_size


def _max_clique(adj):
    n = P['n']
    deadline = time.time() + 50.0  # safety margin under 90s wall clock
    node_counter = [0]

    order = _degeneracy_order(adj)
    # Map vertex -> position in degeneracy order
    pos = {v: i for i, v in enumerate(order)}

    best = []
    best_size = 0
    for i, v in enumerate(order):
        if time.time() > deadline:
            break
        # P = neighbors of v that appear after v in the degeneracy order
        P_set = set()
        for w in adj[v]:
            if pos.get(w, -1) > i:
                P_set.add(w)
        # X = neighbors of v that appear before v (for correctness of max clique)
        X = set()
        for w in adj[v]:
            if pos.get(w, -1) < i:
                X.add(w)
        best, best_size = _expand(adj, {v}, P_set, X, best, best_size, deadline, node_counter)

    if not best:
        # Fallback: full Bron-Kerbosch from scratch
        best, best_size = _expand(adj, set(), set(range(n)), set(), best, best_size, deadline, node_counter)

    return best


def _bfs_path(adj, u, v):
    visited = {u}
    queue = deque([(u, [u])])
    while queue:
        node, path = queue.popleft()
        if node == v:
            return path
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [nb]))
    return None


def construct():
    adj = _build_adj()
    clique = _max_clique(adj)

    if len(clique) < 2:
        u, v = P['edges'][0]
        clique = [u, v]

    clique = sorted(clique)
    branches = list(clique)

    # Since clique vertices are pairwise adjacent, direct edge is a valid simple path.
    paths = []
    for u, v in combinations(clique, 2):
        if v in adj[u]:
            paths.append([u, v])
        else:
            p = _bfs_path(adj, u, v)
            paths.append(p if p is not None else [u, v])

    return {'branches': branches, 'paths': paths}
# EVOLVE-BLOCK-END
