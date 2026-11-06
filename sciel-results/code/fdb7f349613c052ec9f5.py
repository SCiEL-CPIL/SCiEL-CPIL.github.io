"""Combined approach: use the greedy triangle-free maximization (inspiration 1)
seeded with the Grötzsch graph, then apply a bounded local-improvement / edge-swap
search to try to exceed the greedy optimum while keeping the graph not 3-colourable."""
import time
import random

DEADLINE_MARGIN = 5.0
HARD_LIMIT = 145.0


def _is_3_colorable_fast(adj, n):
    """
    Bitmask-based 3-colourability check with degree ordering and forward checking.
    Returns True if the graph is 3-colourable, False otherwise.
    """
    # Order by descending degree
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i

    colors = [-1] * n

    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        # Forbidden colours from already-coloured neighbours
        forbidden = 0
        for nb in adj[v]:
            c = colors[nb]
            if c != -1:
                forbidden |= (1 << c)
        for c in range(3):
            if forbidden & (1 << c):
                continue
            colors[v] = c
            if backtrack(idx + 1):
                return True
            colors[v] = -1
        return False

    return backtrack(0)


def construct():
    start_time = time.time()
    deadline = start_time + HARD_LIMIT
    n = 16

    adj = [set() for _ in range(n)]

    def add_edge(u, v):
        if u != v and v not in adj[u]:
            adj[u].add(v)
            adj[v].add(u)

    def remove_edge(u, v):
        adj[u].discard(v)
        adj[v].discard(u)

    def can_add_edge(u, v):
        if v in adj[u]:
            return False
        # Triangle-free: no common neighbour
        if adj[u] & adj[v]:
            return False
        return True

    def edge_count():
        return sum(len(adj[u]) for u in range(n)) // 2

    def to_edge_list():
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    # --- Build the Grötzsch graph (Mycielski of C5) on vertices 0..10 ---
    # C5: 0-1-2-3-4-0
    for i in range(5):
        add_edge(i, (i + 1) % 5)
    # Mycielski layer 5..9: vertex 5+i connects to neighbours of i in C5
    for i in range(5):
        add_edge(5 + i, (i + 1) % 5)
        add_edge(5 + i, (i - 1) % 5)
    # Universal vertex 10 connects to 5..9
    for i in range(5):
        add_edge(10, 5 + i)

    # --- Greedy: add as many edges as possible while keeping triangle-free ---
    # Since the base is not 3-colourable, any supergraph (that contains it) is also not 3-colourable.
    # So we just need to maximise edges under triangle-free constraint.
    changed = True
    while changed and time.time() < deadline - DEADLINE_MARGIN:
        changed = False
        for u in range(n):
            for v in range(u + 1, n):
                if can_add_edge(u, v):
                    add_edge(u, v)
                    changed = True

    best_edges = to_edge_list()
    best_count = len(best_edges)

    # --- Local improvement: try edge swaps to increase count ---
    # We can't simply add more (greedy exhausted), but maybe a different
    # triangle-free supergraph of the Grötzsch has more edges.
    # Strategy: randomly remove an edge, then greedily re-add, see if count improves.
    rng = random.Random(42)

    for _ in range(2000):
        if time.time() > deadline - DEADLINE_MARGIN:
            break

        # Pick a random existing edge to remove
        all_edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
        if not all_edges:
            break
        ru, rv = rng.choice(all_edges)
        remove_edge(ru, rv)

        # Greedily re-add
        for u in range(n):
            for v in range(u + 1, n):
                if can_add_edge(u, v):
                    add_edge(u, v)

        cnt = edge_count()
        if cnt > best_count:
            best_count = cnt
            best_edges = to_edge_list()

    return best_edges
