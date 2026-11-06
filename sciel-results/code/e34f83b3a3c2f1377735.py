"""New design: 4-partite core + cross-connected leftover, refined by simulated
annealing with exact verification. Returns a K4-free graph on 18 vertices with
a small maximum triangle-free induced subgraph."""
import time
import random

_N = 18
_DEADLINE = 130.0   # wall-clock guard (evaluator kills at 150s)


def _max_triangle_free_size(adj):
    """Exact branch-and-bound: largest triangle-free induced subgraph.
    adj: list of 18 int bitmasks. Returns (size, frozenset_of_vertices)."""
    best_size = 0
    best_set = frozenset()

    def can_add(S, v):
        """v not in S; v can be added without forming a triangle."""
        nb = adj[v] & S
        # nb must be an independent set
        x = nb
        while x:
            lsb = x & (-x)
            u = lsb.bit_length() - 1
            if adj[u] & (nb ^ lsb):
                return False
            x ^= lsb
        return True

    def search(S, size):
        nonlocal best_size, best_set
        if size > best_size:
            best_size = size
            best_set = frozenset(i for i in range(_N) if (S >> i) & 1)
        cand = 0
        for v in range(_N):
            if (S >> v) & 1:
                continue
            if can_add(S, v):
                cand |= (1 << v)
        if cand == 0:
            return
        # order candidates by fewest neighbors in S (more constrained first)
        order = []
        x = cand
        while x:
            lsb = x & (-x)
            v = lsb.bit_length() - 1
            order.append((bin(adj[v] & S).count('1'), v))
            x ^= lsb
        order.sort()
        for _, v in order:
            search(S | (1 << v), size + 1)

    search(0, 0)
    return best_size, best_set


def _is_k4_free(adj):
    for i in range(_N):
        for j in range(i + 1, _N):
            if not ((adj[i] >> j) & 1):
                continue
            common = adj[i] & adj[j]
            # common must be a clique
            x = common
            while x:
                lsb = x & (-x)
                u = lsb.bit_length() - 1
                if (adj[u] & common) != (common ^ lsb):
                    return False
                x ^= lsb
    return True


def _build_adj(edges):
    adj = [0] * _N
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj


def _adj_to_edges(adj):
    edges = []
    for i in range(_N):
        for j in range(i + 1, _N):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return edges


def _initial_graph():
    # parts: A=6, B=4, C=3, D=2, E=3
    A = list(range(0, 6))
    B = list(range(6, 10))
    C = list(range(10, 13))
    D = list(range(13, 15))
    E = list(range(15, 18))
    edges = []
    # core: complete 4-partite on A,B,C,D
    for a in A:
        for b in B:
            edges.append([a, b])
        for c in C:
            edges.append([a, c])
        for d in D:
            edges.append([a, d])
    for b in B:
        for c in C:
            edges.append([b, c])
        for d in D:
            edges.append([b, d])
    for c in C:
        for d in D:
            edges.append([c, d])
    # E: each E vertex connected to all A and all B (10 core neighbors), E independent
    for e in E:
        for a in A:
            edges.append([e, a])
        for b in B:
            edges.append([e, b])
    return edges


def construct():
    start = time.time()
    rng = random.Random(12345)

    edges = _initial_graph()
    adj = _build_adj(edges)
    if not _is_k4_free(adj):
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        edges = [[a, b] for i in range(3) for j in range(i + 1, 3)
                 for a in parts[i] for b in parts[j]]
        adj = _build_adj(edges)

    best_size, _ = _max_triangle_free_size(adj)
    best_edges = [list(e) for e in edges]

    T = 2.0
    cooling = 0.9995
    while time.time() < _DEADLINE:
        u = rng.randrange(_N)
        v = rng.randrange(_N)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        present = (adj[u] >> v) & 1
        new_adj = [a for a in adj]
        if present:
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
        else:
            new_adj[u] |= 1 << v
            new_adj[v] |= 1 << u
        if not _is_k4_free(new_adj):
            continue
        new_size, _ = _max_triangle_free_size(new_adj)
        if new_size < best_size:
            best_size = new_size
            best_edges = _adj_to_edges(new_adj)
            adj = new_adj
            T = 2.0
        elif new_size == best_size and rng.random() < 0.05:
            adj = new_adj
        else:
            T *= cooling
        if best_size <= 8:
            break
    return best_edges
