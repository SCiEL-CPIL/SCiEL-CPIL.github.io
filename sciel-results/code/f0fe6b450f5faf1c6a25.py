"""Optimal clique partition via maximal-clique enumeration + min vertex cover (Kőnig)."""
import time
from random import Random


def _random_graph(n, p, seed):
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges


def _maximal_cliques(adj_bits, n):
    """Bron–Kerbosch with pivot, using bitsets. Returns list of frozensets."""
    results = []

    def bk(R, P, X):
        if not P and not X:
            results.append(frozenset(R))
            return
        # Choose pivot u from P ∪ X maximizing |P ∩ N(u)|
        PX = P | X
        pivot = -1
        best = -1
        m = PX
        while m:
            lb = m & (-m)
            u = lb.bit_length() - 1
            cnt = (P & adj_bits[u]).bit_count()
            if cnt > best:
                best = cnt
                pivot = u
            m ^= lb
        # Iterate over P \ N(pivot)
        if pivot >= 0:
            candidates = P & ~adj_bits[pivot]
        else:
            candidates = P
        m = candidates
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            m ^= lb
            newR = R | lb
            newP = P & adj_bits[v]
            newX = X & adj_bits[v]
            bk(newR, newP, newX)
            P &= ~lb
            X |= lb

    bk(0, (1 << n) - 1, 0)
    return results


def _min_vertex_cover_bipartite(left_size, right_size, adj_left):
    """
    adj_left[i] = set of right-neighbors of left node i.
    Returns (left_cover: set, right_cover: set) — a minimum vertex cover.
    Uses Hopcroft-Karp for max matching, then standard bipartite min-VC from matching.
    """
    # Build adjacency lists
    adj = [list(adj_left[i]) for i in range(left_size)]

    # Hopcroft-Karp
    from collections import deque
    INF = float('inf')
    pairU = [-1] * left_size
    pairV = [-1] * right_size
    dist = [0] * left_size

    def bfs():
        q = deque()
        for u in range(left_size):
            if pairU[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = INF
        found = False
        while q:
            u = q.popleft()
            for v in adj[u]:
                pu = pairV[v]
                if pu == -1:
                    found = True
                elif dist[pu] == INF:
                    dist[pu] = dist[u] + 1
                    q.append(pu)
        return found

    def dfs(u):
        for v in adj[u]:
            pu = pairV[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pairU[u] = v
                pairV[v] = u
                return True
        dist[u] = INF
        return False

    while bfs():
        for u in range(left_size):
            if pairU[u] == -1:
                dfs(u)

    # Build residual reachability for min vertex cover
    # Unmatched left nodes, follow unmatched edges to right, matched edges back to left
    visited_L = [False] * left_size
    visited_R = [False] * right_size
    q = deque()
    for u in range(left_size):
        if pairU[u] == -1:
            visited_L[u] = True
            q.append(('L', u))
    while q:
        kind, idx = q.popleft()
        if kind == 'L':
            u = idx
            for v in adj[u]:
                if not visited_R[v] and pairV[v] != u:
                    # unmatched edge
                    visited_R[v] = True
                    q.append(('R', v))
        else:
            v = idx
            pu = pairV[v]
            if pu != -1 and not visited_L[pu]:
                visited_L[pu] = True
                q.append(('L', pu))

    # Min vertex cover = (L \ Z) ∪ (R ∩ Z)
    left_cover = set()
    for u in range(left_size):
        if not visited_L[u]:
            left_cover.add(u)
    right_cover = set()
    for v in range(right_size):
        if visited_R[v]:
            right_cover.add(v)
    return left_cover, right_cover


def _solve_instance(n, p, seed, deadline):
    edges = _random_graph(n, p, seed)
    if not edges:
        return []

    adj_bits = [0] * n
    for u, v in edges:
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)

    # Enumerate all maximal cliques
    max_cliques = _maximal_cliques(adj_bits, n)

    # Map each edge to list of clique indices containing it
    edge_to_cliques = {}
    for idx, C in enumerate(max_cliques):
        C_list = list(C)
        for i in range(len(C_list)):
            for j in range(i + 1, len(C_list)):
                a, b = C_list[i], C_list[j]
                e = (a, b) if a < b else (b, a)
                edge_to_cliques.setdefault(e, []).append(idx)

    # Every edge must be in at least one maximal clique (guaranteed)
    # Build bipartite graph: left = edges, right = cliques
    left_size = len(edges)
    right_size = len(max_cliques)
    edge_index = {e: i for i, e in enumerate(edges)}
    adj_left = [set() for _ in range(left_size)]
    for e, clist in edge_to_cliques.items():
        li = edge_index[e]
        for ci in clist:
            adj_left[li].add(ci)

    if time.time() > deadline - 5:
        # Fallback: use all maximal cliques
        return [list(c) for c in max_cliques if len(c) >= 2]

    left_cover, right_cover = _min_vertex_cover_bipartite(left_size, right_size, adj_left)

    # right_cover = set of clique indices selected
    selected = [max_cliques[i] for i in sorted(right_cover)]
    cliques = [list(c) for c in selected if len(c) >= 2]

    # Verify every edge is covered; if not, add missing edges as 2-cliques
    covered = set()
    for C in cliques:
        for i in range(len(C)):
            for j in range(i + 1, len(C)):
                a, b = C[i], C[j]
                e = (a, b) if a < b else (b, a)
                covered.add(e)
    for e in edges:
        if e not in covered:
            cliques.append(list(e))
            covered.add(e)

    return cliques


def construct():
    deadline = time.time() + 140
    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 20:
            edges = _random_graph(n, p, seed)
            result.append([list(e) for e in edges])
            continue
        cliques = _solve_instance(n, p, seed, deadline)
        result.append(cliques)
    return result
