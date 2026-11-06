"""Clique partition of two fixed dense random graphs: branch-and-bound with bitsets."""
import time
import random

_DEADLINE = 140.0
_START = time.monotonic()


def _elapsed():
    return time.monotonic() - _START


def _random_graph(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges


def _validate(n, edges, cliques):
    edge_set = set(edges)
    seen = set()
    for c in cliques:
        if len(c) < 2:
            return False
        k = len(c)
        for i in range(k):
            a = c[i]
            for j in range(i + 1, k):
                b = c[j]
                e = (a, b) if a < b else (b, a)
                if e not in edge_set:
                    return False
                if e in seen:
                    return False
                seen.add(e)
    return seen == edge_set


def _greedy_solve(n, edges):
    """Fast greedy maximal clique packing as initial solution / fallback."""
    if not edges:
        return []
    adj = [0] * n
    for (u, v) in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Precompute edge triangle counts for ordering
    edge_tri = {}
    for (u, v) in edges:
        a, b = min(u, v), max(u, v)
        cand = adj[a] & adj[b]
        cand &= ~((1 << (b + 1)) - 1)
        edge_tri[(a, b)] = cand.bit_count()

    sorted_edges = sorted(edges, key=lambda e: edge_tri.get((min(e[0], e[1]), max(e[0], e[1])), 0))

    used = set()
    cliques = []
    for (u, v) in sorted_edges:
        if (u, v) in used:
            continue
        max_uv = max(u, v)
        clique = [u, v]
        cand = adj[u] & adj[v]
        cand &= ~((1 << (max_uv + 1)) - 1)
        while cand:
            best_w = -1
            best_deg = float('inf')
            temp = cand
            while temp:
                w = (temp & -temp).bit_length() - 1
                deg = adj[w].bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_w = w
                temp &= temp - 1
            if best_w < 0:
                break
            can_add = all((adj[best_w] >> x) & 1 for x in clique)
            if can_add:
                clique.append(best_w)
                cand = cand & adj[best_w]
                new_max = max(clique)
                cand &= ~((1 << (new_max + 1)) - 1)
            else:
                cand &= ~(1 << best_w)
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                a, b = clique[i], clique[j]
                used.add((a, b) if a < b else (b, a))
        cliques.append(clique)

    for (u, v) in edges:
        if (u, v) not in used:
            cliques.append([u, v])
    return cliques


def _max_clique_size_bitset(adj, n, mask):
    """Bron-Kerbosch with pivot to find max clique size within mask."""
    best = [0]

    def bronk(R, P, X):
        if _elapsed() > _DEADLINE:
            return
        if not P and not X:
            if len(R) > best[0]:
                best[0] = len(R)
            return
        # Pivot: choose vertex in P|X with max neighbors in P
        U = P | X
        if U:
            pivot = (U & -U).bit_length() - 1
            candidates = P & ~adj[pivot]
        else:
            candidates = P
        temp = candidates
        while temp:
            v_bit = temp & -temp
            v = v_bit.bit_length() - 1
            temp ^= v_bit
            bronk(R | v_bit, P & adj[v], X & adj[v])
            P ^= v_bit
            X |= v_bit

    bronk(0, mask, 0)
    return best[0]


def _lower_bound_cliques(remaining_edges, adj, n):
    """Lower bound: each clique of size k covers k*(k-1)/2 edges.
    Use max clique size as upper bound on edges per clique."""
    if not remaining_edges:
        return 0
    # Max clique size gives upper bound on edges per clique
    # Build mask of vertices incident to remaining edges
    vert_mask = 0
    for (u, v) in remaining_edges:
        vert_mask |= (1 << u) | (1 << v)
    mc = _max_clique_size_bitset(adj, n, vert_mask)
    if mc < 2:
        return len(remaining_edges)
    edges_per_clique = mc * (mc - 1) // 2
    return (len(remaining_edges) + edges_per_clique - 1) // edges_per_clique


def _branch_and_bound(n, edges, time_limit):
    """Branch and bound: recursively find clique for first uncovered edge."""
    if not edges:
        return []

    adj = [0] * n
    for (u, v) in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    all_edges_set = set(edges)
    # Greedy initial solution
    best_cliques = _greedy_solve(n, edges)
    best_count = len(best_cliques)

    # Sort edges for deterministic branching
    edge_list = sorted(edges)

    # Precompute all maximal cliques containing each edge? Too expensive.
    # Instead: for the first uncovered edge, enumerate all cliques containing it.

    def enumerate_cliques_containing_edge(u, v, adj, n):
        """Enumerate all cliques containing edge (u,v) using Bron-Kerbosch."""
        cliques = []
        max_uv = max(u, v)
        # Common neighbors with index > max_uv
        P = adj[u] & adj[v]
        P &= ~((1 << (max_uv + 1)) - 1)
        X = 0
        R = (1 << u) | (1 << v)

        def bk(R, P, X):
            if _elapsed() > time_limit:
                return
            if not P and not X:
                cliques.append(R)
                return
            U = P | X
            if U:
                pivot = (U & -U).bit_length() - 1
                candidates = P & ~adj[pivot]
            else:
                candidates = P
            temp = candidates
            while temp:
                v_bit = temp & -temp
                v = v_bit.bit_length() - 1
                temp ^= v_bit
                bk(R | v_bit, P & adj[v], X & adj[v])
                P ^= v_bit
                X |= v_bit

        bk(R, P, X)
        return cliques

    # State: set of uncovered edges
    uncovered = set(edge_list)
    # Order edges: pick most constrained first (fewest cliques containing it)
    # For B&B, we pick the first uncovered edge in sorted order

    def solve(uncovered, current_cliques, current_count):
        nonlocal best_cliques, best_count
        if _elapsed() > time_limit:
            return
        if not uncovered:
            if current_count < best_count:
                best_cliques = current_cliques[:]
                best_count = current_count
            return
        # Lower bound
        lb = current_count + _lower_bound_cliques(list(uncovered), adj, n)
        if lb >= best_count:
            return

        # Pick first uncovered edge (sorted)
        edge = min(uncovered)
        u, v = edge
        # Enumerate all cliques containing this edge
        cliques = enumerate_cliques_containing_edge(u, v, adj, n)
        if not cliques:
            # Should not happen since (u,v) is an edge
            cliques = [[u, v]]

        # Sort by size descending (larger cliques first for better pruning)
        cliques.sort(key=lambda c: -c.bit_count())

        for clique_mask in cliques:
            if _elapsed() > time_limit:
                return
            # Compute edges covered by this clique
            verts = []
            temp = clique_mask
            while temp:
                v_bit = temp & -temp
                verts.append(v_bit.bit_length() - 1)
                temp ^= v_bit
            covered = set()
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    a, b = verts[i], verts[j]
                    e = (a, b) if a < b else (b, a)
                    if e in uncovered:
                        covered.add(e)
            if not covered:
                continue
            new_uncovered = uncovered - covered
            # Recurse
            current_cliques.append(verts)
            solve(new_uncovered, current_cliques, current_count + 1)
            current_cliques.pop()

    solve(uncovered, [], 0)
    return best_cliques


def _solve_instance(n, p, seed):
    edges = _random_graph(n, p, seed)
    if not edges:
        return []

    # Get greedy solution first
    greedy_cliques = _greedy_solve(n, edges)
    greedy_count = len(greedy_cliques)

    # Try branch-and-bound with time limit
    remaining_time = _DEADLINE - _elapsed() - 5.0  # leave 5s for validation and other instance
    if remaining_time > 2.0:
        bnb_cliques = _branch_and_bound(n, edges, _START + remaining_time)
        if len(bnb_cliques) < greedy_count:
            result = bnb_cliques
        else:
            result = greedy_cliques
    else:
        result = greedy_cliques

    if not _validate(n, edges, result):
        result = [[u, v] for (u, v) in edges]
    return result


def construct():
    results = []
    for (n, p, seed) in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
