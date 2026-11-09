"""Clique partition solver: Bron-Kerbosch maximal cliques + greedy edge-disjoint
set packing + local split-and-re-tile improvement. Minimizes total number of cliques."""

import time

_DEADLINE = time.time() + 105.0  # safety margin under 120s wall-clock


def _random_edges(n, p, seed):
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges


def _build_adj(n, edges):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj


def _maximal_cliques(adj, n):
    # Bron-Kerbosch with pivot, using bitmasks.
    results = []
    full = (1 << n) - 1

    def bk(R, P, X):
        if not P and not X:
            results.append(R)
            return
        # pivot: vertex in P|X with max |P & N(w)|
        union = P | X
        if union:
            w = union.bit_length() - 1  # any vertex; refine below
            best = -1
            u = union
            while u:
                v = (u & -u).bit_length() - 1
                c = (P & adj[v]).bit_count()
                if c > best:
                    best = c
                    w = v
                u &= u - 1
            candidates = P & ~adj[w]
        else:
            candidates = P
        u = candidates
        while u:
            v = (u & -u).bit_length() - 1
            bit = 1 << v
            R2 = R | bit
            P2 = P & adj[v]
            X2 = X & adj[v]
            bk(R2, P2, X2)
            P &= ~bit
            X |= bit
            u &= u - 1

    bk(0, full, 0)
    return results


def _clique_edges(clique):
    out = []
    m = len(clique)
    for i in range(m):
        a = clique[i]
        for j in range(i + 1, m):
            out.append((a, clique[j]))
    return out


def _pack(cliques, edges, adj, n):
    # Greedy edge-disjoint set packing, largest-coverage-first.
    remaining = set(edges)
    if not remaining:
        return []
    # Precompute edge set for each candidate clique.
    cand = []
    for cl in cliques:
        if len(cl) < 2:
            continue
        es = _clique_edges(cl)
        if any(e in remaining for e in es):
            cand.append((cl, es))
    # Sort by coverage potential descending.
    cand.sort(key=lambda x: len(x[1]), reverse=True)
    chosen = []
    # We'll repeatedly pick the candidate covering the most remaining edges.
    import heapq
    # Use a simple loop: each iteration find best candidate.
    # For efficiency, maintain a counter of remaining edges per candidate.
    # Build edge -> list of candidate indices.
    edge_to_cand = {}
    for idx, (cl, es) in enumerate(cand):
        for e in es:
            edge_to_cand.setdefault(e, []).append(idx)
    remaining_count = [len([e for e in es if e in remaining]) for (cl, es) in cand]
    # Max-heap by remaining_count.
    heap = [(-remaining_count[i], i) for i in range(len(cand)) if remaining_count[i] > 0]
    heapq.heapify(heap)
    while heap and remaining:
        negc, idx = heapq.heappop(heap)
        c = -negc
        if c != remaining_count[idx] or c == 0:
            continue
        cl, es = cand[idx]
        # Pick this clique; remove its still-remaining edges.
        chosen.append(list(cl))
        for e in es:
            if e in remaining:
                remaining.discard(e)
                for j in edge_to_cand.get(e, ()):
                    if j != idx and remaining_count[j] > 0:
                        remaining_count[j] -= 1
        # Push updated counts.
        for j in range(len(cand)):
            if remaining_count[j] > 0:
                heapq.heappush(heap, (-remaining_count[j], j))
    # Cover leftover edges one by one.
    for e in sorted(remaining):
        chosen.append(list(e))
    return chosen


def _split_and_retile(partition, adj, n, budget):
    # Local improvement: for a clique with size>=4, try splitting into two
    # sub-cliques and re-tiling the freed edges. Accept if total count drops.
    if len(partition) < 2:
        return partition
    improved = True
    while improved:
        improved = False
        # Find a large clique to split.
        target = None
        for i, cl in enumerate(partition):
            if len(cl) >= 4:
                target = (i, cl)
                break
        if target is None:
            break
        i, cl = target
        m = len(cl)
        # Try splitting at various sizes.
        best = None
        # Build edges of the whole clique.
        all_edges = set(_clique_edges(cl))
        # Edges not in this clique (the rest of the graph).
        # For each split size k (2..m-2), partition cl into A (size k) and B (size m-k),
        # both must be cliques (they are, since cl is a clique).
        # Freed edges = all_edges minus edges within A and within B.
        # Re-tile freed edges greedily using candidate cliques (from maximal cliques)
        # that are subsets of... actually we re-tile freed edges with any clique.
        # Simplify: candidate re-tile cliques = maximal cliques that only use vertices
        # in cl (so we don't steal edges from other partition cliques).
        # We'll precompute this per call.
        for k in range(2, m - 1):
            # Choose A = first k, B = rest (a heuristic split).
            A = cl[:k]
            B = cl[k:]
            edgesA = set(_clique_edges(A))
            edgesB = set(_clique_edges(B))
            freed = all_edges - edgesA - edgesB
            if not freed:
                continue
            # Candidate cliques: maximal cliques subset of cl.
            # We'll enumerate all cliques of cl that cover freed edges greedily.
            # Simpler: re-tile freed edges using sub-cliques of cl.
            # Generate all cliques of cl of size>=2 that are subsets.
            subcliques = []
            from itertools import combinations
            for size in range(2, m + 1):
                for combo in combinations(cl, size):
                    subcliques.append(list(combo))
            # Greedy pack freed edges.
            remaining = set(freed)
            subcliques.sort(key=lambda c: len(_clique_edges(c)), reverse=True)
            new_local = []
            for sc in subcliques:
                if not remaining:
                    break
                es = _clique_edges(sc)
                cover = [e for e in es if e in remaining]
                if not cover:
                    continue
                new_local.append(list(sc))
                for e in cover:
                    remaining.discard(e)
            for e in sorted(remaining):
                new_local.append(list(e))
            # New total = len(partition) - 1 (remove cl) + 2 (A,B) + len(new_local)
            new_total = len(partition) - 1 + 2 + len(new_local)
            if new_total < len(partition):
                best = (k, A, B, new_local, new_total)
                break  # take first improving split
        if best is not None:
            k, A, B, new_local, new_total = best
            # Replace partition[i] with A, B, and new_local.
            new_partition = []
            for j, clj in enumerate(partition):
                if j == i:
                    new_partition.append(list(A))
                    new_partition.append(list(B))
                    new_partition.extend(new_local)
                else:
                    new_partition.append(clj)
            partition = new_partition
            improved = True
            if time.time() > budget:
                break
    return partition


def _solve(n, p, seed, budget):
    edges = _random_edges(n, p, seed)
    adj = _build_adj(n, edges)
    # Enumerate maximal cliques.
    mcs = _maximal_cliques(adj, n)
    # Ensure we include all edges as singletons as fallback.
    partition = _pack(mcs, edges, adj, n)
    # Local improvement.
    partition = _split_and_retile(partition, adj, n, budget)
    return partition


def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        remaining_budget = _DEADLINE - 5.0
        partition = _solve(n, p, seed, remaining_budget)
        results.append(partition)
    return results
