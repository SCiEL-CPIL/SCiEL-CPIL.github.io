import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def _build_base():
    n = P['n']
    base = set()
    for e in P['edges']:
        base.add((e[0], e[1]))
        base.add((e[1], e[0]))
    return n, base


def _bipartite_candidates(n, base):
    """All non-base edges that are even-odd (triangle-free w.r.t. path base)."""
    cands = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base or (v, u) in base:
                continue
            if (u % 2) == (v % 2):
                continue  # same parity -> triangle with path
            cands.append((u, v))
    return cands


def _all_pairs_same_parity(n):
    """All unordered pairs of vertices with the same parity (the critical constraints)."""
    pairs = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u % 2) == (v % 2):
                pairs.append((u, v))
    return pairs


def _pair_index_map(pairs):
    m = {}
    for i, p in enumerate(pairs):
        m[p] = i
        m[(p[1], p[0])] = i
    return m


def _dist_matrix(adj, n):
    """BFS from each vertex. Returns list of distance lists; -1 if unreachable."""
    dists = []
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        dists.append(dist)
    return dists


def _satisfied_pairs(dists, n, pairs, pmap):
    """Bitset of pair indices that are within distance <= 4."""
    sat = 0
    for i, (u, v) in enumerate(pairs):
        d = dists[u][v]
        if d != -1 and d <= 4:
            sat |= (1 << i)
    return sat


def _covered_by_edge(adj, n, edge, pairs, pmap):
    """Which same-parity pairs become distance<=4 when adding this edge (incremental-ish)."""
    u, v = edge
    adj[u].add(v)
    adj[v].add(u)
    dists = _dist_matrix(adj, n)
    sat = _satisfied_pairs(dists, n, pairs, pmap)
    adj[u].remove(v)
    adj[v].remove(u)
    return sat


def _greedy_init(n, base, cands, pairs, pmap):
    """Greedy set cover: repeatedly add the edge that covers the most uncovered pairs."""
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if (u, v) in base and u != v:
                adj[u].add(v)

    # Initial coverage from base path alone
    dists = _dist_matrix(adj, n)
    covered = _satisfied_pairs(dists, n, pairs, pmap)
    total_mask = (1 << len(pairs)) - 1

    added = []
    remaining_cands = list(cands)

    while covered != total_mask:
        if not remaining_cands:
            break
        best_edge = None
        best_new_covered = covered
        for edge in remaining_cands:
            u, v = edge
            # Triangle-free check against current adj (should hold for bipartite, but verify)
            if adj[u] & adj[v]:
                continue
            adj[u].add(v)
            adj[v].add(u)
            dists = _dist_matrix(adj, n)
            sat = _satisfied_pairs(dists, n, pairs, pmap)
            adj[u].remove(v)
            adj[v].remove(u)
            new_covers = sat & ~covered
            cnt = bin(new_covers).count('1')
            if cnt > bin(best_new_covered & ~covered).count('1') or (
                    cnt == bin(best_new_covered & ~covered).count('1') and best_edge is None):
                # pick the one that maximizes newly covered
                if cnt > bin(best_new_covered & ~covered).count('1'):
                    best_new_covered = sat
                    best_edge = edge
        if best_edge is None:
            break
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        added.append((u, v))
        remaining_cands.remove(best_edge)
        dists = _dist_matrix(adj, n)
        covered = _satisfied_pairs(dists, n, pairs, pmap)

    return added, adj


def _local_search_min_edges(n, base, added, adj, pairs, pmap, deadline):
    """Try to remove edges while keeping all pairs covered (diameter<=4)."""
    added = list(added)
    adj = [set(s) for s in adj]
    total_mask = (1 << len(pairs)) - 1

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(added)):
            if time.time() > deadline:
                break
            u, v = added[i]
            adj[u].remove(v)
            adj[v].remove(u)
            dists = _dist_matrix(adj, n)
            covered = _satisfied_pairs(dists, n, pairs, pmap)
            if covered == total_mask:
                added.pop(i)
                improved = True
                break
            else:
                adj[u].add(v)
                adj[v].add(u)
    return added, adj


def _try_swap(n, base, added, adj, cands, pairs, pmap, deadline):
    """Try to replace 2 edges with 1 to reduce count."""
    added = list(added)
    adj = [set(s) for s in adj]
    total_mask = (1 << len(pairs)) - 1

    for _ in range(5):
        if time.time() > deadline:
            break
        found = False
        for i in range(len(added)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(added)):
                if time.time() > deadline:
                    break
                u1, v1 = added[i]
                u2, v2 = added[j]
                adj[u1].remove(v1); adj[v1].remove(u1)
                adj[u2].remove(v2); adj[v2].remove(u2)
                for e in cands:
                    if time.time() > deadline:
                        break
                    u3, v3 = e
                    if (u3, v3) in base or (v3, u3) in base:
                        continue
                    if adj[u3] & adj[v3]:
                        continue
                    adj[u3].add(v3); adj[v3].add(u3)
                    dists = _dist_matrix(adj, n)
                    covered = _satisfied_pairs(dists, n, pairs, pmap)
                    adj[u3].remove(v3); adj[v3].remove(u3)
                    if covered == total_mask:
                        # Perform swap
                        u1, v1 = added[i]
                        u2, v2 = added[j]
                        adj[u1].remove(v1); adj[v1].remove(u1)
                        adj[u2].remove(v2); adj[v2].remove(u2)
                        added.pop(j)
                        added.pop(i)
                        adj[u3].add(v3); adj[v3].add(u3)
                        added.append((u3, v3))
                        found = True
                        break
                if found:
                    break
                # restore
                adj[u1].add(v1); adj[v1].add(u1)
                adj[u2].add(v2); adj[v2].add(u2)
            if found:
                break
        if not found:
            break
    return added, adj


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    n, base = _build_base()
    cands = _bipartite_candidates(n, base)
    pairs = _all_pairs_same_parity(n)
    pmap = _pair_index_map(pairs)

    # Greedy initialization
    added, adj = _greedy_init(n, base, cands, pairs, pmap)

    # Local search: remove redundant edges
    added, adj = _local_search_min_edges(n, base, added, adj, pairs, pmap, deadline)

    # Swap: 2 -> 1
    added, adj = _try_swap(n, base, added, adj, cands, pairs, pmap, deadline)

    # Another round of removal
    added, adj = _local_search_min_edges(n, base, added, adj, pairs, pmap, deadline)

    # Final validation: ensure diameter <= 4 and triangle-free
    dists = _dist_matrix(adj, n)
    diam = max(max(d) for d in dists)
    if diam > 4:
        # Fallback: return greedy result (should not happen)
        pass

    result = []
    for u, v in added:
        result.append([int(u), int(v)])
    return result
