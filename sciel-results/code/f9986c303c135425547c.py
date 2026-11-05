import time
import random
import sys
from collections import defaultdict, deque

sys.setrecursionlimit(100000)


def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adj(edge_set, n):
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _max_matching_bipartite(n, edge_set):
    """
    Maximum cardinality matching in bipartite graph (V_out, V_in)
    where edge (u_out, v_in) exists iff (min(u,v),max(u,v)) in edge_set and u != v.
    Uses Hopcroft-Karp. Returns list of matched pairs (u, v) meaning u_out matched v_in.
    """
    # Build adjacency: for each u, list of v such that (u,v) is an edge (u!=v)
    adj = [[] for _ in range(n)]
    for u, v in edge_set:
        adj[u].append(v)
        adj[v].append(u)

    pair_u = [-1] * n
    pair_v = [-1] * n
    dist = [0] * n

    def bfs():
        q = deque()
        for u in range(n):
            if pair_u[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1
        found = False
        while q:
            u = q.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == -1:
                    dist[pu] = dist[u] + 1
                    q.append(pu)
        return found

    def dfs(u):
        for v in adj[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = -1
        return False

    matching = 0
    while bfs():
        for u in range(n):
            if pair_u[u] == -1:
                if dfs(u):
                    matching += 1

    result = []
    for u in range(n):
        if pair_u[u] != -1:
            result.append((u, pair_u[u]))
    return result


def _extract_cycles_from_matching(matching, n):
    """
    Given matching as list of (u, v) meaning u_out -> v_in,
    decompose into cycles and paths. Return list of cycles (each a list of vertices).
    """
    # Build functional graph: u -> v (u_out matched to v_in)
    # and v -> u (v_in matched to u_out)
    out_map = {}
    in_map = {}
    for u, v in matching:
        out_map[u] = v
        in_map[v] = u

    visited = set()
    cycles = []
    for start in range(n):
        if start in visited:
            continue
        # Follow the alternating path: start as an "out" node
        path = [start]
        cur = start
        is_out = True
        while True:
            if is_out:
                if cur in out_map:
                    nxt = out_map[cur]
                    visited.add(cur)
                    path.append(nxt)
                    cur = nxt
                    is_out = False
                else:
                    break
            else:
                if cur in in_map:
                    nxt = in_map[cur]
                    visited.add(cur)
                    path.append(nxt)
                    cur = nxt
                    is_out = True
                else:
                    break
        # path is a sequence of nodes; check if it's a cycle
        # A cycle: path[0] == path[-1] and all internal distinct, length >= 3
        if len(path) >= 4 and path[0] == path[-1]:
            cyc = path[:-1]
            # Verify all distinct
            if len(set(cyc)) == len(cyc):
                cycles.append(cyc)
    return cycles


def _solve_one(edges, n, deadline):
    if not edges:
        return []
    m = len(edges)
    edge_set = set(tuple(sorted(e)) for e in edges)
    adj = _build_adj(edge_set, n)

    if m < 3:
        return [list(e) for e in edges]

    all_cycles = []
    used_edges_global = set()
    seen_cycles = set()

    def _canonical(cycle):
        if len(cycle) < 3:
            return None
        min_val = min(cycle)
        idxs = [i for i, v in enumerate(cycle) if v == min_val]
        candidates = []
        for idx in idxs:
            c1 = cycle[idx:] + cycle[:idx]
            c2 = list(reversed(c1))
            candidates.append(tuple(c1) if c1 < c2 else tuple(c2))
        return min(candidates)

    def _try_add_cycle(cycle):
        if len(cycle) < 3:
            return False
        es = set()
        L = len(cycle)
        for i in range(L):
            a = cycle[i]
            b = cycle[(i + 1) % L]
            e = tuple(sorted((a, b)))
            if e not in edge_set:
                return False
            es.add(e)
        if es & used_edges_global:
            return False
        key = _canonical(cycle)
        if key in seen_cycles:
            return False
        seen_cycles.add(key)
        used_edges_global.update(es)
        all_cycles.append(list(cycle))
        return True

    # Iteratively find max matching on remaining edges, extract cycles
    remaining = set(edge_set)
    for it in range(60):
        if time.time() > deadline - 0.5:
            break
        if not remaining:
            break
        # Build edge set for matching
        sub_edges = [tuple(sorted(e)) for e in remaining]
        matching = _max_matching_bipartite(n, set(sub_edges))
        if not matching:
            break
        cycles = _extract_cycles_from_matching(matching, n)
        added = 0
        for c in cycles:
            if _try_add_cycle(c):
                added += 1
        if added == 0:
            # No cycles found; break to avoid infinite loop
            break
        # Update remaining
        remaining = edge_set - used_edges_global

    # If we still have time, try to find more cycles via random walks / BFS on remaining
    if remaining and time.time() < deadline - 1.0:
        # Find cycles in remaining via BFS
        rem_adj = [set() for _ in range(n)]
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        for u, v in list(remaining):
            if time.time() > deadline - 0.5:
                break
            # BFS from u to v avoiding edge (u,v)
            parent = {u: None}
            q = deque([u])
            found = False
            while q and not found:
                cur = q.popleft()
                for nb in rem_adj[cur]:
                    if nb == v and cur == u:
                        continue
                    if nb not in parent:
                        parent[nb] = cur
                        if nb == v:
                            found = True
                            break
                        q.append(nb)
            if found:
                path = [v]
                cur = v
                while cur != u:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                _try_add_cycle(path)
                remaining = edge_set - used_edges_global
                if not remaining:
                    break

    pieces = [list(c) for c in all_cycles]
    for e in (edge_set - used_edges_global):
        pieces.append(list(e))
    return pieces


def construct():
    start_time = time.time()
    total_budget = 145.0
    per_graph_budget = total_budget / 3.0

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []

    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        graph_deadline = start_time + per_graph_budget * (i + 1)
        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0

        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)

        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break

    return results


if __name__ == "__main__":
    res = construct()
    for i, pieces in enumerate(res):
        print(i, len(pieces))
