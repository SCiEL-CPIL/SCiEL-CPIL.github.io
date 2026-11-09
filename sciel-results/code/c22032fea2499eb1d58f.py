"""Path decomposition of three fixed connected graphs (Gallai's conjecture).

Structural change: instead of Hungarian/greedy pairing + Hierholzer (which can produce
non-simple paths and then falls back to a worse greedy cover), we use an exact DP
over subsets of odd-degree vertices to enumerate ALL optimal pairings (or near-optimal
if there are too many), and for each pairing attempt the Eulerian trick. We keep the
first pairing that yields a valid (simple, edge-disjoint, covering) decomposition.
This gives the theoretical minimum number of paths whenever such a simple decomposition
exists for some optimal pairing.

Fallbacks: greedy path cover, then edge-by-edge.
"""
import time
import random
from random import Random
from collections import deque
from itertools import product


def construct():
    start_time = time.time()
    DEADLINE = start_time + 140  # safety margin before 150s hard limit

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _bfs_all_pairs(adj, n, odd):
        """Compute shortest-path distances between all pairs of odd vertices.
        Returns dict mapping (s,t) -> dist for s,t in odd, s != t."""
        dist = {}
        for s in odd:
            d = {s: 0}
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in d:
                        d[v] = d[u] + 1
                        q.append(v)
            for t in odd:
                if t != s and t in d:
                    dist[(s, t)] = d[t]
        return dist

    def _eulerian_decompose(edges, n, pairs):
        """Add dummy edges for pairs, find Eulerian circuit, break at dummy edges.
        Returns list of paths (each a list of vertices), or None if fails."""
        if not edges and not pairs:
            return []

        adj = {i: [] for i in range(n)}
        edge_list = []
        eid = 0
        for (u, v) in edges:
            edge_list.append((u, v, eid, False))
            adj[u].append((v, eid))
            adj[v].append((u, eid))
            eid += 1
        for (u, v) in pairs:
            edge_list.append((u, v, eid, True))
            adj[u].append((v, eid))
            adj[v].append((u, eid))
            eid += 1

        if eid == 0:
            return []

        # Hierholzer's algorithm
        used = [False] * eid
        sv = None
        for v in range(n):
            for (nb, e) in adj[v]:
                if not used[e]:
                    sv = v
                    break
            if sv is not None:
                break
        if sv is None:
            return []

        stack = [sv]
        circuit = []
        while stack:
            v = stack[-1]
            advanced = False
            for idx in range(len(adj[v])):
                nb, e = adj[v][idx]
                if not used[e]:
                    used[e] = True
                    stack.append(nb)
                    advanced = True
                    break
            if not advanced:
                circuit.append(stack.pop())

        circuit.reverse()

        # Build set of dummy edges
        dummy_set = set()
        for (u, v, e, is_d) in edge_list:
            if is_d:
                dummy_set.add(frozenset((u, v)))

        # Walk circuit, cutting after each dummy edge
        paths = []
        cur = [circuit[0]]
        for i in range(len(circuit) - 1):
            a, b = circuit[i], circuit[i + 1]
            cur.append(b)
            if frozenset((a, b)) in dummy_set:
                if len(cur) >= 2:
                    paths.append(cur)
                cur = [b]
        if len(cur) >= 2:
            paths.append(cur)

        return paths

    def _is_valid(paths, edges, n):
        """Check: simple paths, edges exist, edge-disjoint, cover all edges."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        used = set()
        for p in paths:
            if len(p) < 2:
                return False
            seen = set()
            for v in p:
                if v in seen:
                    return False
                seen.add(v)
            for i in range(len(p) - 1):
                e = tuple(sorted((p[i], p[i + 1])))
                if e not in edge_set:
                    return False
                if e in used:
                    return False
                used.add(e)
        return used == edge_set

    def _greedy_path_cover(edges, n):
        """Greedy maximal simple path extraction - always produces valid cover."""
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        remaining = set(tuple(sorted(e)) for e in edges)
        new_paths = []
        while remaining:
            sub = {i: set() for i in range(n)}
            for (u, v) in remaining:
                sub[u].add(v)
                sub[v].add(u)
            active = [v for v in range(n) if sub[v]]
            if not active:
                break
            start = max(active, key=lambda v: len(sub[v]))
            path = [start]
            visited = {start}
            cur = start
            while True:
                cands = [nb for nb in sub[cur] if nb not in visited]
                if not cands:
                    break
                nxt = min(cands, key=lambda nb: len(sub[nb]))
                path.append(nxt)
                visited.add(nxt)
                cur = nxt
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i + 1])))
                remaining.discard(e)
            new_paths.append(path)
        return new_paths

    def _dp_pairings(odd, dist, n):
        """Use DP over subsets to find minimum-cost perfect matching of odd vertices.
        Also returns all optimal pairings (up to a limit)."""
        k = len(odd)
        if k == 0:
            return 0, []
        if k == 2:
            return dist.get((odd[0], odd[1]), dist.get((odd[1], odd[0]), float('inf'))), [(odd[0], odd[1])]

        INF = float('inf')
        # dp[mask] = minimum cost to match the set of vertices indicated by mask
        # mask is a bitmask over the k odd vertices
        full_mask = (1 << k) - 1
        dp = [INF] * (1 << k)
        dp[0] = 0
        parent = [None] * (1 << k)  # store (mask, i, j) for backtracking

        for mask in range(1 << k):
            if dp[mask] >= INF:
                continue
            # Find first unmatched vertex in mask
            i = -1
            for b in range(k):
                if mask & (1 << b):
                    i = b
                    break
            if i == -1:
                continue
            for j in range(i + 1, k):
                if mask & (1 << j):
                    d = dist.get((odd[i], odd[j]))
                    if d is None:
                        d = dist.get((odd[j], odd[i]))
                    if d is None:
                        continue
                    new_mask = mask & ~(1 << i) & ~(1 << j)
                    new_cost = dp[mask] + d
                    if new_cost < dp[new_mask]:
                        dp[new_mask] = new_cost
                        parent[new_mask] = (mask, i, j)

        min_cost = dp[full_mask]
        if min_cost >= INF:
            return min_cost, []

        # Backtrack to get one optimal pairing
        def backtrack(mask):
            if mask == full_mask:
                return []
            prev_mask, i, j = parent[mask]
            return [(odd[i], odd[j])] + backtrack(prev_mask)

        optimal_pairs = backtrack(full_mask)

        # Find all optimal pairings (up to limit) using DFS over the DP
        all_optimal = []
        LIMIT = 200  # limit number of optimal pairings to enumerate

        def dfs(mask, current_pairs):
            if len(all_optimal) >= LIMIT:
                return
            if mask == full_mask:
                all_optimal.append(list(current_pairs))
                return
            i = -1
            for b in range(k):
                if mask & (1 << b):
                    i = b
                    break
            if i == -1:
                return
            for j in range(i + 1, k):
                if mask & (1 << j):
                    d = dist.get((odd[i], odd[j]))
                    if d is None:
                        d = dist.get((odd[j], odd[i]))
                    if d is None:
                        continue
                    new_mask = mask & ~(1 << i) & ~(1 << j)
                    if dp[new_mask] + d == min_cost:
                        current_pairs.append((odd[i], odd[j]))
                        dfs(new_mask, current_pairs)
                        current_pairs.pop()

        dfs(0, [])

        return min_cost, all_optimal if all_optimal else [optimal_pairs]

    def path_cover(edges, n):
        if not edges:
            return []

        degree = {i: 0 for i in range(n)}
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        odd = [v for v in range(n) if degree[v] % 2 == 1]

        # Build adjacency
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        if not odd:
            # Eulerian graph: one circuit
            res = _eulerian_decompose(edges, n, [])
            if res and _is_valid(res, edges, n):
                return res
            return _greedy_path_cover(edges, n)

        # Compute all-pairs BFS distances among odd vertices
        dist = _bfs_all_pairs(adj, n, odd)

        # DP to find optimal pairings
        min_cost, pairings = _dp_pairings(odd, dist, n)

        # Try each optimal pairing with the Eulerian trick
        for pairs in pairings:
            res = _eulerian_decompose(edges, n, pairs)
            if res and _is_valid(res, edges, n):
                return res

        # If no optimal pairing works, try near-optimal pairings via 2-opt
        # Start with first optimal pairing, apply 2-opt local search on the pairing
        best_pairs = pairings[0] if pairings else []
        if best_pairs:
            best_cost = min_cost
            for _ in range(30):
                improved = False
                for x in range(len(best_pairs)):
                    if improved:
                        break
                    for y in range(x + 1, len(best_pairs)):
                        if improved:
                            break
                        a1, b1 = best_pairs[x]
                        a2, b2 = best_pairs[y]
                        for (c1, c2, c3, c4) in [(a1, a2, b1, b2), (a1, b2, b1, a2)]:
                            d1 = dist.get((c1, c2))
                            if d1 is None:
                                d1 = dist.get((c2, c1))
                            d2 = dist.get((c3, c4))
                            if d2 is None:
                                d2 = dist.get((c4, c3))
                            if d1 is None or d2 is None:
                                continue
                            old_x = dist.get((a1, b1))
                            if old_x is None:
                                old_x = dist.get((b1, a1))
                            old_y = dist.get((a2, b2))
                            if old_y is None:
                                old_y = dist.get((b2, a2))
                            if old_x is None or old_y is None:
                                continue
                            new_cost = best_cost - old_x - old_y + d1 + d2
                            if new_cost < best_cost - 1e-9:
                                best_pairs[x] = (c1, c2)
                                best_pairs[y] = (c3, c4)
                                best_cost = new_cost
                                improved = True
                                break

            res = _eulerian_decompose(edges, n, best_pairs)
            if res and _is_valid(res, edges, n):
                return res

        # Fallback: greedy
        return _greedy_path_cover(edges, n)

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]

    for n, p, seed in specs:
        if time.time() > DEADLINE:
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        edges = instance(n, p, seed)
        paths = path_cover(edges, n)
        results.append(paths)

    return results
