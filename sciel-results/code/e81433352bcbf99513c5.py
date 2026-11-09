import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_greedy(edges, n):
        """
        Greedy path cover: repeatedly find a long simple path in the remaining graph
        using a heuristic that prefers neighbors with fewer unused edges (Warnsdorff-like).
        """
        if not edges:
            return []

        edge_set = set(tuple(sorted(e)) for e in edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Work with a mutable copy of adjacency
        remaining_adj = {u: set(adj[u]) for u in adj}
        used_edges = set()
        paths = []

        # Sort vertices by degree (descending) for tie-breaking
        deg_order = sorted(range(n), key=lambda x: len(remaining_adj.get(x, set())), reverse=True)

        while len(used_edges) < len(edge_set):
            # Find an unused edge to start from
            start_edge = None
            for e in edge_set:
                if e not in used_edges:
                    start_edge = e
                    break
            if start_edge is None:
                break

            u, v = start_edge
            # Try both orientations
            best_path = None

            for start, end in [(u, v), (v, u)]:
                path = [start, end]
                used_edges.add(start_edge)
                last = end
                while True:
                    # Find next vertex: prefer neighbor with fewest unused edges
                    candidates = []
                    for w in remaining_adj.get(last, set()):
                        e = tuple(sorted((last, w)))
                        if e in used_edges:
                            continue
                        if w in path:
                            continue
                        # Count unused edges from w
                        unused_count = 0
                        for x in remaining_adj.get(w, set()):
                            ee = tuple(sorted((w, x)))
                            if ee not in used_edges:
                                unused_count += 1
                        candidates.append((unused_count, w))
                    if not candidates:
                        break
                    candidates.sort()
                    next_w = candidates[0][1]
                    path.append(next_w)
                    used_edges.add(tuple(sorted((last, next_w))))
                    last = next_w

                if best_path is None or len(path) > len(best_path):
                    best_path = path

            # Commit the best path
            if best_path:
                paths.append(best_path)
            else:
                # Should not happen
                break

        return paths

    def solve_lp(edges, n):
        """
        Solve edge-disjoint path cover via min-cost flow / LP relaxation.
        Each undirected edge becomes two directed arcs with capacity 1.
        Minimize sum of path endpoints.
        """
        from scipy.optimize import linprog
        import numpy as np

        if not edges:
            return []

        edge_set = set(tuple(sorted(e)) for e in edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        arcs = []
        arc_index = {}
        for u, v in edge_set:
            arcs.append((u, v))
            arc_index[(u, v)] = len(arcs) - 1
            arcs.append((v, u))
            arc_index[(v, u)] = len(arcs) - 1

        m = len(arcs)
        c = [0.0] * m + [1.0] * n + [1.0] * n

        A_eq = []
        b_eq = []
        for u in range(n):
            row = [0.0] * m + [0.0] * n + [0.0] * n
            for (a, b) in arcs:
                if a == u:
                    row[arc_index[(a, b)]] = 1.0
                if b == u:
                    row[arc_index[(a, b)]] = -1.0
            row[u] = -1.0
            row[n + u] = 1.0
            A_eq.append(row)
            b_eq.append(0.0)

        bounds = [(0, 1)] * m + [(0, None)] * n + [(0, None)] * n

        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if not res.success:
            return None

        used_arcs = set()
        for i in range(m):
            if res.x[i] > 0.5:
                used_arcs.add(arcs[i])

        adj_used = defaultdict(list)
        for u, v in used_arcs:
            adj_used[u].append(v)

        in_deg = defaultdict(int)
        out_deg = defaultdict(int)
        for u, v in used_arcs:
            out_deg[u] += 1
            in_deg[v] += 1

        starts = [u for u in range(n) if out_deg[u] > in_deg[u]]
        if not starts:
            return None

        paths = []
        for start in starts:
            if not adj_used[start]:
                continue
            path = [start]
            cur = start
            while adj_used[cur]:
                next_node = adj_used[cur].pop(0)
                path.append(next_node)
                cur = next_node
            paths.append(path)

        return paths

    def verify(paths, edges):
        edge_set = set(tuple(sorted(e)) for e in edges)
        used = set()
        for path in paths:
            if len(path) < 2:
                return False
            if len(set(path)) != len(path):
                return False
            for j in range(len(path) - 1):
                e = tuple(sorted((path[j], path[j + 1])))
                if e not in edge_set:
                    return False
                if e in used:
                    return False
                used.add(e)
        return used == edge_set

    results = []
    for spec in [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue

        edges = instance(*spec)
        n = spec[0]

        # Try LP first (may give optimal or near-optimal)
        lp_paths = solve_lp(edges, n)
        if lp_paths is not None and verify(lp_paths, edges):
            results.append(lp_paths)
        else:
            # Fall back to greedy
            greedy_paths = solve_greedy(edges, n)
            if verify(greedy_paths, edges):
                results.append(greedy_paths)
            else:
                # Last resort: single-edge paths
                results.append([[u, v] for u, v in edges])

    return results
