import time
from random import Random
from collections import deque

def construct():
    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]

    def fallback():
        out = []
        for spec in specs:
            edges = instance(*spec)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colour.items() if set(f) & near}
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([colour[e] for e in edges])
        return out

    def compress(color):
        m = len(color)
        if m == 0:
            return []
        mapping = {}
        new = [0] * m
        nxt = 0
        for i, c in enumerate(color):
            if c not in mapping:
                mapping[c] = nxt
                nxt += 1
            new[i] = mapping[c]
        return new

    def solve_instance(n, d, seed, deadline):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        # Build conflict graph (line graph of the square of the graph)
        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)

        # For each edge i, compute its "forbidden" set: edges that conflict with it
        # Two edges conflict if they share a vertex or are joined by an edge
        edge_vertices = [set() for _ in range(m)]
        for i, (u, v) in enumerate(edges):
            edge_vertices[i] = {u, v}
            # Add neighbors of u and v (excluding each other)
            for nb in adj_orig[u]:
                if nb != v:
                    edge_vertices[i].add(nb)
            for nb in adj_orig[v]:
                if nb != u:
                    edge_vertices[i].add(nb)

        # Conflict graph: edge i conflicts with edge j if they share a vertex or are joined
        conflict = [set() for _ in range(m)]
        for i in range(m):
            ui, vi = edges[i]
            for j in range(i + 1, m):
                uj, vj = edges[j]
                # Share a vertex?
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    conflict[i].add(j)
                    conflict[j].add(i)
                # Joined by an edge? (i.e., an endpoint of i is adjacent to an endpoint of j)
                elif uj in adj_orig[ui] or uj in adj_orig[vi] or vj in adj_orig[ui] or vj in adj_orig[vi]:
                    conflict[i].add(j)
                    conflict[j].add(i)

        # Degrees in conflict graph
        deg = [len(conflict[i]) for i in range(m)]

        # Greedy coloring with most-constrained-first ordering
        def greedy_color(order):
            color = [-1] * m
            for v in order:
                forbidden = set()
                for u in conflict[v]:
                    if color[u] != -1:
                        forbidden.add(color[u])
                c = 0
                while c in forbidden:
                    c += 1
                color[v] = c
            return color

        # Initial greedy: order by degree descending
        order = sorted(range(m), key=lambda x: deg[x], reverse=True)
        best_color = greedy_color(order)
        best_K = max(best_color) + 1

        # Lower bound: clique number of conflict graph
        # Simple lower bound: max degree + 1
        lb = max(deg) + 1

        # Try to reduce colors using iterative local search
        rng = Random(seed * 7919 + 13)
        current_color = best_color[:]
        current_K = best_K

        # Simulated annealing / random walk to reduce color count
        def try_reduce(color, K, steps, rng, deadline):
            """Try to reduce from K colors to K-1 via random valid recolorings."""
            if K <= 1:
                return color, K
            count_max = sum(1 for c in color if c == K - 1)
            if count_max == 0:
                return color, K - 1

            for _ in range(steps):
                if time.time() > deadline:
                    break
                v = rng.randrange(m)
                c = rng.randrange(K - 1)
                # Check validity
                valid = True
                for u in conflict[v]:
                    if color[u] == c:
                        valid = False
                        break
                if valid:
                    old = color[v]
                    color[v] = c
                    if old == K - 1:
                        count_max -= 1
                        if count_max == 0:
                            return color, K - 1
            return color, K

        # Main search loop
        while time.time() < deadline - 0.5:
            # Try random walk reduction
            trial_color = current_color[:]
            trial_K = current_K
            steps = 5000
            trial_color, trial_K = try_reduce(trial_color, trial_K, steps, rng, deadline - 0.5)

            if trial_K < current_K:
                current_color = trial_color
                current_K = trial_K
                if current_K < best_K:
                    best_color = current_color[:]
                    best_K = current_K
                if best_K <= lb:
                    break
            else:
                # Failed: try a new random greedy restart
                perm = list(range(m))
                rng.shuffle(perm)
                new_col = greedy_color(perm)
                new_K = max(new_col) + 1
                if new_K < current_K:
                    current_color = new_col
                    current_K = new_K
                    if new_K < best_K:
                        best_color = new_col[:]
                        best_K = new_K
                        if best_K <= lb:
                            break

        return compress(best_color)

    try:
        start = time.time()
        # Budget: 145s total, leave 5s margin
        deadline = start + 145.0
        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining = deadline - time.time()
            # Give more time to harder instances (higher d)
            shares = [1.0, 1.3, 1.6]
            total_share = sum(shares[idx:])
            share = remaining * shares[idx] / total_share
            share = min(share, 50.0)
            share = max(share, 1.0)
            inst_deadline = min(deadline - 0.5, time.time() + share)
            out.append(solve_instance(n, d, seed, inst_deadline))
        return out
    except Exception:
        return fallback()
