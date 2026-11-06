import time
import sys
from random import Random

def construct():
    deadline = time.time() + 145  # safety margin under 150s kill

    def generate_graph(n, d, seed):
        rng = Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if e in edges:
                continue
            if degree[u] < d and degree[v] < d:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict(n, edges):
        m = len(edges)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        H_adj = [set() for _ in range(m)]
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    H_adj[i].add(j)
                    H_adj[j].add(i)
                    continue
                if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
                    H_adj[i].add(j)
                    H_adj[j].add(i)
        return H_adj

    def dsatur_greedy(H_adj, m, order):
        colors = [-1] * m
        color_deg = [0] * m  # number of distinct colors among colored neighbors
        # We'll track for each vertex the set of colors used by its colored neighbors
        # For efficiency with small m, just recompute
        for idx in order:
            used = set()
            for nb in H_adj[idx]:
                if colors[nb] != -1:
                    used.add(colors[nb])
            c = 0
            while c in used:
                c += 1
            colors[idx] = c
        return colors

    def local_compression(H_adj, m, colors, rng, time_limit):
        """Try to reduce number of colors by recoloring vertices to lower colors."""
        colors = colors[:]
        ncolors = max(colors) + 1
        start_t = time.time()
        improved = True
        while improved and (time.time() - start_t) < time_limit:
            improved = False
            # Try to eliminate the highest color
            top = ncolors - 1
            vertices_with_top = [i for i in range(m) if colors[i] == top]
            if not vertices_with_top:
                ncolors = top
                continue
            rng.shuffle(vertices_with_top)
            # Try to recolor each vertex in vertices_with_top to a lower color
            for v in vertices_with_top:
                # Find available colors < ncolors-1
                used = set()
                for nb in H_adj[v]:
                    if colors[nb] != -1:
                        used.add(colors[nb])
                for c in range(ncolors - 1):
                    if c not in used:
                        colors[v] = c
                        improved = True
                        break
            if improved:
                # Recompute ncolors
                ncolors = max(colors) + 1
        return colors, ncolors

    def solve_instance(n, d, seed):
        edges = generate_graph(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        H_adj = build_conflict(n, edges)
        degs = [len(nei) for nei in H_adj]

        # Multiple greedy orderings
        best_colors = None
        best_nc = m + 1

        # Order 1: descending degree
        order1 = sorted(range(m), key=lambda x: (-degs[x], x))
        c1 = dsatur_greedy(H_adj, m, order1)
        nc1 = max(c1) + 1
        if nc1 < best_nc:
            best_nc = nc1
            best_colors = c1[:]

        # Order 2: ascending degree
        order2 = sorted(range(m), key=lambda x: (degs[x], x))
        c2 = dsatur_greedy(H_adj, m, order2)
        nc2 = max(c2) + 1
        if nc2 < best_nc:
            best_nc = nc2
            best_colors = c2[:]

        # Order 3: random
        if m > 0:
            rng0 = Random(seed + 777)
            order3 = list(range(m))
            rng0.shuffle(order3)
            c3 = dsatur_greedy(H_adj, m, order3)
            nc3 = max(c3) + 1
            if nc3 < best_nc:
                best_nc = nc3
                best_colors = c3[:]

        # Local compression on best
        if time.time() < deadline:
            rng = Random(seed + 12345)
            remaining = max(0.0, deadline - time.time() - 5.0)
            # Split time among instances: give this instance a fraction
            time_limit = remaining * 0.33
            best_colors, _ = local_compression(H_adj, m, best_colors, rng, time_limit)

        return best_colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        if time.time() >= deadline:
            # Fallback: return a trivial valid coloring (each edge its own color)
            edges = generate_graph(n, d, seed)
            out.append(list(range(len(edges))))
        else:
            out.append(solve_instance(n, d, seed))

    return out

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end - start}")
    for i, r in enumerate(res):
        print(f"Instance {i}: {max(r) + 1 if r else 0} colors")
