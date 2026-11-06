"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    # Deadline guard: kill after 140s to be safe (150s limit)
    t_start = time.time()
    DEADLINE = 140.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))

    def decompose(edges, n):
        """
        Decompose edges into cycles and single edges.
        Returns list of pieces. A piece is either [u,v] (edge) or [v1,v2,...,vk] (cycle, k>=3).
        Strategy:
          1. Build adjacency.
          2. Repeatedly find a cycle in the remaining graph and remove it.
          3. Leftover edges become single-edge pieces.
          We try to find long cycles to minimize piece count.
        """
        # Build adjacency list with edge IDs
        edge_set = set()
        for e in edges:
            edge_set.add(tuple(sorted(e)))

        adj = defaultdict(list)  # vertex -> list of (neighbor, edge_tuple)
        for (u, v) in edge_set:
            adj[u].append((v, (u, v)))
            adj[v].append((u, (u, v)))

        pieces = []
        used_edges = set()

        # We'll iteratively find cycles
        # To find a cycle, do DFS from each vertex in order
        # We want to find a cycle that uses many edges (long cycle) to reduce piece count

        def find_cycle(adj, used_edges):
            """Find a simple cycle in the current graph. Returns list of vertices or None."""
            n = max(adj.keys()) + 1 if adj else 0
            color = [0] * (n + 1)  # 0=white, 1=gray, 2=black
            parent = [-1] * (n + 1)
            cycle = None

            def dfs(u, start):
                nonlocal cycle
                color[u] = 1
                for (v, edge) in adj[u]:
                    if edge in used_edges:
                        continue
                    if color[v] == 0:
                        parent[v] = u
                        if dfs(v, start):
                            return True
                    elif color[v] == 1:
                        # Found a back edge to a gray node -> cycle
                        if v == start:
                            # Cycle from u back to start
                            cyc = [u]
                            cur = u
                            while cur != start:
                                cur = parent[cur]
                                if cur == -1:
                                    return False
                                cyc.append(cur)
                            # cyc has start at end, u at front; reverse to get path start->...->u
                            cyc.reverse()
                            # Check length >= 3
                            if len(cyc) >= 3:
                                cycle = cyc
                                return True
                        else:
                            # Cycle from v to u
                            cyc = [u]
                            cur = u
                            while cur != v:
                                cur = parent[cur]
                                if cur == -1:
                                    return False
                                cyc.append(cur)
                            cyc.append(v)
                            if len(cyc) >= 3:
                                cycle = cyc
                                return True
                color[u] = 2
                return False

            for start in range(n + 1):
                if color[start] == 0 and adj[start]:
                    if dfs(start, start):
                        return cycle
            return None

        # Iteratively find cycles
        # We'll do this: repeatedly find a cycle, remove its edges
        # To encourage longer cycles, we could try multiple random orderings

        # For efficiency and simplicity, we do a greedy approach:
        # Find a cycle, remove it, repeat.

        # But we can improve by trying to find cycles that are "long"
        # Let's do: try to find a cycle using DFS, but we can also try to extend

        # Simple greedy: find any cycle, remove it
        # This may not be optimal but is fast

        # Let's add a heuristic: prefer cycles that are longer
        # We'll do multiple passes: first find all cycles, then pick the longest?
        # That's complex. Let's just do iterative removal.

        # Actually, a better approach:
        # 1. While there's a cycle in the graph, find one and remove it.
        # 2. Remaining edges are single-edge pieces.

        # To find cycles efficiently, we can use the fact that a graph has a cycle
        # iff it's not a forest. We can repeatedly find a cycle via DFS.

        # Let's implement:
        while True:
            if time.time() > DEADLINE:
                break
            # Check if any cycle exists
            # Build current adj without used edges
            cur_adj = defaultdict(list)
            for (u, v) in edge_set:
                if (u, v) in used_edges:
                    continue
                cur_adj[u].append(v)
                cur_adj[v].append(u)

            # Find a cycle
            n_max = max(cur_adj.keys()) + 1 if cur_adj else 0
            if n_max == 0:
                break

            color = [0] * (n_max + 1)
            parent = [-1] * (n_max + 1)
            found_cycle = None

            def dfs2(u, start):
                nonlocal found_cycle
                color[u] = 1
                for v in cur_adj[u]:
                    if color[v] == 0:
                        parent[v] = u
                        if dfs2(v, start):
                            return True
                    elif color[v] == 1:
                        if v == start:
                            cyc = [u]
                            cur = u
                            while cur != start:
                                cur = parent[cur]
                                if cur == -1:
                                    return False
                                cyc.append(cur)
                            cyc.reverse()
                            if len(cyc) >= 3:
                                found_cycle = cyc
                                return True
                        else:
                            cyc = [u]
                            cur = u
                            while cur != v:
                                cur = parent[cur]
                                if cur == -1:
                                    return False
                                cyc.append(cur)
                            cyc.append(v)
                            if len(cyc) >= 3:
                                found_cycle = cyc
                                return True
                color[u] = 2
                return False

            for s in range(n_max + 1):
                if color[s] == 0 and cur_adj[s]:
                    if dfs2(s, s):
                        break

            if found_cycle is None:
                break

            # Remove edges in the cycle
            for i in range(len(found_cycle)):
                u = found_cycle[i]
                v = found_cycle[(i + 1) % len(found_cycle)]
                edge = tuple(sorted((u, v)))
                used_edges.add(edge)

            pieces.append(found_cycle)

        # Remaining edges
        for e in edge_set:
            if e not in used_edges:
                pieces.append(list(e))

        return pieces

    results = []
    for spec in specs:
        if time.time() > DEADLINE:
            # Fallback: just return single edges for remaining
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        n = spec[0]
        result = decompose(edges, n)
        results.append(result)

    return results
# EVOLVE-BLOCK-END
