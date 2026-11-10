"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from collections import defaultdict, deque
from random import Random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges, n):
        """
        Decompose a graph into cycles and single edges.
        Strategy: repeatedly find a long cycle in the remaining graph,
        remove its edges, until no cycle remains. Leftover edges become single-edge pieces.
        """
        if not edges:
            return []

        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        remaining_edges = set(edges)
        pieces = []

        # Iteratively find cycles
        max_iters = 10000
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            if not remaining_edges:
                break

            # Find a cycle using DFS
            cycle = _find_cycle(adj, remaining_edges)
            if not cycle:
                break

            # Remove cycle edges from remaining
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                remaining_edges.discard(e)
                adj[u].discard(v)
                adj[v].discard(u)

            pieces.append(cycle)

        # Remaining edges become single-edge pieces
        for u, v in remaining_edges:
            pieces.append([u, v])

        return pieces

    def _find_cycle(adj, remaining_edges):
        """
        Find a cycle in the graph defined by adj (which should match remaining_edges).
        Uses DFS to find any cycle, then tries to extend it.
        Returns a list of vertices forming the cycle, or None.
        """
        visited = set()
        parent = {}

        def dfs(start):
            stack = [(start, -1, [start])]
            visited.add(start)
            while stack:
                node, par, path = stack.pop()
                for nbr in sorted(adj[node], key=lambda x: len(adj[x])):  # prefer low-degree
                    if nbr not in visited:
                        visited.add(nbr)
                        parent[nbr] = node
                        stack.append((nbr, node, path + [nbr]))
                    elif nbr != par and nbr in visited:
                        # Found a cycle: reconstruct from path
                        # Find nbr in path
                        if nbr in path:
                            idx = path.index(nbr)
                            cycle = path[idx:]
                            return cycle
            return None

        # Try DFS from each unvisited vertex
        for v in range(len(adj)):
            if v not in visited:
                cycle = dfs(v)
                if cycle and len(cycle) >= 3:
                    return cycle

        return None

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n)
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
