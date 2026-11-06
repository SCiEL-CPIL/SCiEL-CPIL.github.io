"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = 145.0

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_cycle(adj, remaining_edges):
        """Find a cycle in the graph using DFS. Returns list of vertices or None."""
        # Build adjacency from remaining edges
        adj_map = {}
        for u, v in remaining_edges:
            if u not in adj_map:
                adj_map[u] = []
            if v not in adj_map:
                adj_map[v] = []
            adj_map[u].append(v)
            adj_map[v].append(u)

        for start in adj_map:
            # DFS with parent tracking
            stack = [(start, -1, [start])]
            visited = set()
            while stack:
                if time.time() > deadline:
                    return None
                node, parent, path = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                for neighbor in adj_map.get(node, []):
                    if neighbor == parent:
                        continue
                    e = (min(node, neighbor), max(node, neighbor))
                    if e not in remaining_edges:
                        continue
                    if neighbor == start and len(path) >= 3:
                        return path
                    if neighbor not in visited:
                        stack.append((neighbor, node, path + [neighbor]))
        return None

    def decompose(edges, n):
        """Decompose graph into cycles and single edges."""
        remaining = set((u, v) if u < v else (v, u) for u, v in edges)
        pieces = []

        while remaining:
            if time.time() > deadline:
                break

            # Try to find a cycle
            cycle = find_cycle(None, remaining)
            if cycle:
                # Remove cycle edges
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    remaining.discard(e)
                pieces.append(cycle)
            else:
                # No cycle, take a single edge
                e = next(iter(remaining))
                remaining.discard(e)
                pieces.append(list(e))

        return pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n)
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
