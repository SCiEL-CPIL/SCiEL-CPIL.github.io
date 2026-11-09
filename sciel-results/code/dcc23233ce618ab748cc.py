"""Decompose three fixed graphs into few cycles and edges (Erdős-Gallai style)."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    DEADLINE = 120.0
    t0 = time.time()

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_cycle_dfs(adj, start, max_visits=200000):
        """DFS with backtracking to find a cycle through start. Returns list of vertices or None."""
        if start not in adj or len(adj[start]) < 2:
            return None
        neighbors = list(adj[start])
        if len(neighbors) < 2:
            return None
        
        # Try each pair of neighbors to find a path between them (excluding start)
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                nb1, nb2 = neighbors[i], neighbors[j]
                # DFS from nb1 to nb2, avoiding start
                stack = [(nb1, [nb1])]
                visited = {start, nb1}
                visits = 0
                while stack:
                    if time.time() - t0 > DEADLINE - 3:
                        return None
                    node, path = stack.pop()
                    visits += 1
                    if visits > max_visits:
                        break
                    if node == nb2:
                        return [start] + path
                    for nb in adj[node]:
                        if nb != start and nb not in visited:
                            visited.add(nb)
                            stack.append((nb, path + [nb]))
                if visits > max_visits:
                    break
        return None

    def find_best_cycle(adj):
        """Find the longest cycle by trying multiple starts with DFS."""
        if not adj:
            return None
        vertices = [v for v in adj if len(adj[v]) >= 2]
        if len(vertices) < 3:
            return None
        
        # Sort by degree ascending (low-degree vertices more likely in small cycles, 
        # but also try high-degree)
        vertices_sorted = sorted(vertices, key=lambda v: len(adj[v]))
        
        best_cycle = None
        # Try a subset of starts: lowest degree, highest degree, and some random
        starts_to_try = set()
        for v in vertices_sorted[:5]:
            starts_to_try.add(v)
        for v in vertices_sorted[-5:]:
            starts_to_try.add(v)
        # Add a few random
        for _ in range(10):
            starts_to_try.add(random.choice(vertices))
        
        for start in starts_to_try:
            if time.time() - t0 > DEADLINE - 3:
                break
            cycle = find_cycle_dfs(adj, start)
            if cycle and len(cycle) >= 3:
                if best_cycle is None or len(cycle) > len(best_cycle):
                    best_cycle = cycle
                    if len(cycle) >= len(vertices):  # Hamiltonian
                        break
        return best_cycle

    def decompose(edges):
        if not edges:
            return []
        remaining = set(edges)
        adj = build_adj(edges)
        pieces = []
        
        while remaining:
            if time.time() - t0 > DEADLINE - 1:
                break
            
            cycle = find_best_cycle(adj)
            if cycle and len(cycle) >= 3:
                # Verify
                valid = True
                cycle_edges = []
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                    e = tuple(sorted((u, v)))
                    if e not in remaining:
                        valid = False
                        break
                    cycle_edges.append(e)
                if valid:
                    for e in cycle_edges:
                        remaining.discard(e)
                        u, v = e
                        adj[u].discard(v)
                        adj[v].discard(u)
                    pieces.append(cycle)
                    continue
            
            # No cycle: pick edge incident to min-degree vertex
            min_deg_vertex = min(adj.keys(), key=lambda v: len(adj[v])) if adj else None
            if min_deg_vertex is not None and adj[min_deg_vertex]:
                e = tuple(sorted((min_deg_vertex, next(iter(adj[min_deg_vertex])))))
            else:
                e = next(iter(remaining))
            
            u, v = e
            remaining.discard(e)
            adj[u].discard(v)
            adj[v].discard(u)
            pieces.append([u, v])
            
            # Clean up isolated vertices
            for v in list(adj.keys()):
                if not adj[v]:
                    del adj[v]
        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() - t0 > DEADLINE - 1:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        result = decompose(edges)
        results.append(result)

    return results
# EVOLVE-BLOCK-END
