"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    from collections import defaultdict, deque

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_bfs_path(start, end, remaining_edges, adj_dict):
        """Find a simple path from start to end using BFS on remaining edges."""
        if start == end:
            return [start]
        
        visited = {start}
        parent = {start: None}
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            if node == end:
                path = []
                cur = end
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            
            for neighbor in sorted(adj_dict.get(node, [])):
                if neighbor in visited:
                    continue
                e = tuple(sorted((node, neighbor)))
                if e in remaining_edges:
                    visited.add(neighbor)
                    parent[neighbor] = node
                    queue.append(neighbor)
        
        return None

    def get_odd_vertices(edge_set, n):
        d = [0] * n
        for u, v in edge_set:
            d[u] += 1
            d[v] += 1
        return [i for i in range(n) if d[i] % 2 == 1]

    def solve(n, edges):
        if not edges:
            return []
        
        remaining = set(edges)
        adj_dict = defaultdict(set)
        for u, v in edges:
            adj_dict[u].add(v)
            adj_dict[v].add(u)
        
        paths = []
        
        while remaining:
            odd = get_odd_vertices(remaining, n)
            
            if len(odd) == 0:
                # Eulerian: pick any edge, find a path that covers as many edges as possible
                # For simplicity, just take one edge at a time (suboptimal but valid)
                # Better: find an Eulerian path
                e = next(iter(remaining))
                u, v = e
                # Try to find a long path
                path = find_bfs_path(u, v, remaining, adj_dict)
                if path and len(path) > 2:
                    paths.append(path)
                    for i in range(len(path) - 1):
                        ee = tuple(sorted((path[i], path[i+1])))
                        remaining.discard(ee)
                else:
                    paths.append([u, v])
                    remaining.discard(e)
            elif len(odd) == 2:
                u, v = odd
                path = find_bfs_path(u, v, remaining, adj_dict)
                if path:
                    paths.append(path)
                    for i in range(len(path) - 1):
                        ee = tuple(sorted((path[i], path[i+1])))
                        remaining.discard(ee)
                else:
                    e = next(iter(remaining))
                    paths.append([e[0], e[1]])
                    remaining.discard(e)
            else:
                # Pair up odd vertices
                paired = False
                for i in range(0, len(odd), 2):
                    u, v = odd[i], odd[i+1]
                    path = find_bfs_path(u, v, remaining, adj_dict)
                    if path and len(path) >= 2:
                        paths.append(path)
                        for j in range(len(path) - 1):
                            ee = tuple(sorted((path[j], path[j+1])))
                            remaining.discard(ee)
                        paired = True
                        break
                if not paired:
                    e = next(iter(remaining))
                    paths.append([e[0], e[1]])
                    remaining.discard(e)
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = solve(n, edges)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END

