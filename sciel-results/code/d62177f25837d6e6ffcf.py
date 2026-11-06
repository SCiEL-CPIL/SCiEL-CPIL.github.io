import time
import sys
from collections import defaultdict
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_lp(edges, time_limit=30.0):
        """
        Solve edge-disjoint path decomposition using LP relaxation.
        The LP will give a fractional solution; we'll round it.
        If LP fails or is too slow, fall back to greedy.
        """
        try:
            from scipy.optimize import linprog
        except ImportError:
            return None

        if not edges:
            return []

        n_nodes = max(max(e) for e in edges) + 1
        m = len(edges)

        # Build node adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Enumerate all simple paths. To keep it tractable, we limit path length.
        # For n=30, we can't enumerate all simple paths (exponential).
        # Instead, use a different approach: formulate as minimum path cover in a directed sense?
        
        # Actually, let's use a different strategy: 
        # We know that for connected graphs, the minimum number of paths in an edge decomposition
        # is at least ceil(number_of_odd_degree_nodes / 2).
        # Gallai's conjecture says it's at most floor(n/2).
        
        # Let's try a different approach: use the fact that we can pair up odd-degree vertices
        # and find paths between them.
        
        # Actually, the simplest improvement: use Chinese Postman style approach.
        # Add dummy edges to make all degrees even, find Eulerian circuits, then break at vertices.
        
        # For now, let's try a smarter greedy: 
        # 1. Identify odd-degree vertices
        # 2. Pair them up
        # 3. For each pair, find a shortest path between them (in the remaining graph)
        # 4. Extract that path, remove its edges
        # 5. Repeat until all edges are covered
        
        return None  # Skip LP for now, use improved greedy

    def decompose_smart_greedy(edges, max_time=20.0):
        """
        Improved greedy: 
        1. Find odd-degree vertices
        2. Pair them up (greedily by proximity)
        3. For each pair, find a path between them using BFS/DFS
        4. Extract path, remove edges
        5. If no odd vertices remain, find Eulerian circuits in components
        """
        if not edges:
            return []

        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        remaining_edges = set(tuple(sorted(e)) for e in edges)
        paths = []
        elapsed = 0.0

        while remaining_edges:
            if elapsed > max_time:
                break

            # Build degree map for remaining graph
            deg = defaultdict(int)
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1

            # Find odd-degree vertices
            odd_nodes = [v for v in deg if deg[v] % 2 == 1]

            if len(odd_nodes) <= 2:
                # 0 or 2 odd vertices: find Eulerian path/circuit
                # For 2 odd vertices, start at one, end at the other
                # For 0, any start
                if len(odd_nodes) == 2:
                    start = odd_nodes[0]
                else:
                    start = next(iter(deg)) if deg else None
                
                if start is not None:
                    path = find_eulerian_path(start, remaining_edges, adj)
                    if path and len(path) > 1:
                        paths.append(path)
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i + 1]
                            remaining_edges.discard(tuple(sorted((u, v))))
                        continue

            # More than 2 odd vertices: pair them up
            if len(odd_nodes) >= 2:
                # Find pairs of odd nodes that are close together
                # Greedy: for each odd node, find nearest odd node
                used = set()
                pairs = []
                for u in odd_nodes:
                    if u in used:
                        continue
                    # Find nearest unused odd node
                    best_v = None
                    best_dist = float('inf')
                    for v in odd_nodes:
                        if v in used or v == u:
                            continue
                        # BFS to find shortest path distance
                        dist = bfs_distance(u, v, remaining_edges, adj)
                        if dist < best_dist:
                            best_dist = dist
                            best_v = v
                    if best_v is not None:
                        pairs.append((u, best_v))
                        used.add(u)
                        used.add(best_v)

                # For each pair, find a path between them
                for u, v in pairs:
                    path = find_shortest_path(u, v, remaining_edges, adj)
                    if path and len(path) > 1:
                        paths.append(path)
                        for i in range(len(path) - 1):
                            a, b = path[i], path[i + 1]
                            remaining_edges.discard(tuple(sorted((a, b))))
            else:
                # Shouldn't happen if remaining_edges is non-empty
                e = next(iter(remaining_edges))
                paths.append([e[0], e[1]])
                remaining_edges.discard(e)

            elapsed = time.time() - start_time

        # If we timed out, add remaining edges as single-edge paths
        if remaining_edges:
            for u, v in remaining_edges:
                paths.append([u, v])

        return paths

    def find_eulerian_path(start, edges, adj):
        """Find an Eulerian path starting from 'start' in the graph defined by edges."""
        if not edges:
            return []

        # Hierholzer's algorithm
        adj_local = defaultdict(list)
        for u, v in edges:
            adj_local[u].append(v)
            adj_local[v].append(u)

        # Make sure we have the right edge multiset
        edge_set = set(tuple(sorted(e)) for e in edges)
        used_edges = set()

        stack = [start]
        path = []
        visited = {}  # vertex -> index in adj_local list

        while stack:
            v = stack[-1]
            if v not in visited:
                visited[v] = 0
            
            # Find an unused edge from v
            next_v = None
            while visited[v] < len(adj_local.get(v, [])):
                nb = adj_local[v][visited[v]]
                visited[v] += 1
                e = tuple(sorted((v, nb)))
                if e not in used_edges:
                    next_v = nb
                    break
            
            if next_v is not None:
                e = tuple(sorted((v, next_v)))
                used_edges.add(e)
                stack.append(next_v)
            else:
                path.append(stack.pop())
        
        path.reverse()
        return path

    def bfs_distance(u, v, edges, adj):
        """BFS shortest path distance between u and v using only edges in 'edges'."""
        if u == v:
            return 0
        
        from collections import deque
        dist = {u: 0}
        q = deque([u])
        while q:
            cur = q.popleft()
            for nb in adj.get(cur, []):
                e = tuple(sorted((cur, nb)))
                if e in edges and nb not in dist:
                    dist[nb] = dist[cur] + 1
                    if nb == v:
                        return dist[nb]
                    q.append(nb)
        return float('inf')

    def find_shortest_path(u, v, edges, adj):
        """Find shortest path from u to v using only edges in 'edges'."""
        if u == v:
            return [u]
        
        from collections import deque
        parent = {u: None}
        q = deque([u])
        while q:
            cur = q.popleft()
            if cur == v:
                break
            for nb in adj.get(cur, []):
                e = tuple(sorted((cur, nb)))
                if e in edges and nb not in parent:
                    parent[nb] = cur
                    q.append(nb)
        
        if v not in parent:
            return None
        
        # Reconstruct path
        path = []
        cur = v
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    def decompose_multi_start_smart(edges, num_restarts=3, max_time_per=10.0):
        """Run smart greedy multiple times with different randomizations."""
        if not edges:
            return []
        
        best_result = None
        best_count = float('inf')
        
        for i in range(num_restarts):
            elapsed = time.time() - start_time
            if elapsed > deadline - 5:
                break
            
            # Randomize by shuffling edge processing order
            shuffled_edges = list(edges)
            random.Random(42 + i).shuffle(shuffled_edges)
            
            paths = decompose_smart_greedy(shuffled_edges, max_time=max_time_per)
            
            if paths and len(paths) < best_count:
                best_count = len(paths)
                best_result = paths

        return best_result if best_result else []

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for idx, spec in enumerate(specs):
        if time.time() > deadline:
            break
        edges = instance(*spec)
        
        time_left = deadline - time.time()
        if time_left > 60:
            restarts = 5
            max_time_per = 8.0
        elif time_left > 30:
            restarts = 3
            max_time_per = 5.0
        else:
            restarts = 1
            max_time_per = 2.0
        
        paths = decompose_multi_start_smart(edges, restarts, max_time_per)
        results.append(paths)

    # Fill remaining with trivial edge paths
    while len(results) < 3:
        spec = specs[len(results)]
        edges = instance(*spec)
        results.append([[u, v] for u, v in edges])

    return results

