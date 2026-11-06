"""Path decomposition using Minimum Weight Perfect Matching with refined greedy fallback."""
import time
import sys
from collections import defaultdict, deque
import heapq
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def all_pairs_shortest_paths(n, adj):
        """Compute shortest paths between all pairs using BFS (unweighted)."""
        dist = [[-1] * n for _ in range(n)]
        prev = [[-1] * n for _ in range(n)]
        
        for s in range(n):
            dist[s][s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[s][v] == -1:
                        dist[s][v] = dist[s][u] + 1
                        prev[s][v] = u
                        queue.append(v)
        return dist, prev

    def reconstruct_path(prev, s, t):
        """Reconstruct path from s to t using prev table."""
        if prev[s][t] == -1 and s != t:
            return []
        path = []
        cur = t
        while cur != -1:
            path.append(cur)
            if cur == s:
                break
            cur = prev[s][cur]
        path.reverse()
        return path

    def min_weight_perfect_matching(odd_vertices, dist):
        """
        Find min weight perfect matching on odd_vertices using DP with bitmask.
        odd_vertices: list of vertex indices.
        dist: distance matrix.
        Returns list of pairs (u, v) and total weight.
        """
        k = len(odd_vertices)
        if k == 0:
            return [], 0
        
        size = 1 << k
        dp = [float('inf')] * size
        parent = [(-1, -1)] * size
        dp[0] = 0
        
        # Precompute weights between odd vertices
        w = [[0] * k for _ in range(k)]
        for i in range(k):
            for j in range(i+1, k):
                u = odd_vertices[i]
                v = odd_vertices[j]
                w[i][j] = dist[u][v]
                w[j][i] = dist[u][v]
        
        for mask in range(size):
            if dp[mask] == float('inf'):
                continue
            # Find first unset bit
            i = -1
            for b in range(k):
                if not (mask & (1 << b)):
                    i = b
                    break
            if i == -1:
                continue
            
            # Try to match i with j
            for j in range(i + 1, k):
                if not (mask & (1 << j)):
                    new_mask = mask | (1 << i) | (1 << j)
                    new_cost = dp[mask] + w[i][j]
                    if new_cost < dp[new_mask]:
                        dp[new_mask] = new_cost
                        parent[new_mask] = (i, j)
        
        # Backtrack
        pairs = []
        mask = size - 1
        while mask != 0:
            i, j = parent[mask]
            pairs.append((odd_vertices[i], odd_vertices[j]))
            mask ^= (1 << i) | (1 << j)
            
        return pairs, dp[size - 1]

    def find_eulerian_circuit(adj_mult, start):
        """
        Find Eulerian circuit in a multigraph using Hierholzer's algorithm.
        adj_mult: dict of list of neighbors.
        """
        adj_work = defaultdict(list)
        for u, neighbors in adj_mult.items():
            adj_work[u] = list(neighbors)
        
        path = []
        stack = [start]
        while stack:
            u = stack[-1]
            if adj_work[u]:
                v = adj_work[u].pop()
                stack.append(v)
            else:
                path.append(stack.pop())
        
        path.reverse()
        return path

    def validate_paths(paths, edges, n):
        """Validate that paths are simple, use graph edges, are edge-disjoint and cover all edges."""
        all_edges = set(tuple(sorted(e)) for e in edges)
        used = set()
        for p in paths:
            if len(set(p)) != len(p):
                return False
            for i in range(len(p)-1):
                e = tuple(sorted((p[i], p[i+1])))
                if e not in all_edges:
                    return False
                if e in used:
                    return False
                used.add(e)
        return used == all_edges

    def greedy_decompose(edges, adj, n, max_time=30):
        """
        Greedy path decomposition with Warnsdorff-like heuristic.
        Repeatedly extend paths to maximize coverage.
        """
        remaining = set(tuple(sorted(e)) for e in edges)
        paths = []
        deg = defaultdict(int)
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        
        start_time_greedy = time.time()
        
        while remaining and (time.time() - start_time_greedy) < max_time:
            # Find a vertex with highest remaining degree
            degs = defaultdict(int)
            for u, v in remaining:
                degs[u] += 1
                degs[v] += 1
            if not degs:
                break
            start = max(degs.keys(), key=lambda x: (degs[x], -x))
            
            # Try to build a long path from start
            best_path = None
            best_len = 0
            attempts = min(10, max(1, int((max_time - (time.time() - start_time_greedy)) * 2)))
            
            for attempt in range(attempts):
                path = [start]
                current = start
                used_in_path = {start}
                while True:
                    candidates = []
                    for v in adj[current]:
                        e = tuple(sorted((current, v)))
                        if e in remaining and v not in used_in_path:
                            # Warnsdorff: prefer vertices with fewer unvisited neighbors
                            score = 0
                            for w in adj[v]:
                                if w not in used_in_path:
                                    ew = tuple(sorted((v, w)))
                                    if ew in remaining:
                                        score += 1
                            candidates.append((score, v))
                    
                    if not candidates:
                        break
                    
                    # Sort by score (fewest options first) with some randomness
                    candidates.sort()
                    # Pick from top few with randomness
                    top_k = min(3, len(candidates))
                    chosen = random.choice(candidates[:top_k])
                    next_v = chosen[1]
                    path.append(next_v)
                    used_in_path.add(next_v)
                    remaining.remove(tuple(sorted((current, next_v))))
                    current = next_v
                
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path
            
            if best_path and len(best_path) > 1:
                paths.append(best_path)
            elif remaining:
                # Fallback: take any remaining edge
                e = next(iter(remaining))
                paths.append(list(e))
                remaining.remove(e)
        
        # Handle any remaining edges
        while remaining:
            e = next(iter(remaining))
            paths.append(list(e))
            remaining.remove(e)
        
        return paths

    def decompose_graph(edges, n):
        """
        Decompose graph into minimum number of simple paths using matching approach.
        """
        adj = get_adjacency(edges)
        
        # 1. Find odd vertices
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        odd_vertices = [v for v in range(n) if deg[v] % 2 == 1]
        
        if len(odd_vertices) % 2 != 0:
            return greedy_decompose(edges, adj, n)
        
        if not odd_vertices:
            # Eulerian graph: try to find a single Hamiltonian path or use greedy
            # For random graphs, this is rare. Use greedy.
            return greedy_decompose(edges, adj, n)
        
        # 2. Compute shortest paths
        dist, prev = all_pairs_shortest_paths(n, adj)
        
        # 3. Min weight perfect matching
        pairs, weight = min_weight_perfect_matching(odd_vertices, dist)
        
        # 4. Construct augmented multigraph
        adj_mult = defaultdict(list)
        for u, v in edges:
            adj_mult[u].append(v)
            adj_mult[v].append(u)
        
        # Track added edges
        added_edges = set()
        for u, v in pairs:
            path = reconstruct_path(prev, u, v)
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                added_edges.add(e)
                adj_mult[path[i]].append(path[i+1])
                adj_mult[path[i+1]].append(path[i])
        
        # Find Eulerian circuit
        start = odd_vertices[0]
        circuit = find_eulerian_circuit(adj_mult, start)
        
        # Split circuit at added edges
        paths = []
        current_path = []
        
        for i in range(len(circuit) - 1):
            u = circuit[i]
            v = circuit[i+1]
            e = tuple(sorted((u, v)))
            
            if not current_path:
                current_path.append(u)
            
            if e in added_edges:
                # Cut at added edge
                if len(current_path) > 1:
                    paths.append(current_path)
                current_path = [v]
            else:
                current_path.append(v)
        
        if len(current_path) > 1:
            paths.append(current_path)
        
        # Verify and fallback if needed
        if validate_paths(paths, edges, n):
            return paths
        
        # Fallback to greedy
        return greedy_decompose(edges, adj, n)

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        adj = get_adjacency(edges)
        
        if time.time() - start_time > deadline - 15:
            # Not enough time, use fast greedy
            results.append(greedy_decompose(edges, adj, n, max_time=5))
            continue
        
        # Try matching-based decomposition
        paths = decompose_graph(edges, n)
        
        # If matching-based didn't work well, try greedy with more time
        if paths is None or not validate_paths(paths, edges, n):
            paths = greedy_decompose(edges, adj, n, max_time=20)
        
        results.append(paths)
    
    return results
