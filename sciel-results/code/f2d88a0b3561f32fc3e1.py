"""Path decomposition of three fixed connected graphs (Gallai's conjecture)."""
import time
import random
from collections import defaultdict, deque

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _solve_graph(n, p, seed, time_budget=30.0):
    edges = _instance(n, p, seed)
    m = len(edges)
    if m == 0:
        return []
    
    edge_set = set()
    for (u, v) in edges:
        edge_set.add((min(u, v), max(u, v)))
    
    adj = defaultdict(set)
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Compute degrees
    deg = [0] * n
    for (u, v) in edges:
        deg[u] += 1
        deg[v] += 1
    
    # Odd-degree vertices
    odd_vertices = [v for v in range(n) if deg[v] % 2 == 1]
    
    # Strategy 1: If all degrees are even, we can do an Eulerian circuit
    # and then break it into simple paths optimally
    if not odd_vertices:
        # Find Eulerian circuit
        euler = _eulerian_circuit(n, adj, edge_set)
        if euler:
            # Break Eulerian circuit into simple paths
            # The minimum number of simple paths to cover an Eulerian circuit
            # depends on the structure, but we can use a greedy approach
            paths = _break_circuit_into_paths(euler, edge_set, n)
            return paths
    
    # Strategy 2: Use edge coloring / matching approach
    # The minimum number of paths = m - max_matching
    # where max_matching is in the bipartite double cover
    
    # Build bipartite graph
    left_adj = [[] for _ in range(n)]
    for (u, v) in edges:
        left_adj[u].append(v)
        left_adj[v].append(u)
    
    # Maximum bipartite matching using Hopcroft-Karp for better performance
    matching = _hopcroft_karp(n, left_adj)
    
    # Build succ/pred from matching
    succ = [-1] * n
    pred = [-1] * n
    matched_edge_set = set()
    
    for v in range(n):
        if matching[v] != -1:
            u = matching[v]
            succ[u] = v
            pred[v] = u
            matched_edge_set.add((min(u, v), max(u, v)))
    
    # Extract paths from matching
    all_paths = []
    visited = [False] * n
    
    # Find path starts (vertices with no predecessor)
    for v in range(n):
        if pred[v] == -1 and succ[v] != -1:
            path = [v]
            current = v
            while succ[current] != -1:
                current = succ[current]
                path.append(current)
                visited[current] = True
            all_paths.append(path)
            visited[v] = True
    
    # Find cycles
    for v in range(n):
        if not visited[v] and pred[v] != -1 and succ[v] != -1:
            cycle = [v]
            current = v
            seen = {v}
            while True:
                current = succ[current]
                if current in seen:
                    if current == v:
                        all_paths.append(cycle)
                    break
                cycle.append(current)
                seen.add(current)
                visited[current] = True
    
    # Add single-edge paths for unmatched edges
    for (u, v) in edges:
        e = (min(u, v), max(u, v))
        if e not in matched_edge_set:
            all_paths.append([u, v])
    
    # Verify
    covered = set()
    valid = True
    for path in all_paths:
        if len(path) < 2:
            continue
        if len(set(path)) != len(path):
            valid = False
            break
        for i in range(len(path) - 1):
            a, b = path[i], path[i+1]
            e = (min(a, b), max(a, b))
            if e in covered or e not in edge_set:
                valid = False
                break
            covered.add(e)
        if not valid:
            break
    
    if not valid or covered != edge_set:
        # Fallback: try greedy path extension
        fallback = _greedy_path_decomposition(n, adj, edge_set, time_budget)
        if fallback:
            return fallback
        return [[u, v] for (u, v) in edges]
    
    return all_paths

def _hopcroft_karp(n, left_adj):
    """Hopcroft-Karp algorithm for maximum bipartite matching."""
    INF = float('inf')
    pair_u = [-1] * n  # pair_u[u] = v if u is matched to v
    pair_v = [-1] * n  # pair_v[v] = u if v is matched to u
    dist = [0] * n
    
    from collections import deque
    
    def bfs():
        queue = deque()
        for u in range(n):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = INF
        found = False
        while queue:
            u = queue.popleft()
            for v in left_adj[u]:
                if pair_v[v] == -1:
                    found = True
                elif dist[pair_v[v]] == INF:
                    dist[pair_v[v]] = dist[u] + 1
                    queue.append(pair_v[v])
        return found
    
    def dfs(u):
        for v in left_adj[u]:
            if pair_v[v] == -1 or (dist[pair_v[v]] == dist[u] + 1 and dfs(pair_v[v])):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = INF
        return False
    
    matching_size = 0
    while bfs():
        for u in range(n):
            if pair_u[u] == -1:
                if dfs(u):
                    matching_size += 1
    
    return pair_v  # pair_v[v] = u means v is matched to u

def _eulerian_circuit(n, adj, edge_set):
    """Find an Eulerian circuit in the graph. Returns list of vertices or None."""
    # Use Hierholzer's algorithm
    # First, find a starting vertex with non-zero degree
    start = None
    for v in range(n):
        if adj[v]:
            start = v
            break
    if start is None:
        return None
    
    # We need to work with edge counts
    remaining = defaultdict(set)
    for (u, v) in edge_set:
        remaining[u].add(v)
        remaining[v].add(u)
    
    circuit = []
    stack = [start]
    
    while stack:
        v = stack[-1]
        if remaining[v]:
            # Pick an edge
            next_v = next(iter(remaining[v]))
            remaining[v].remove(next_v)
            remaining[next_v].remove(v)
            stack.append(next_v)
        else:
            circuit.append(stack.pop())
    
    circuit.reverse()
    
    # Verify it's a valid Eulerian circuit
    if len(circuit) != len(edge_set) + 1:
        return None
    for i in range(len(circuit) - 1):
        e = (min(circuit[i], circuit[i+1]), max(circuit[i], circuit[i+1]))
        if e not in edge_set:
            return None
    
    return circuit

def _break_circuit_into_paths(circuit, edge_set, n):
    """Break an Eulerian circuit into simple paths."""
    # Greedy approach: extend paths as much as possible without repeating vertices
    paths = []
    used_edges = set()
    remaining_edges = set(edge_set)
    
    # Try to find long simple paths
    for i in range(len(circuit) - 1):
        if (min(circuit[i], circuit[i+1]), max(circuit[i], circuit[i+1])) in used_edges:
            continue
        if (min(circuit[i], circuit[i+1]), max(circuit[i], circuit[i+1])) not in remaining_edges:
            continue
        
        # Start a path from circuit[i]
        path = [circuit[i]]
        current = circuit[i]
        visited_in_path = {current}
        
        while True:
            # Find an unused edge from current
            next_v = None
            for neighbor in adj.get(current, []):
                e = (min(current, neighbor), max(current, neighbor))
                if e in remaining_edges and neighbor not in visited_in_path:
                    next_v = neighbor
                    break
            
            if next_v is None:
                break
            
            path.append(next_v)
            visited_in_path.add(next_v)
            remaining_edges.remove((min(current, next_v), max(current, next_v)))
            used_edges.add((min(current, next_v), max(current, next_v)))
            current = next_v
        
        if len(path) > 1:
            paths.append(path)
    
    # Add remaining edges as single-edge paths
    for e in remaining_edges:
        paths.append([e[0], e[1]])
    
    return paths

def _greedy_path_decomposition(n, adj, edge_set, time_budget=30.0):
    """Greedy path decomposition: repeatedly find the longest simple path."""
    paths = []
    remaining = set(edge_set)
    start_time = time.time()
    
    # Precompute adjacency for quick lookup
    adj_list = defaultdict(list)
    for (u, v) in edge_set:
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    # Sort vertices by degree (descending) to start from high-degree vertices
    deg = [len(adj_list[v]) for v in range(n)]
    vertices_by_degree = sorted(range(n), key=lambda x: -deg[x])
    
    while remaining:
        if time.time() - start_time > time_budget:
            break
        
        best_path = None
        best_len = 0
        
        # Try starting from each vertex
        for start_v in vertices_by_degree:
            if not adj_list[start_v]:
                continue
            
            # Use DFS to find longest simple path
            path, length = _find_longest_simple_path(start_v, adj_list, remaining, n)
            if length > best_len:
                best_len = length
                best_path = path
        
        if best_path and len(best_path) > 1:
            paths.append(best_path)
            # Remove used edges
            for i in range(len(best_path) - 1):
                e = (min(best_path[i], best_path[i+1]), max(best_path[i], best_path[i+1]))
                remaining.discard(e)
        else:
            # No longer path found, add a single edge
            e = next(iter(remaining))
            paths.append([e[0], e[1]])
            remaining.discard(e)
    
    # Add any remaining edges as single-edge paths
    for e in remaining:
        paths.append([e[0], e[1]])
    
    return paths

def _find_longest_simple_path(start, adj_list, remaining_edges, n, depth_limit=15):
    """Find a long simple path starting from 'start' using DFS with pruning."""
    best_path = [start]
    best_len = 1
    
    def dfs(current, path, visited):
        nonlocal best_path, best_len
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
        
        if len(path) >= depth_limit:
            return
        
        for neighbor in adj_list[current]:
            e = (min(current, neighbor), max(current, neighbor))
            if e in remaining_edges and neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(neighbor, path, visited)
                path.pop()
                visited.remove(neighbor)
    
    dfs(start, [start], {start})
    return best_path, best_len

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s kill
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            # Not enough time left, use fallback
            edges = _instance(*spec)
            results.append([[u, v] for (u, v) in edges])
        else:
            # Allocate time budget based on remaining time
            remaining_time = deadline - elapsed - 5
            time_budget = max(5.0, min(30.0, remaining_time / 3))
            paths = _solve_graph(*spec, time_budget=time_budget)
            results.append(paths)
    
    return results
