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

def _hopcroft_karp(n, edges):
    """
    Finds maximum matching in the bipartite double cover of the graph.
    Left nodes: 0..n-1
    Right nodes: 0..n-1
    Edge (u, v) in original graph creates (L_u, R_v) and (L_v, R_u) in bipartite graph.
    Returns matching array of size n, where matching[i] = j means L_i is matched to R_j.
    """
    adj = [[] for _ in range(n)]
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    pair_u = [-1] * n
    pair_v = [-1] * n
    
    INF = float('inf')
    dist = [0] * n
    
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
            for v in adj[u]:
                w = pair_v[v]
                if w != -1 and dist[w] == INF:
                    dist[w] = dist[u] + 1
                    queue.append(w)
                elif w == -1:
                    found = True
        return found

    def dfs(u):
        for v in adj[u]:
            w = pair_v[v]
            if w == -1 or (dist[w] == dist[u] + 1 and dfs(w)):
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
                    
    return pair_u

def _extract_paths_from_matching(n, edges, matching):
    """
    Given a matching (succ array) derived from bipartite matching, 
    extract edge-disjoint paths and cycles.
    """
    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    
    succ = [None] * n
    pred = [None] * n
    for u in range(n):
        v = matching[u]
        if v is not None:
            succ[u] = v
            pred[v] = u
            
    paths = []
    visited = [False] * n
    
    # Find paths starting from nodes with no predecessor
    for u in range(n):
        if pred[u] is None and not visited[u]:
            path = [u]
            visited[u] = True
            curr = u
            while succ[curr] is not None:
                curr = succ[curr]
                if visited[curr]:
                    break
                path.append(curr)
                visited[curr] = True
            if len(path) > 1:
                paths.append(path)
                
    # Find cycles
    for u in range(n):
        if not visited[u]:
            cycle = [u]
            visited[u] = True
            curr = u
            while succ[curr] is not None and not visited[succ[curr]]:
                curr = succ[curr]
                cycle.append(curr)
                visited[curr] = True
            if len(cycle) > 1:
                paths.append(cycle)

    # Verify validity
    covered_edges = set()
    valid = True
    for path in paths:
        if len(path) < 2:
            valid = False
            break
        if len(set(path)) != len(path):
            valid = False
            break
        for i in range(len(path) - 1):
            a, b = path[i], path[i+1]
            e = (min(a, b), max(a, b))
            if e in covered_edges:
                valid = False
                break
            if e not in edge_set:
                valid = False
                break
            covered_edges.add(e)
        if not valid:
            break
    
    if not valid or len(covered_edges) != len(edge_set):
        return None
        
    return paths

def _greedy_longest_path(n, edges, time_budget=10.0):
    """
    Heuristic: Greedy extraction of long paths with multiple restarts.
    Uses Warnsdorff's rule (choose neighbor with fewest options) to extend paths.
    """
    adj = defaultdict(set)
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    remaining_edges = set((min(u, v), max(u, v)) for u, v in edges)
    paths = []
    start_time = time.time()
    
    while remaining_edges:
        if time.time() - start_time > time_budget:
            # Fallback: single edges
            for e in remaining_edges:
                paths.append([e[0], e[1]])
            break
            
        best_path = None
        best_len = 0
        
        # Get active vertices
        active_vertices = set()
        for u, v in remaining_edges:
            active_vertices.add(u)
            active_vertices.add(v)
        
        vertices_list = list(active_vertices)
        random.shuffle(vertices_list)
        
        # Limit attempts to keep speed up
        max_starts = min(30, len(vertices_list))
        
        for start in vertices_list[:max_starts]:
            # Quick check: does start have any remaining edges?
            has_edge = False
            for n in adj[start]:
                e = (min(start, n), max(start, n))
                if e in remaining_edges:
                    has_edge = True
                    break
            if not has_edge:
                continue

            current_path = [start]
            visited = {start}
            current_node = start
            
            while True:
                valid_neighbors = []
                for n in adj[current_node]:
                    if n in visited:
                        continue
                    e = (min(current_node, n), max(current_node, n))
                    if e in remaining_edges:
                        valid_neighbors.append(n)
                
                if not valid_neighbors:
                    break
                
                # Heuristic: choose neighbor with fewest remaining options (Warnsdorff)
                def count_options(node):
                    count = 0
                    for m in adj[node]:
                        if m in visited:
                            continue
                        e = (min(node, m), max(node, m))
                        if e in remaining_edges:
                            count += 1
                    return count
                
                valid_neighbors.sort(key=count_options)
                next_node = valid_neighbors[0]
                
                current_path.append(next_node)
                visited.add(next_node)
                current_node = next_node
            
            if len(current_path) > 1:
                if len(current_path) > best_len:
                    best_path = current_path
                    best_len = len(current_path)
        
        if best_path:
            paths.append(best_path)
            # Remove edges
            for i in range(len(best_path) - 1):
                u, v = best_path[i], best_path[i+1]
                e = (min(u, v), max(u, v))
                remaining_edges.discard(e)
        else:
            # If no path found, take a single edge
            if remaining_edges:
                e = next(iter(remaining_edges))
                paths.append([e[0], e[1]])
                remaining_edges.discard(e)
                
    return paths

def _solve_graph(n, p, seed, time_budget=10.0):
    edges = _instance(n, p, seed)
    m = len(edges)
    if m == 0:
        return []
    
    # 1. Try Maximum Matching approach (structural lower bound)
    matching = _hopcroft_karp(n, edges)
    paths_mm = _extract_paths_from_matching(n, edges, matching)
    
    # 2. Try Greedy Longest Path heuristic (often better in practice for random graphs)
    #    We run a few restarts or just one strong pass.
    paths_greedy = _greedy_longest_path(n, edges, time_budget)
    
    # 3. Choose the better result (fewer paths)
    if paths_mm is None:
        return paths_greedy
        
    if len(paths_greedy) < len(paths_mm):
        return paths_greedy
    else:
        return paths_mm

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            # Not enough time left, use fallback
            edges = _instance(*spec)
            results.append([[u, v] for (u, v) in edges])
        else:
            # Allocate time budget
            remaining_time = deadline - elapsed - 5
            time_budget = max(1.0, min(20.0, remaining_time / 3))
            paths = _solve_graph(*spec, time_budget=time_budget)
            results.append(paths)
    
    return results
