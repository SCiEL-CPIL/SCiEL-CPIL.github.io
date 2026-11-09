import random
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        if not edges:
            return []
        
        # Build adjacency list
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining_edges = set(edges)
        paths = []
        
        def get_degree(node):
            count = 0
            for nb in adj[node]:
                edge = tuple(sorted((node, nb)))
                if edge in remaining_edges:
                    count += 1
            return count

        def find_longest_path(start_node, max_steps=1000):
            """Find a long simple path starting from start_node using backtracking with heuristics."""
            best_path = [start_node]
            best_len = 1
            
            def dfs(path, visited):
                nonlocal best_path, best_len
                current = path[-1]
                
                # Update best if this path is longer
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path[:]
                
                # Get available neighbors
                candidates = []
                for neighbor in adj[current]:
                    if neighbor in visited:
                        continue
                    edge = tuple(sorted((current, neighbor)))
                    if edge in remaining_edges:
                        candidates.append(neighbor)
                
                if not candidates:
                    return
                
                # Sort by degree (prefer high-degree nodes to keep options open)
                candidates.sort(key=lambda x: get_degree(x), reverse=True)
                
                # Try each candidate (limited depth to avoid exponential blowup)
                for next_node in candidates[:3]:  # Limit branching factor
                    edge = tuple(sorted((current, next_node)))
                    remaining_edges.discard(edge)
                    path.append(next_node)
                    visited.add(next_node)
                    dfs(path, visited)
                    path.pop()
                    visited.remove(next_node)
                    remaining_edges.add(edge)
            
            dfs([start_node], {start_node})
            return best_path

        def find_greedy_path(start_node):
            """Fast greedy path finding."""
            path = [start_node]
            visited_nodes = {start_node}
            while True:
                current = path[-1]
                candidates = []
                for neighbor in adj[current]:
                    if neighbor in visited_nodes:
                        continue
                    edge = tuple(sorted((current, neighbor)))
                    if edge in remaining_edges:
                        candidates.append(neighbor)
                
                if not candidates:
                    break
                
                # Choose neighbor with highest remaining degree
                candidates.sort(key=lambda x: get_degree(x), reverse=True)
                next_node = candidates[0]
                
                path.append(next_node)
                visited_nodes.add(next_node)
                
                edge = tuple(sorted((current, next_node)))
                remaining_edges.discard(edge)
            
            return path

        def try_merge_paths(paths_list, remaining):
            """Try to merge two paths into one if they share an endpoint."""
            # This is a placeholder for future optimization
            return paths_list

        # Main loop
        while remaining_edges:
            if time.time() - start_time > time_limit:
                # Return best effort - add remaining edges as individual paths
                for e in remaining_edges:
                    paths.append(list(e))
                break
            
            # Get nodes with remaining edges
            nodes_with_edges = set()
            for u, v in remaining_edges:
                nodes_with_edges.add(u)
                nodes_with_edges.add(v)
            
            if not nodes_with_edges:
                break
            
            # Sort nodes by degree to prioritize starting at high-degree nodes
            sorted_nodes = sorted(nodes_with_edges, key=get_degree, reverse=True)
            
            # Try multiple start nodes to find the longest path
            best_path = []
            best_len = 0
            
            # Try top few start nodes with both greedy and limited backtracking
            for start_node in sorted_nodes[:8]:
                # First try greedy (fast)
                path = find_greedy_path(start_node)
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path[:]
                
                # Check if we should try backtracking (only if time permits)
                if time.time() - start_time < time_limit - 5.0 and best_len < 10:
                    # Save state
                    saved_edges = remaining_edges.copy()
                    bt_path = find_longest_path(start_node)
                    if len(bt_path) > best_len:
                        best_len = len(bt_path)
                        best_path = bt_path[:]
                    # Restore state if backtracking didn't improve
                    if len(bt_path) <= best_len or len(bt_path) == len(best_path):
                        remaining_edges = saved_edges
            
            if len(best_path) > 1:
                paths.append(best_path)
            else:
                # If no path found, take one edge
                u, v = next(iter(remaining_edges))
                paths.append([u, v])
                remaining_edges.discard((u, v))
                
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
        
    return results
