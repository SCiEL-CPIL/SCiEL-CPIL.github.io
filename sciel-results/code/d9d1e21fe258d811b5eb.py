import time
import random
from collections import defaultdict

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []
    start_time = time.time()
    # Leave 10 seconds buffer
    deadline = start_time + 110
    
    for n, p, seed in specs:
        edges = instance(n, p, seed)
        # Build adjacency list
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(edges)
        pieces = []
        
        # Helper to check if time is up
        def time_up():
            return time.time() > deadline

        # Find longest cycle in the subgraph defined by remaining_edges
        def find_longest_cycle():
            if not remaining_edges:
                return None
            
            # Get active vertices
            active_vertices = set()
            for u, v in remaining_edges:
                active_vertices.add(u)
                active_vertices.add(v)
            
            if not active_vertices:
                return None
            
            # Try to find a cycle of length L from len(active_vertices) down to 3
            # To speed up, we can randomize the start vertex and neighbor order
            # We use IDDFS-like search but bounded by time
            
            # Optimization: If the graph is small, we can try to find a Hamiltonian cycle first
            # But IDDFS from max_len down to 3 is safer for finding *a* long cycle quickly if Hamiltonian is hard.
            # However, for n=30, IDDFS might be slow if it explores many dead ends.
            # Let's use a heuristic: Randomized DFS with backtracking, trying to maximize depth.
            
            best_cycle = None
            max_len = len(active_vertices)
            
            # We'll try a few random restarts of a greedy DFS to find a long cycle
            # This is often faster than exhaustive IDDFS for small n
            
            for _ in range(100): # 100 restarts
                if time_up():
                    break
                start_node = random.choice(list(active_vertices))
                # Greedy DFS
                path = [start_node]
                visited = {start_node}
                current = start_node
                while True:
                    if time_up():
                        break
                    # Get neighbors in remaining graph
                    neighbors = [v for v in adj[current] if (min(current, v), max(current, v)) in remaining_edges]
                    if not neighbors:
                        break
                    
                    # Prefer neighbors not in path to extend
                    unvisited_neighbors = [v for v in neighbors if v not in visited]
                    if unvisited_neighbors:
                        # Heuristic: pick neighbor with highest degree in remaining graph?
                        # Or random? Random is good for diversity.
                        # Let's pick the one with highest degree to keep options open
                        next_node = max(unvisited_neighbors, key=lambda x: len([y for y in adj[x] if (min(x, y), max(x, y)) in remaining_edges]))
                        path.append(next_node)
                        visited.add(next_node)
                        current = next_node
                    else:
                        # All neighbors are in path. Check if we can close cycle
                        # The start node is in path. If current is connected to start_node, we have a cycle.
                        if (min(current, start_node), max(current, start_node)) in remaining_edges:
                            # Found a cycle
                            cycle_len = len(path)
                            if best_cycle is None or cycle_len > len(best_cycle):
                                best_cycle = path[:]
                            break
                        else:
                            # Dead end, cannot extend, cannot close
                            break
                
                if best_cycle is not None:
                    # If we found a cycle, we might try to improve it, but for now, we accept it
                    # To be more aggressive, we could continue searching for longer cycles
                    # But given time constraints, a greedy long cycle is a good bet
                    pass
            
            # If no cycle found via greedy, try a simple DFS for *any* cycle (length >= 3)
            if best_cycle is None:
                # Find any cycle
                visited = set()
                parent = {}
                def dfs(node):
                    if time_up():
                        return None
                    visited.add(node)
                    for neighbor in adj[node]:
                        if (min(node, neighbor), max(node, neighbor)) not in remaining_edges:
                            continue
                        if neighbor not in visited:
                            parent[neighbor] = node
                            res = dfs(neighbor)
                            if res:
                                return res
                        elif neighbor != parent.get(node, -1) and neighbor in visited:
                            # Back edge found, reconstruct cycle
                            cycle = [neighbor]
                            curr = node
                            while curr != neighbor:
                                cycle.append(curr)
                                curr = parent[curr]
                            return cycle
                    return None
                
                for v in active_vertices:
                    if time_up():
                        break
                    if v not in visited:
                        parent[v] = -1
                        res = dfs(v)
                        if res:
                            best_cycle = res
                            break
            
            return best_cycle

        while remaining_edges and not time_up():
            cycle = find_longest_cycle()
            if cycle:
                # Ensure cycle is valid and edges exist
                # Convert to list of edges
                cycle_edges = set()
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    if e in remaining_edges:
                        cycle_edges.add(e)
                    else:
                        # Invalid cycle, discard
                        cycle = None
                        break
                
                if cycle:
                    pieces.append(cycle)
                    remaining_edges -= cycle_edges
                else:
                    # If cycle validation failed, treat as no cycle found to avoid infinite loop
                    # Fallback: add single edges
                    for e in list(remaining_edges):
                        pieces.append(list(e))
                    remaining_edges = set()
            else:
                # No cycle found, remaining edges are singletons
                for e in remaining_edges:
                    pieces.append(list(e))
                remaining_edges = set()
        
        results.append(pieces)

    return results
