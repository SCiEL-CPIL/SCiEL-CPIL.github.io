import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Generate the three fixed instance graphs
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    graphs = [instance(*spec) for spec in specs]

    def decompose(edges, time_limit):
        """Decompose edges into cycles and single edges, minimizing piece count."""
        if not edges:
            return []

        # Build adjacency structure
        adj = defaultdict(set)
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u, v), max(u, v)))

        remaining_edges = set(edge_set)
        pieces = []

        def get_edge(u, v):
            return (min(u, v), max(u, v))

        def remove_cycle_edges(cycle):
            """Remove edges of the cycle from remaining_edges and adj."""
            n = len(cycle)
            if n < 3:
                return
            for i in range(n):
                u = cycle[i]
                v = cycle[(i + 1) % n]
                e = get_edge(u, v)
                if e in remaining_edges:
                    remaining_edges.discard(e)
                    adj[u].discard(v)
                    adj[v].discard(u)

        def find_longest_cycle(start, max_time):
            """
            Find the longest cycle starting and ending at 'start' using DFS.
            Returns the cycle as a list of vertices or None.
            """
            best_cycle = None
            best_len = 0
            visited = {start}
            path = [start]
            
            # We limit the search depth to avoid infinite loops in dense graphs
            # though N is small, the branching factor can be high.
            
            def dfs(node, current_len):
                nonlocal best_cycle, best_len
                if time.time() > max_time:
                    return
                
                # Check if we can close the cycle
                if current_len >= 3:
                    if node in adj[start]: # Edge (node, start) exists
                        if current_len > best_len:
                            best_len = current_len
                            best_cycle = path[:]
                
                # Extend path
                # Sort neighbors by degree to find longer paths faster?
                # Or just iterate.
                for neighbor in sorted(adj[node]):
                    if neighbor == start:
                        # Only valid if we close the cycle, handled above
                        continue
                    if neighbor in visited:
                        continue
                    
                    # Check if edge exists (it does, because it's in adj)
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, current_len + 1)
                    path.pop()
                    visited.discard(neighbor)
                    
                    if time.time() > max_time:
                        return

            dfs(start, 1)
            return best_cycle

        def find_any_cycle():
            """
            Fallback: Find any cycle, preferably a triangle.
            """
            # Look for triangles first
            for u in list(adj.keys()):
                if len(adj[u]) < 2:
                    continue
                neighbors_u = list(adj[u])
                for i in range(len(neighbors_u)):
                    v = neighbors_u[i]
                    for w in neighbors_u[i+1:]:
                        if w in adj[v]:
                            return [u, v, w]
            
            # If no triangle, try a simple DFS for any cycle
            visited = set()
            for start in adj.keys():
                if start in visited:
                    continue
                # Simple DFS to find a back edge
                stack = [(start, iter(adj[start]))]
                visited.add(start)
                path = [start]
                parent = {start: None}
                
                while stack:
                    node, neighbors = stack[-1]
                    found = False
                    for neighbor in neighbors:
                        if neighbor == parent[node]:
                            continue
                        if neighbor in visited:
                            # Found a cycle
                            # Extract cycle from path
                            if neighbor in path:
                                idx = path.index(neighbor)
                                cycle = path[idx:]
                                if len(cycle) >= 3:
                                    return cycle
                        else:
                            visited.add(neighbor)
                            parent[neighbor] = node
                            path.append(neighbor)
                            stack.append((neighbor, iter(adj[neighbor])))
                            found = True
                            break
                    if not found:
                        stack.pop()
                        path.pop()
                        
            return None

        # Main loop: extract cycles
        while remaining_edges:
            if time.time() > time_limit:
                break

            # Determine search budget for this iteration
            # Allow up to 0.5s for finding the best cycle
            search_deadline = time.time() + 0.5
            if time_limit - time.time() < 0.5:
                search_deadline = time_limit

            best_cycle = None
            best_len = 0

            # Identify candidate start vertices: those with degree >= 2
            candidates = [v for v in adj if len(adj[v]) >= 2]
            if not candidates:
                # No cycles possible, just add edges
                for u, v in sorted(remaining_edges):
                    pieces.append([u, v])
                remaining_edges.clear()
                break

            # Sort candidates by degree (descending) to find large cycles faster
            candidates.sort(key=lambda x: len(adj[x]), reverse=True)
            
            # Try to find the longest cycle from a few high-degree starts
            for start in candidates[:5]: # Limit starts to avoid explosion
                if time.time() > search_deadline:
                    break
                
                cycle = find_longest_cycle(start, search_deadline)
                if cycle and len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle
                    # If we found a very long cycle, we might stop early
                    # but let's try others if time permits
                    if best_len >= 10:
                        break

            if best_cycle:
                pieces.append(best_cycle)
                remove_cycle_edges(best_cycle)
            else:
                # Fallback to any cycle (triangle)
                cycle = find_any_cycle()
                if cycle:
                    pieces.append(cycle)
                    remove_cycle_edges(cycle)
                else:
                    # No cycles left, add single edges
                    for u, v in sorted(remaining_edges):
                        pieces.append([u, v])
                    remaining_edges.clear()
                    break

        # Any leftover edges become single-edge pieces
        for u, v in sorted(remaining_edges):
            pieces.append([u, v])

        return pieces

    results = []
    for i, edges in enumerate(graphs):
        # Allocate time based on remaining time
        remaining_time = deadline - time.time()
        time_per_graph = remaining_time / (len(graphs) - i)
        pieces = decompose(edges, time.time() + time_per_graph)
        results.append(pieces)

    return results
