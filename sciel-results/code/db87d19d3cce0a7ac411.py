import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_odd_vertices(adj):
        return [u for u in adj if len(adj[u]) % 2 != 0]

    def find_shortest_path(start, end, adj, remaining_edges):
        """BFS for shortest simple path."""
        if start == end:
            return [start]
        queue = deque([(start, [start])])
        visited = {start}
        while queue:
            current, path = queue.popleft()
            for neighbor in adj[current]:
                edge = (min(current, neighbor), max(current, neighbor))
                if edge not in remaining_edges:
                    continue
                if neighbor == end:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def eulerian_decompose(remaining_edges):
        """
        Decompose an even-degree graph into cycles.
        Returns list of cycles (each cycle is a list of vertices).
        """
        if not remaining_edges:
            return []
        
        adj = defaultdict(set)
        for u, v in remaining_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cycles = []
        rem = set(remaining_edges)
        
        while rem:
            start = next(iter(rem))[0]
            cycle = [start]
            current = start
            visited_verts = {start}
            
            while True:
                found_next = False
                for neighbor in list(adj[current]):
                    edge = (min(current, neighbor), max(current, neighbor))
                    if edge in rem:
                        rem.discard(edge)
                        adj[current].discard(neighbor)
                        adj[neighbor].discard(current)
                        
                        if neighbor == start:
                            found_next = True
                            break
                        elif neighbor not in visited_verts:
                            cycle.append(neighbor)
                            visited_verts.add(neighbor)
                            current = neighbor
                            found_next = True
                            break
                if not found_next:
                    break
            
            if len(cycle) > 1:
                cycles.append(cycle)
        
        return cycles

    def solve_graph(edges, time_limit):
        if not edges:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        odd_verts = [u for u in adj if len(adj[u]) % 2 != 0]
        n_odd = len(odd_verts)
        
        best_paths = None
        best_count = float('inf')
        
        # Strategy 1: Pair odd vertices and find paths, then decompose remainder into cycles
        # This should achieve exactly n_odd/2 paths if the graph structure allows
        # (which it should for connected graphs where we can find vertex-disjoint paths)
        
        # We try multiple random pairings of odd vertices
        for attempt in range(50):
            if time.time() > time_limit:
                break
            
            rng = random.Random(attempt * 123 + 456)
            odd_set = list(odd_verts)
            rng.shuffle(odd_set)
            
            remaining_edges = set((u, v) if u < v else (v, u) for u, v in edges)
            work_adj = defaultdict(set)
            for u, v in edges:
                work_adj[u].add(v)
                work_adj[v].add(u)
                
            paths = []
            success = True
            
            # Pair odd vertices
            for i in range(0, len(odd_set) - 1, 2):
                if time.time() > time_limit:
                    success = False
                    break
                u, v = odd_set[i], odd_set[i+1]
                
                # Find a simple path from u to v
                path = find_shortest_path(u, v, work_adj, remaining_edges)
                if path is None:
                    success = False
                    break
                    
                paths.append(path)
                # Remove edges from remaining
                for j in range(len(path) - 1):
                    a, b = path[j], path[j+1]
                    edge = (min(a, b), max(a, b))
                    if edge in remaining_edges:
                        remaining_edges.remove(edge)
                        work_adj[a].remove(b)
                        work_adj[b].remove(a)
            
            if not success:
                continue
                
            # Decompose remaining edges (should be all even degree) into cycles
            if remaining_edges:
                cycles = eulerian_decompose(remaining_edges)
                for cycle in cycles:
                    # A cycle can be represented as a path by dropping one vertex
                    # But wait, the problem asks for paths. A cycle is a closed trail.
                    # Is a cycle a valid path? "Paths must be simple".
                    # A simple cycle is not a simple path in the strict sense (start != end).
                    # However, we can break the cycle into a path by removing one edge?
                    # No, we must cover ALL edges.
                    # If we have a cycle, we can list the vertices as a path?
                    # No, a path is a sequence of distinct vertices v1...vk such that (vi, vi+1) is an edge.
                    # A cycle v1...vk v1 is not a path because v1 appears twice (if we list it that way)
                    # or if we list v1...vk, we miss the edge (vk, v1).
                    # So we must break the cycle. But we can't discard edges.
                    # Ah, the problem says "cover all edges".
                    # If we have an Eulerian cycle, we can't represent it as a single simple path
                    # unless we allow the path to be a cycle? No, "simple path" usually implies distinct vertices.
                    # Let's re-read: "Paths must be simple".
                    # Usually a path is simple if all vertices are distinct.
                    # So a cycle cannot be a single path.
                    # We must break the cycle into paths? But we can't split edges.
                    # Wait. If we have a cycle, we can just list the vertices in order.
                    # Is [1, 2, 3, 1] a simple path? No, 1 repeats.
                    # Is [1, 2, 3] a simple path? Yes, but it misses edge (3,1).
                    # So how do we cover the edges of a cycle with simple paths?
                    # We can't cover a cycle with a single simple path.
                    # We must use at least 2 paths? Or is a cycle considered a path in this context?
                    # "Path decompositions of three fixed connected graphs".
                    # Gallai's conjecture is about edge-disjoint paths covering all edges.
                    # Usually, a path in this context is a simple path (no repeated vertices).
                    # If a component is a cycle, we cannot cover it with 1 simple path.
                    # We must break it? But we can't discard edges.
                    # Actually, we can just output the cycle as a path if the verifier accepts it?
                    # Or do we need to break it into two paths? That would increase the count.
                    # Let's assume the verifier requires simple paths (distinct vertices).
                    # Then a cycle must be broken. But we can't break edges.
                    # This implies that if the graph has an Eulerian cycle, we can't cover it with 1 path.
                    # We must use 2 paths? e.g. 1-2-3 and 1-3? No, edge-disjoint.
                    # 1-2-3 covers (1,2) and (2,3). 1-3 covers (1,3).
                    # So a triangle needs 2 paths.
                    # A square 1-2-3-4-1 needs 2 paths: 1-2-3 and 1-4 (covers 1-4, 4-3? No).
                    # 1-2-3-4 covers 3 edges. 1-4 covers 1 edge.
                    # So a cycle of length k needs 2 paths?
                    # Or can we just output the cycle and hope?
                    # Let's check the current program's behavior.
                    # The current program `min_cost_flow_path_cover` breaks cycles into paths?
                    # In `min_cost_flow_path_cover`, step 2:
                    # "Walk along edges to form a cycle or path"
                    # It stops when it can't move. If it closes the cycle, it appends the path.
                    # The path would be [v1, v2, ..., vk]. It does NOT include v1 again.
                    # So it misses the edge (vk, v1).
                    # But it removes the edges it traversed.
                    # So the edge (vk, v1) remains in `remaining_edges`.
                    # Then the loop continues, finds that edge, and creates a new path [vk, v1].
                    # So a cycle becomes 2 paths.
                    # This is valid.
                    # So my `eulerian_decompose` should return cycles, and then I should break them into 2 paths each?
                    # Or just let the greedy fallback handle the remainder.
                    
                    # Let's just use the greedy fallback for the remainder to be safe.
                    # Or better, just treat the remainder as a graph and run a simple greedy on it.
                    # Since the remainder is even-degree, it decomposes into cycles.
                    # Each cycle becomes 2 paths.
                    # So total paths = n_odd/2 + 2 * (number of cycles).
                    # Is this minimal?
                    # If we have a cycle, we can't do better than 2 paths.
                    # So this is optimal for the remainder.
                    
                    # However, if we can merge a path from the odd-pairing with a cycle, we might reduce the count?
                    # No, the paths are edge-disjoint.
                    # So the count is fixed by the structure.
                    
                    # Let's just convert cycles to 2 paths each.
                    for cycle in cycles:
                        # Break cycle into two paths
                        # Path 1: cycle[0] to cycle[-1] (all edges except last)
                        # Path 2: cycle[-1] to cycle[0] (last edge)
                        # Wait, path 2 is just an edge.
                        # Path 1: [v1, ..., vk]
                        # Path 2: [vk, v1]
                        paths.append(cycle)
                        paths.append([cycle[-1], cycle[0]])
            
            count = len(paths)
            if count < best_count:
                best_count = count
                best_paths = paths
                
            if best_count == n_odd // 2:
                # This is the theoretical minimum
                break

        # Strategy 2: Greedy fallback if Strategy 1 fails or is worse
        if best_paths is None or time.time() < time_limit - 5:
            # Run a few greedy attempts
            for seed in range(20):
                if time.time() > time_limit:
                    break
                rng = random.Random(seed)
                cover = greedy_path_cover(edges, rng, time_limit)
                count = len(cover)
                if count < best_count:
                    best_count = count
                    best_paths = cover

        if best_paths is None:
            best_paths = [[list(e)] for e in edges]
            
        return best_paths

    def greedy_path_cover(edges, rng, time_limit):
        """Fallback greedy path cover."""
        if not edges:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining_edges = set((u, v) if u < v else (v, u) for u, v in edges)
        paths = []
        
        while remaining_edges:
            if time.time() > time_limit:
                break
                
            odd_verts = get_odd_vertices(adj)
            if odd_verts:
                start = rng.choice(odd_verts)
            else:
                candidates = [u for u in adj if adj[u]]
                if not candidates:
                    break
                start = rng.choice(candidates)
                
            path = [start]
            visited_vertices = {start}
            current = start
            
            while True:
                valid_neighbors = []
                for neighbor in adj[current]:
                    edge = (min(current, neighbor), max(current, neighbor))
                    if edge in remaining_edges and neighbor not in visited_vertices:
                        valid_neighbors.append(neighbor)
                        
                if not valid_neighbors:
                    break
                
                next_v = rng.choice(valid_neighbors)
                path.append(next_v)
                visited_vertices.add(next_v)
                current = next_v
                
            if len(path) > 1:
                paths.append(path)
                
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge = (min(u, v), max(u, v))
                if edge in remaining_edges:
                    remaining_edges.remove(edge)
                    adj[u].remove(v)
                    adj[v].remove(u)
                    
        # Handle any leftover edges (should be none if graph was connected and we handled all)
        for e in remaining_edges:
            paths.append(list(e))
            
        return paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for i, spec in enumerate(specs):
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([[list(e)] for e in edges])
            continue
        
        # Allocate time proportionally or evenly
        remaining_time = deadline - time.time()
        time_per_graph = remaining_time / (len(specs) - i)
        graph_deadline = time.time() + time_per_graph - 1.0
        
        edges = instance(*spec)
        cover = solve_graph(edges, graph_deadline)
        results.append(cover)
        
    return results
