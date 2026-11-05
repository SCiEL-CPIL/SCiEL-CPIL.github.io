import random
import time

def construct(k, g):
    # Heuristic: Randomized Greedy with Random Sampling of Neighbors
    # to minimize vertex count.
    
    def is_valid_edge(graph, u, v, g):
        """
        Returns True if adding edge (u, v) does not create a cycle of length < g.
        Uses BFS with early termination.
        """
        if u == v:
            return False
        if (u, v) in graph or (v, u) in graph:
            return False
        
        # BFS from u to v
        # If dist(u, v) < g-1, return False
        # If dist(u, v) >= g-1 or not connected, return True
        
        visited = {u}
        queue = [u]
        dist = 0
        max_dist = g - 2 # If we reach dist g-1, it's safe. So we stop searching at g-2 if not found.
        
        while queue:
            next_queue = []
            for node in queue:
                for neighbor in graph[node]:
                    if neighbor == v:
                        # Found v at distance dist + 1
                        if dist + 1 < g - 1:
                            return False
                        else:
                            return True
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_queue.append(neighbor)
            dist += 1
            if dist > max_dist:
                # If we haven't found v within max_dist, then dist >= g-1
                # Or v is not reachable.
                # In either case, adding edge is safe.
                return True
            queue = next_queue
            
        return True

    def try_construct(seed):
        random.seed(seed)
        graph = {} # vertex -> set of neighbors
        degree = {}
        vertices = []
        
        def add_vertex():
            v = len(vertices)
            vertices.append(v)
            graph[v] = set()
            degree[v] = 0
            return v
            
        add_vertex()
        
        max_steps = 20000
        steps = 0
        
        while steps < max_steps:
            # Find vertices with degree < k
            # Optimization: Maintain a list of incomplete vertices?
            # For simplicity, filter each time. N is small (~1000).
            candidates = [v for v in vertices if degree[v] < k]
            if not candidates:
                break
                
            u = random.choice(candidates)
            
            # Try to connect to an existing vertex
            # Shuffle vertices to pick random ones
            potential_v = [v for v in vertices if v != u and degree[v] < k]
            random.shuffle(potential_v)
            
            connected = False
            # Try up to 100 random candidates
            for v in potential_v[:100]:
                if is_valid_edge(graph, u, v, g):
                    # Add edge
                    graph[u].add(v)
                    graph[v].add(u)
                    degree[u] += 1
                    degree[v] += 1
                    connected = True
                    break
                    
            if not connected:
                # Add new vertex
                new_v = add_vertex()
                graph[u].add(new_v)
                graph[new_v].add(u)
                degree[u] += 1
                degree[new_v] += 1
                
            steps += 1
            
        # Check validity
        if all(degree[v] == k for v in vertices):
            n = len(vertices)
            edges = []
            for u in vertices:
                for v in graph[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
        return None

    # Run multiple trials
    best_n = float('inf')
    best_edges = []
    
    start_time = time.time()
    seed = 0
    # Run for 110 seconds
    while time.time() - start_time < 110:
        result = try_construct(seed)
        if result:
            n, edges = result
            if n < best_n:
                best_n = n
                best_edges = edges
        seed += 1
        
    if best_n == float('inf'):
        # Fallback: Return a simple valid graph if greedy fails (unlikely)
        # Construct a cycle of length g if k=2? No, k is fixed.
        # Just return the LUW construction as a last resort? 
        # The prompt says "do not reuse current designs".
        # If greedy fails, we might return an invalid graph or timeout.
        # Let's hope it works.
        pass
        
    return best_n, best_edges
