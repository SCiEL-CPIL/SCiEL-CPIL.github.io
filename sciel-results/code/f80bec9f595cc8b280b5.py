import random
import time

def construct():
    """
    Decomposes three fixed connected graphs into edge-disjoint simple paths.
    Minimizes the total number of paths.
    """
    
    def instance_graph(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges, n, max_time=5.0):
        """
        Decomposes a graph given by its edge list into a list of simple paths.
        Returns the path with the minimum number of segments found within max_time.
        """
        if not edges:
            return []
        
        best_paths = None
        best_count = float('inf')
        
        start_time = time.time()
        iterations = 0
        
        # Precompute adjacency list for the full graph
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        edge_set = set(edges)
        
        while time.time() - start_time < max_time:
            iterations += 1
            remaining = set(edge_set)
            paths = []
            
            while remaining:
                # Pick a random edge to start a path
                # Using list(remaining)[0] is deterministic if set order is fixed, 
                # but sets are not ordered in Python 3.7+? 
                # To ensure randomness, we pick a random key.
                if not remaining:
                    break
                start_edge = random.choice(list(remaining))
                u, v = start_edge
                
                path = [u, v]
                remaining.discard(start_edge)
                
                # Extend the path greedily
                # We try to extend from both ends
                # To avoid bias, we alternate or randomize
                extend_left = random.random() < 0.5
                
                while True:
                    extended = False
                    
                    # Try extending from the right end (v)
                    if not extend_left:
                        # Choose a random neighbor to try first, or iterate
                        neighbors = list(adj[v])
                        random.shuffle(neighbors)
                        for w in neighbors:
                            edge = (v, w)
                            if edge in remaining and w not in path:
                                path.append(w)
                                remaining.discard(edge)
                                v = w
                                extended = True
                                break
                        if extended:
                            extend_left = random.random() < 0.5 # Randomly decide next direction
                            continue
                        # If couldn't extend right, try left
                        extend_left = True
                    else:
                        # Try extending from the left end (u)
                        neighbors = list(adj[u])
                        random.shuffle(neighbors)
                        for x in neighbors:
                            edge = (u, x)
                            if edge in remaining and x not in path:
                                path.insert(0, x)
                                remaining.discard(edge)
                                u = x
                                extended = True
                                break
                        if extended:
                            extend_left = random.random() < 0.5
                            continue
                        # If couldn't extend left, try right
                        extend_left = False
                    
                    # If no extension was possible in either direction
                    if not extended:
                        break

                paths.append(path)
            
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [p[:] for p in paths]
                
            # Optional: If we hit the theoretical lower bound, we can stop early?
            # Lower bound is max(1, odd_vertices // 2)
            # But for simple paths, it might be higher.
            # We'll just run for the time budget.

        if best_paths is None:
            # Fallback: single edges
            return [[u, v] for u, v in edges]
            
        return best_paths

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    # Allocate time budget
    # Total 150s. Leave 10s margin. 140s / 3 graphs = 46s per graph.
    # But the algorithm is fast. 5s per graph is likely overkill and safe.
    # Let's use 10s per graph to be safe and allow for more iterations if needed.
    
    for n, p, seed in specs:
        edges = instance_graph(n, p, seed)
        paths = decompose_graph(edges, n, max_time=10.0)
        results.append(paths)
        
    return results

# EVOLVE-BLOCK-START
def construct():
    """
    Decomposes three fixed connected graphs into edge-disjoint simple paths.
    Minimizes the total number of paths.
    """
    
    def instance_graph(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges, n, max_time=10.0):
        """
        Decomposes a graph given by its edge list into a list of simple paths.
        Returns the path with the minimum number of segments found within max_time.
        """
        if not edges:
            return []
        
        best_paths = None
        best_count = float('inf')
        
        start_time = time.time()
        
        # Precompute adjacency list for the full graph
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        edge_set = set(edges)
        
        while time.time() - start_time < max_time:
            remaining = set(edge_set)
            paths = []
            
            while remaining:
                if not remaining:
                    break
                start_edge = random.choice(list(remaining))
                u, v = start_edge
                
                path = [u, v]
                remaining.discard(start_edge)
                
                # Extend the path greedily
                extend_left = random.random() < 0.5
                
                while True:
                    extended = False
                    
                    # Try extending from the right end (v) if not forced left
                    if not extend_left:
                        neighbors = list(adj[v])
                        random.shuffle(neighbors)
                        for w in neighbors:
                            edge = (v, w)
                            if edge in remaining and w not in path:
                                path.append(w)
                                remaining.discard(edge)
                                v = w
                                extended = True
                                break
                        if extended:
                            extend_left = random.random() < 0.5
                            continue
                        extend_left = True
                    else:
                        neighbors = list(adj[u])
                        random.shuffle(neighbors)
                        for x in neighbors:
                            edge = (u, x)
                            if edge in remaining and x not in path:
                                path.insert(0, x)
                                remaining.discard(edge)
                                u = x
                                extended = True
                                break
                        if extended:
                            extend_left = random.random() < 0.5
                            continue
                        extend_left = False
                    
                    if not extended:
                        break

                paths.append(path)
            
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [p[:] for p in paths]

        if best_paths is None:
            return [[u, v] for u, v in edges]
            
        return best_paths

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for n, p, seed in specs:
        edges = instance_graph(n, p, seed)
        paths = decompose_graph(edges, n, max_time=10.0)
        results.append(paths)
        
    return results
# EVOLVE-BLOCK-END
