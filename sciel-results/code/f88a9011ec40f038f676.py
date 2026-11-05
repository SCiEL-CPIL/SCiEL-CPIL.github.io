import time
import random

# Pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute degrees for heuristic
    degrees = [len(adj[i]) for i in range(n)]
    
    best_path = []
    best_len = 0
    
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin
    
    rng = random.Random(42)
    
    # Strategy: Randomized Greedy with Local Improvement (Insertion)
    
    def improve_path(path, visited):
        """Try to insert unvisited vertices into the path to make it longer."""
        improved = True
        while improved:
            improved = False
            # Check ends first (cheaper)
            # Prepend
            start_node = path[0]
            for u in list(adj[start_node]):
                if u not in visited:
                    path.insert(0, u)
                    visited.add(u)
                    improved = True
                    break
            # Append
            if improved:
                continue
            end_node = path[-1]
            for u in list(adj[end_node]):
                if u not in visited:
                    path.append(u)
                    visited.add(u)
                    improved = True
                    break
            
            # Insert in middle
            if improved:
                continue
                
            # For each unvisited vertex u, check if it can be inserted between path[i] and path[i+1]
            # This requires edges (u, path[i]) and (u, path[i+1])
            # To speed up, iterate over unvisited vertices
            unvisited = [u for u in range(n) if u not in visited]
            for u in unvisited:
                # If u is not connected to any node in path, skip
                # Quick check: does u share an edge with any neighbor in path?
                # Actually, we just need to find i such that path[i] in adj[u] and path[i+1] in adj[u]
                # Let's check neighbors of u that are in the path
                neighbors_in_path = [i for i, node in enumerate(path) if node in adj[u]]
                if len(neighbors_in_path) < 2:
                    continue
                
                # We need two neighbors in the path that are adjacent in the path sequence
                # i.e., indices i and i+1 in path such that path[i] in adj[u] and path[i+1] in adj[u]
                # Or path[i+1] in adj[u] and path[i] in adj[u]
                
                # Optimization: create a set of neighbors in path for faster lookup?
                # path is a list.
                # Let's just check consecutive pairs
                found_insertion = False
                for i in range(len(path) - 1):
                    if path[i] in adj[u] and path[i+1] in adj[u]:
                        path.insert(i+1, u)
                        visited.add(u)
                        improved = True
                        found_insertion = True
                        break
                    # Also check reverse? No, path is directed sequence.
                    # If path[i] and path[i+1] are both connected to u, we can insert u between them.
                    # The new segment is path[i] - u - path[i+1].
                    # Edges (path[i], u) and (u, path[i+1]) must exist.
                    # This is exactly what we check.
                
                if found_insertion:
                    break
        
        return path, visited

    # Run multiple randomized greedy searches
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Pick a random start node
        start = rng.randint(0, n-1)
        
        # Greedy construction with random tie-breaking based on degree
        path = [start]
        visited = {start}
        
        while True:
            last = path[-1]
            candidates = [v for v in adj[last] if v not in visited]
            if not candidates:
                break
            
            # Heuristic: pick candidate with lowest degree (Warnsdorff)
            # Add some randomness to escape local minima
            # Sort by degree, then pick randomly from top K?
            # Or just pick randomly from candidates with low degree.
            
            # Let's use a weighted random choice based on inverse degree?
            # Or simply: pick the one with min degree, but with 20% chance pick random.
            
            if rng.random() < 0.2:
                next_node = rng.choice(candidates)
            else:
                # Min degree
                next_node = min(candidates, key=lambda x: (degrees[x], x))
            
            path.append(next_node)
            visited.add(next_node)
        
        # Local improvement
        path, visited = improve_path(path, visited)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
            
        # Optional: if we found a Hamiltonian path (len 100), we can stop
        if best_len == n:
            break
            
    return best_path

