import time
import random
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s to be safe
    
    n = P['n']
    edges = P['edges']
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Helper to find a path between u and v in graph G (adj list)
    # avoiding a set of forbidden vertices
    def find_path(u, v, forbidden):
        if u == v:
            return [u]
        if u in forbidden or v in forbidden:
            return None
        visited = {u}
        queue = deque([(u, [u])])
        while queue:
            node, path = queue.popleft()
            if node == v:
                return path
            for neighbor in adj[node]:
                if neighbor not in visited and neighbor not in forbidden:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    # Try to find a K_k subdivision
    # We iterate k from high to low
    # For each k, we try to find a set of k vertices and disjoint paths
    
    # Heuristic: pick vertices with high degree first
    degrees = {i: len(adj[i]) for i in range(n)}
    sorted_vertices = sorted(range(n), key=lambda x: -degrees[x])
    
    best_result = None
    
    # Try k from 10 down to 2
    for k in range(10, 1, -1):
        # Generate candidate sets
        # We can try random subsets, or subsets from the top-degree vertices
        # Let's try a mix: some random, some from top-degree
        
        attempts = 0
        max_attempts = 5000 # Adjust based on time
        
        while time.time() < deadline and attempts < max_attempts:
            attempts += 1
            
            # Choose k vertices
            # Bias towards high degree
            if random.random() < 0.5:
                # Pick from top 2k vertices
                candidates = sorted_vertices[:2*k]
                branch_set = random.sample(candidates, k)
            else:
                branch_set = random.sample(range(n), k)
            
            # Sort branch set for consistent processing?
            # Let's fix the order of branches to be the order they were picked, 
            # but we need to find paths for all pairs.
            
            # We will find paths one by one.
            # To increase success rate, we can order the pairs by the length of the shortest path in the original graph.
            
            # Pre-compute shortest path lengths for this branch set?
            # That's expensive. Let's just use a heuristic order:
            # Sort branches by degree?
            
            # Let's try to find paths for all pairs (i, j) with i < j
            # We maintain a set of forbidden vertices (interiors of found paths)
            # and the set of branches themselves are forbidden for interiors.
            
            forbidden = set(branch_set)
            paths = {}
            success = True
            
            # Get all pairs
            pairs = list(combinations(range(k), 2))
            
            # Sort pairs by some heuristic?
            # Let's just iterate in a fixed order, but maybe shuffle?
            # Shuffling might help avoid getting stuck in a local minimum.
            
            random.shuffle(pairs)
            
            for i, j in pairs:
                u = branch_set[i]
                v = branch_set[j]
                
                # Find a path between u and v avoiding forbidden
                path = find_path(u, v, forbidden)
                if path is None:
                    success = False
                    break
                
                # Add the interior of the path to forbidden
                for node in path[1:-1]:
                    forbidden.add(node)
                
                paths[(i, j)] = path
            
            if success:
                # We found a valid set of branches and paths
                # Convert to required format
                branches = [int(v) for v in branch_set]
                path_list = []
                for (i, j) in pairs: # Wait, we need all pairs, not just the shuffled ones?
                    # The task says "one simple path for each unordered pair"
                    # So we need a list of paths.
                    # The order of paths in the list doesn't matter, as long as they correspond to the pairs.
                    # But we need to ensure we have all of them.
                    pass
                
                # Reconstruct path list from paths dict
                # The keys are (i, j) where i, j are indices in branch_set
                # We need to output the paths as lists of vertices
                result_paths = []
                # We need to ensure we return paths for ALL pairs
                # The 'paths' dict has all pairs because we iterated through all of them
                for (i, j) in combinations(range(k), 2):
                    result_paths.append([int(v) for v in paths[(i, j)]])
                
                best_result = {'branches': branches, 'paths': result_paths}
                return best_result
                
    # If no k >= 2 found (should not happen), return a trivial one
    u, v = edges[0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
