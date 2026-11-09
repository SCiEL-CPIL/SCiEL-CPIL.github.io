import random
import time

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Convert to lists for faster iteration and shuffling
    adj_lists = [list(s) for s in adj]
    
    best_path = []
    start_time = time.time()
    time_limit = 50.0 # Safety margin for 60s/90s limits
    
    # Use a fixed seed for reproducibility within the run? 
    # The task says "pinned graph", but the search can be randomized.
    # We should probably use a random seed to explore different paths.
    # However, to ensure we don't get stuck in a bad random sequence, 
    # we can use a global random instance.
    rng = random.Random()
    
    while time.time() - start_time < time_limit:
        # Pick a random start node
        start = rng.randrange(n)
        
        # Perform randomized DFS
        path = [start]
        visited = {start}
        
        # We will simulate the DFS iteratively or recursively.
        # Recursive is simpler but might hit recursion limits if path is long (max 100, so it's fine).
        # Let's use a recursive helper that returns the path.
        
        def dfs(node, path, visited):
            # Try to extend the path
            neighbors = [x for x in adj_lists[node] if x not in visited]
            if not neighbors:
                return path
            
            # Shuffle to randomize choice
            rng.shuffle(neighbors)
            
            best_subpath = path
            for next_node in neighbors:
                new_path = path + [next_node]
                visited.add(next_node)
                
                # Heuristic: If we want to maximize length, we just go deep.
                # But we can also try to prune if we are clearly not beating best_path?
                # For now, just go deep.
                
                subpath = dfs(next_node, new_path, visited)
                if len(subpath) > len(best_subpath):
                    best_subpath = subpath
                
                visited.remove(next_node)
                
                # Early termination if we found a full path? 
                # Max possible is 100. If we reach 100, we can stop.
                if len(best_subpath) == n:
                    return best_subpath
                    
            return best_subpath

        current_path = dfs(start, path, visited)
        
        if len(current_path) > len(best_path):
            best_path = current_path
            
            # If we found a path covering all nodes, we can stop immediately
            if len(best_path) == n:
                break
                
    return best_path
# EVOLVE-BLOCK-END
