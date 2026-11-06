import sys
import time

def construct():
        # Data from the prompt
        P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}
        
        n = P['n']
        edge_list = P['edge_order']
        
        # 1. Map edges to arrival index (1-based)
        edge_index = {}
        for i, (u, v) in enumerate(edge_list):
            if u > v:
                u, v = v, u
            edge_index[(u, v)] = i + 1
            
        # 2. Precompute adjacency lists sorted by edge weight? 
        #    Actually, for binary search, we just need to check connectivity in subgraph G_T.
        
        # 3. Binary Search for minimum T
        #    Lower bound: 24 (at least 24 edges needed, but max index is what matters)
        #    Upper bound: 276
        
        #    Helper: Check if Hamiltonian Cycle exists in G_T
        def has_hamiltonian_cycle(T):
            # Build adjacency list for edges with index <= T
            adj = [[] for _ in range(n)]
            for (u, v), idx in edge_index.items():
                if idx <= T:
                    adj[u].append(v)
                    adj[v].append(u)
                    
            # Check degrees
            for i in range(n):
                if len(adj[i]) < 2:
                    return False, None
                    
            # Check connectivity (simple BFS/DFS)
            visited = [False] * n
            stack = [0]
            visited[0] = True
            count = 1
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        count += 1
                        stack.append(v)
            if count != n:
                return False, None
                
            # Backtracking to find Hamiltonian Cycle
            # Start at 0
            path = [0]
            visited_bt = [False] * n
            visited_bt[0] = True
            
            # Heuristic: Order neighbors by degree (ascending) to fail fast
            # Pre-sort adjacency lists by degree of the neighbor?
            # Degrees are fixed for this T.
            deg = [len(adj[i]) for i in range(n)]
            
            def backtrack(u):
                if len(path) == n:
                    # Check if last vertex connects to 0
                    if 0 in adj[u]:
                        return True
                    return False
                    
                # Get neighbors of u that are not visited
                # Sort by degree to try constrained nodes first
                neighbors = [v for v in adj[u] if not visited_bt[v]]
                neighbors.sort(key=lambda x: deg[x])
                
                for v in neighbors:
                    # Pruning: Check if unvisited nodes (excluding v) are still connected 
                    # and have valid degrees in the remaining subgraph?
                    # Simple pruning: If any unvisited node (that is not v) has no neighbors 
                    # in the set of unvisited nodes (including v if it was the only one) 
                    # plus current u? 
                    # Actually, a simpler strong pruning:
                    # If we pick v, does it isolate any other unvisited node?
                    # Let's just do the degree check for the remaining graph.
                    
                    visited_bt[v] = True
                    path.append(v)
                    
                    if backtrack(v):
                        return True
                        
                    path.pop()
                    visited_bt[v] = False
                    
                return False

            if backtrack(0):
                return True, path
            else:
                return False, None

        low, high = 1, len(edge_list)
        best_path = list(range(n)) # Fallback to identity
        
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 85.0
        
        while low <= high:
            if time.time() > deadline - 5.0: # 5s safety margin
                break
                
            mid = (low + high) // 2
            exists, path = has_hamiltonian_cycle(mid)
            
            if exists:
                best_path = path
                high = mid - 1
            else:
                low = mid + 1
                
        # Calculate score of best_path
        # The score returned by construct() is just the path.
        # The evaluator calculates the score.
        # We want to return the path that minimizes the max index.
        
        return best_path
