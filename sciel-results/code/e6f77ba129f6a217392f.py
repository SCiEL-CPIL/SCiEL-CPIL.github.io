# EVOLVE-BLOCK-START
import sys

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Try 3-coloring with backtracking
    # If fails, return 4-coloring i % 4
    
    def solve_3_colors():
        coloring = [-1] * n
        
        # Order vertices by degree (all 4) or just 0..n
        # Use a simple heuristic: try to assign colors
        
        def is_valid(v, c):
            for w in adj[v]:
                if coloring[w] == c:
                    return False
            return True
            
        def check_bicolored(v, c):
            # Check if adding v with color c creates a bicolored cycle
            # We only need to check pairs involving c
            for other_c in range(3):
                if other_c == c:
                    continue
                # Check subgraph induced by {c, other_c}
                # If it has a cycle, return False
                # To speed up, we can just check if v is part of a cycle in this subgraph
                # Or simply run cycle detection on the current partial subgraph
                
                # Build set of vertices with color c or other_c
                vset = set()
                for i in range(n):
                    if coloring[i] == c or coloring[i] == other_c:
                        vset.add(i)
                        
                if len(vset) < 3:
                    continue
                    
                # Check for cycle in this subgraph
                # Use DFS
                visited = {v: False for v in vset}
                parent = {v: -1 for v in vset}
                
                def dfs(node, par):
                    visited[node] = True
                    for neighbor in adj[node]:
                        if neighbor not in vset:
                            continue
                        if not visited[neighbor]:
                            parent[neighbor] = node
                            if dfs(neighbor, node):
                                return True
                        elif neighbor != par:
                            return True
                    return False
                    
                for start in vset:
                    if not visited[start]:
                        if dfs(start, -1):
                            return False
            return True
            
        def backtrack(idx):
            if idx == n:
                return True
            for c in range(3):
                if not is_valid(idx, c):
                    continue
                coloring[idx] = c
                if check_bicolored(idx, c):
                    if backtrack(idx + 1):
                        return True
                coloring[idx] = -1
            return False
            
        if backtrack(0):
            return coloring
        return None
        
    result = solve_3_colors()
    if result is None:
        # Fallback to 4-coloring
        result = [i % 4 for i in range(n)]
        
    return [int(c) for c in result]
# EVOLVE-BLOCK-END
