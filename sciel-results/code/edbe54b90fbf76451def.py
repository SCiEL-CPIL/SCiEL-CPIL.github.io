"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
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
    
    deadline = time.time() + 80  # Safety margin
    
    # Try increasing number of colors
    for k in range(3, 51):
        if time.time() > deadline:
            break
        # Backtracking search
        color = [-1] * n
        # Use DSU-like tracking: for each pair (a,b), track connectivity
        # Actually, let's just check forest property directly
        
        def is_valid_assignment(color, v, c):
            """Check if assigning color c to vertex v is valid."""
            # Check adjacent vertices have different colors
            for u in adj[v]:
                if color[u] == c:
                    return False
            
            # Check bicolored forest constraint
            # For each other color d, the subgraph on {c, d} must be a forest
            for d in range(k):
                if d == c:
                    continue
                # Check if adding v (color c) to the {c,d} subgraph creates a cycle
                # v is connected to neighbors with color c or d
                # A cycle is created if two neighbors of v in the {c,d} subgraph are already connected
                neighbors_in_cd = []
                for u in adj[v]:
                    if color[u] == c or color[u] == d:
                        neighbors_in_cd.append(u)
                
                if len(neighbors_in_cd) >= 2:
                    # Check if any two neighbors in neighbors_in_cd are connected in the {c,d} subgraph
                    # Use BFS/DFS from each neighbor to see if it reaches another
                    for i in range(len(neighbors_in_cd)):
                        start = neighbors_in_cd[i]
                        # BFS in {c,d} subgraph
                        visited = set()
                        queue = [start]
                        visited.add(start)
                        while queue:
                            cur = queue.pop(0)
                            for w in adj[cur]:
                                if w == v:
                                    continue  # v not yet in subgraph
                                if color[w] == c or color[w] == d:
                                    if w not in visited:
                                        visited.add(w)
                                        queue.append(w)
                        for j in range(i + 1, len(neighbors_in_cd)):
                            if neighbors_in_cd[j] in visited:
                                return False  # Cycle would be created
            return True
        
        # Order vertices by degree (descending) for better pruning
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            
            v = order[idx]
            for c in range(k):
                if is_valid_assignment(color, v, c):
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        if backtrack(0):
            # Found valid coloring with k colors
            # Remap colors to 0..k-1 (they already are)
            result = [int(color[i]) for i in range(n)]
            return result
    
    # Fallback: use all distinct colors
    return list(range(n))
# EVOLVE-BLOCK-END
