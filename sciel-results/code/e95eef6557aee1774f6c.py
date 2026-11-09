"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict, deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Order vertices by degree descending
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    
    # Precompute neighbors for each vertex in the ordered list? No, just use adj.
    
    def is_valid_assignment(color, v, c, deadline):
        """
        Check if assigning color c to vertex v is valid.
        color: list of ints, -1 for uncolored.
        """
        # 1. Check properness with colored neighbors
        for u in adj[v]:
            if color[u] != -1 and color[u] == c:
                return False
        
        # 2. Check bicolored cycle constraint
        # For each distinct color d in neighbors (d != c), check if adding v
        # creates a cycle in the subgraph induced by {c, d}.
        # A cycle is created if any two neighbors of v with color d are connected
        # in the existing subgraph induced by {c, d}.
        
        # Get distinct colors d from neighbors
        neighbor_colors = set()
        for u in adj[v]:
            d = color[u]
            if d != -1 and d != c:
                neighbor_colors.add(d)
        
        for d in neighbor_colors:
            # Find all neighbors of v with color d
            neighbors_d = [w for w in adj[v] if color[w] == d]
            
            if len(neighbors_d) < 2:
                continue
            
            # Check if any two vertices in neighbors_d are in the same connected component
            # of the subgraph induced by {c, d}.
            
            # We perform BFS/DFS from each neighbor in neighbors_d that hasn't been visited yet.
            # If we reach another neighbor from neighbors_d, we found a connection.
            
            visited = set()
            
            for start in neighbors_d:
                if start in visited:
                    continue
                
                # BFS from start
                queue = deque([start])
                visited.add(start)
                
                while queue:
                    curr = queue.popleft()
                    for nxt in adj[curr]:
                        # Only traverse vertices with color c or d
                        if color[nxt] in (c, d):
                            # If we hit another neighbor from neighbors_d (and it's not the start itself),
                            # we have a path between two neighbors, which creates a cycle with v.
                            if nxt in neighbors_d and nxt != start:
                                return False
                            if nxt not in visited:
                                visited.add(nxt)
                                queue.append(nxt)
        
        return True

    def backtrack(idx, color, deadline):
        if time.time() > deadline:
            return None
        
        if idx == n:
            return color[:]
        
        v = order[idx]
        
        # Try each color
        for c in range(12): # Try up to 12 colors, though we expect < 10
            if is_valid_assignment(color, v, c, deadline):
                color[v] = c
                res = backtrack(idx + 1, color, deadline)
                if res is not None:
                    return res
                color[v] = -1
        
        return None

    deadline = time.time() + 80.0
    color = [-1] * n
    
    # Try increasing k
    # We can just run backtrack with a high max color and count distinct colors at the end?
    # No, backtrack tries 0..11. If it finds a solution using only 0..3, great.
    # But it might use 0..11. We want to minimize.
    # Better to try k=3, then k=4, etc.
    
    # However, the backtracking above tries colors 0..11 for every vertex.
    # This might find a solution using 5 colors, but we want to know if 4 is possible.
    # To minimize, we should restrict the color domain.
    
    for k in range(3, 13):
        if time.time() > deadline - 2:
            break
        
        # Reset color
        color = [-1] * n
        # We need a version of backtrack that only uses colors 0..k-1
        # and is_valid_assignment that only considers colors 0..k-1?
        # Actually, is_valid_assignment checks specific colors d. If d >= k, it won't appear.
        # So we just need to restrict the loop in backtrack to range(k).
        
        def backtrack_k(idx, color, deadline):
            if time.time() > deadline:
                return None
            
            if idx == n:
                return color[:]
            
            v = order[idx]
            
            for c in range(k):
                if is_valid_assignment(color, v, c, deadline):
                    color[v] = c
                    res = backtrack_k(idx + 1, color, deadline)
                    if res is not None:
                        return res
                    color[v] = -1
            
            return None

        res = backtrack_k(0, color, deadline)
        if res is not None:
            # Verify
            # The verification is implicit in the backtracking, but let's be safe.
            # Actually, the backtracking guarantees validity.
            return [int(x) for x in res]
            
    # Fallback
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
