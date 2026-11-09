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
        # For each neighbor u with color d != c, the pair (c, d) must remain a forest.
        # This is equivalent to: for any pair (a, b) where c is one of them,
        # check if adding v creates a cycle in the {a, b} subgraph.
        
        # We only need to check pairs involving c.
        # Let the other color be d.
        # The subgraph induced by colors {c, d} currently contains all colored vertices with color c or d.
        # We are adding v (color c) and edges (v, u) for u in adj[v] with color d (or c, but those are forbidden by properness).
        # Actually, if u has color c, it's forbidden. So u must have color d.
        # So we only add edges from v to neighbors with color d.
        
        # Let S_d = { u in adj[v] | color[u] == d }
        # A cycle is formed if there exist u1, u2 in S_d such that u1 and u2 are connected
        # in the subgraph induced by colors {c, d} among *previously colored* vertices.
        
        # We can optimize by grouping neighbors by their color.
        # But d varies.
        
        # For each distinct color d present in adj[v] (excluding -1 and c):
        for u in adj[v]:
            d = color[u]
            if d == -1 or d == c:
                continue
            
            # Check pair (c, d)
            # Find all neighbors of v with color d
            neighbors_d = [w for w in adj[v] if color[w] == d]
            
            if len(neighbors_d) < 2:
                continue
            
            # Check if any two neighbors in neighbors_d are connected in the {c, d} subgraph
            # The {c, d} subgraph consists of all vertices w such that color[w] in {c, d}.
            # We perform a BFS from neighbors_d[0] in this subgraph.
            # If we reach any other neighbor in neighbors_d, a cycle is formed.
            
            start = neighbors_d[0]
            target_set = set(neighbors_d)
            target_set.remove(start)
            
            visited = {start}
            queue = deque([start])
            
            while queue:
                curr = queue.popleft()
                for nxt in adj[curr]:
                    if color[nxt] in (c, d):
                        if nxt in target_set:
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
