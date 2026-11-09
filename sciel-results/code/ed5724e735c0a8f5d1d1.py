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
    
    deadline = time.time() + 75.0
    
    def solve_k(k):
        """Try to find a valid coloring with exactly k colors (0 to k-1)."""
        color = [-1] * n
        
        def is_valid(v, c):
            """Check if assigning color c to vertex v is valid."""
            # Check properness
            for u in adj[v]:
                if color[u] == c:
                    return False
            
            # Check bicolored cycle constraint
            neighbor_colors = set()
            for u in adj[v]:
                if color[u] != -1:
                    neighbor_colors.add(color[u])
            
            for d in neighbor_colors:
                if d == c:
                    continue
                
                # Find neighbors of v with color d
                neighbors_d = [u for u in adj[v] if color[u] == d]
                if len(neighbors_d) < 2:
                    continue
                
                # Check if any two neighbors in neighbors_d are connected
                # in the {c, d} subgraph
                start = neighbors_d[0]
                visited = {start}
                queue = deque([start])
                found_cycle = False
                
                while queue and not found_cycle:
                    curr = queue.popleft()
                    for nxt in adj[curr]:
                        if color[nxt] in (c, d):
                            if nxt in neighbors_d[1:]:
                                found_cycle = True
                                break
                            if nxt not in visited:
                                visited.add(nxt)
                                queue.append(nxt)
                
                if found_cycle:
                    return False
            
            return True
        
        # MRV heuristic: select the unassigned vertex with the fewest available colors
        def select_vertex():
            """Select the unassigned vertex with the fewest available colors."""
            best_v = -1
            best_count = k + 1
            
            for v in range(n):
                if color[v] != -1:
                    continue
                
                # Count available colors (colors not used by any neighbor)
                used_colors = set()
                for u in adj[v]:
                    if color[u] != -1:
                        used_colors.add(color[u])
                
                available = k - len(used_colors)
                
                if available < best_count:
                    best_count = available
                    best_v = v
                    if best_count <= 0:
                        break
            
            return best_v
        
        def backtrack():
            """Backtracking search with k colors using MRV."""
            if time.time() > deadline:
                return None
            
            v = select_vertex()
            if v == -1:
                # All vertices assigned
                return color[:]
            
            # Try each color
            for c in range(k):
                if is_valid(v, c):
                    color[v] = c
                    res = backtrack()
                    if res is not None:
                        return res
                    color[v] = -1
            
            return None
        
        return backtrack()
    
    # Try increasing k from 3 upward
    for k in range(3, 15):
        if time.time() > deadline - 5.0:
            break
        
        res = solve_k(k)
        if res is not None:
            # Verify the solution
            valid = True
            # Check properness
            for u, v in edges:
                if res[u] == res[v]:
                    valid = False
                    break
            
            if valid:
                # Check bicolored cycles
                for c1 in range(k):
                    if not valid:
                        break
                    for c2 in range(c1 + 1, k):
                        # Check if the subgraph induced by colors c1 and c2 is a forest
                        vertices_in_subgraph = set()
                        for i in range(n):
                            if res[i] in (c1, c2):
                                vertices_in_subgraph.add(i)
                        
                        if len(vertices_in_subgraph) <= 1:
                            continue
                        
                        # DFS to detect cycle
                        visited = set()
                        has_cycle = False
                        for start in vertices_in_subgraph:
                            if start in visited:
                                continue
                            stack = [(start, -1)]
                            visited.add(start)
                            while stack and not has_cycle:
                                curr, parent = stack.pop()
                                for nxt in adj[curr]:
                                    if res[nxt] in (c1, c2):
                                        if nxt == parent:
                                            continue
                                        if nxt in visited:
                                            has_cycle = True
                                            break
                                        visited.add(nxt)
                                        stack.append((nxt, curr))
                            if has_cycle:
                                break
                        if has_cycle:
                            valid = False
                            break
            
            if valid:
                return [int(x) for x in res]
    
    # Fallback: use all distinct colors
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
