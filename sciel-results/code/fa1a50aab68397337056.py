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
        
    # Sort vertices by degree descending for initial ordering
    # But we will use MRV in backtracking, so we just need a list of vertices
    vertices = list(range(n))
    
    # DSU for cycle detection
    class DSU:
        def __init__(self, size):
            self.parent = list(range(size))
            self.rank = [0] * size
        
        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x
        
        def union(self, x, y):
            px, py = self.find(x), self.find(y)
            if px == py:
                return False
            if self.rank[px] < self.rank[py]:
                px, py = py, px
            self.parent[py] = px
            if self.rank[px] == self.rank[py]:
                self.rank[px] += 1
            return True

    def check_bicolored_cycle(coloring, c1, c2):
        """Check if subgraph induced by colors c1 and c2 has a cycle."""
        vertices_in_subgraph = [v for v in range(n) if coloring[v] in (c1, c2)]
        if len(vertices_in_subgraph) < 3:
            return False
        
        vset = set(vertices_in_subgraph)
        local_map = {v: i for i, v in enumerate(vertices_in_subgraph)}
        local_n = len(vertices_in_subgraph)
        
        dsu = DSU(local_n)
        
        for v in vertices_in_subgraph:
            for w in adj[v]:
                if w in vset:
                    lv, lw = local_map[v], local_map[w]
                    if lv < lw: # Avoid duplicate checks
                        if not dsu.union(lv, lw):
                            return True # Cycle found
        return False

    def solve(num_colors):
        coloring = [-1] * n
        
        def is_valid(v, c):
            # Check neighbors
            for w in adj[v]:
                if coloring[w] == c:
                    return False
            return True

        def check_constraint(v, c):
            # Check if adding v with color c creates a bicolored cycle
            # Only need to check colors present in neighbors
            neighbor_colors = set()
            for w in adj[v]:
                if coloring[w] != -1:
                    neighbor_colors.add(coloring[w])
            
            for other_c in neighbor_colors:
                if other_c == c:
                    continue
                # Check subgraph of c and other_c
                if check_bicolored_cycle(coloring, c, other_c):
                    return False
            return True

        def backtrack(idx):
            if idx == n:
                return True
            
            v = vertices[idx]
            
            # MRV: Find available colors
            available_colors = []
            for c in range(num_colors):
                if is_valid(v, c):
                    available_colors.append(c)
            
            if not available_colors:
                return False
            
            # Try each available color
            for c in available_colors:
                coloring[v] = c
                if check_constraint(v, c):
                    if backtrack(idx + 1):
                        return True
                coloring[v] = -1
                
            return False
        
        # Heuristic: Process vertices in order of degree descending
        # This helps in finding conflicts early
        ordered_vertices = sorted(vertices, key=lambda x: len(adj[x]), reverse=True)
        
        # We need to map the "idx" in backtrack to the actual vertex
        # So we modify backtrack to take a list of unassigned vertices or just use the sorted list
        
        def backtrack_sorted(order_idx):
            if order_idx == n:
                return True
            
            v = ordered_vertices[order_idx]
            
            available_colors = []
            for c in range(num_colors):
                if is_valid(v, c):
                    available_colors.append(c)
            
            if not available_colors:
                return False
            
            for c in available_colors:
                coloring[v] = c
                if check_constraint(v, c):
                    if backtrack_sorted(order_idx + 1):
                        return True
                coloring[v] = -1
                
            return False

        if backtrack_sorted(0):
            return coloring
        return None

    # Try 3 colors first
    result = solve(3)
    if result:
        return [int(c) for c in result]
    
    # Fallback to 4 colors
    result = solve(4)
    if result:
        return [int(c) for c in result]
        
    # Fallback to distinct colors if all else fails (should not happen)
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
