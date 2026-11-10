import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Convert to list of lists for faster access
    adj_list = [adj[i] for i in range(n)]
    
    # Edge list as tuples for faster iteration
    edge_list = [(u, v) for u, v in edges]
    
    def check_validity(coloring):
        """Check if coloring is valid: proper and no bicolored cycles"""
        # Check proper coloring
        for u, v in edge_list:
            if coloring[u] == coloring[v]:
                return False
        
        # Check no bicolored cycles using union-find per color pair
        # For each pair of colors, check if the induced subgraph is a forest
        colors_used = set(coloring)
        color_pairs = set()
        for c in colors_used:
            for d in colors_used:
                if c < d:
                    color_pairs.add((c, d))
        
        for c1, c2 in color_pairs:
            # Build subgraph with only vertices colored c1 or c2
            # and edges between them
            parent = list(range(n))
            rank = [0] * n
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return True  # cycle detected
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return False
            
            for u, v in edge_list:
                cu, cv = coloring[u], coloring[v]
                if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                    if union(u, v):
                        return False  # bicolored cycle found
        
        return True
    
    def count_distinct_colors(coloring):
        return len(set(coloring))
    
    # Try to find a valid coloring with few colors
    # Start with 3 colors, if that fails try 4, etc.
    
    best_coloring = None
    best_num_colors = n + 1
    
    # Try 3-coloring first
    for num_colors in [3, 4, 5]:
        if time.time() > deadline:
            break
        
        found = False
        # Multiple random restarts
        max_restarts = 50 if num_colors == 3 else 100
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initial coloring
            coloring = [random.randint(0, num_colors - 1) for _ in range(n)]
            
            # Local search: try to fix invalidities
            for iteration in range(10000):
                if time.time() > deadline:
                    break
                
                # Check if valid
                if check_validity(coloring):
                    num_used = count_distinct_colors(coloring)
                    if num_used < best_num_colors:
                        best_num_colors = num_used
                        best_coloring = coloring[:]
                        found = True
                        break
                
                # Find a vertex to recolor
                # Priority: vertices involved in violations
                # Simple approach: pick a random vertex and try all colors
                v = random.randint(0, n - 1)
                old_color = coloring[v]
                
                # Try each color
                best_new_color = old_color
                best_improvement = 0
                
                for new_color in range(num_colors):
                    if new_color == old_color:
                        continue
                    
                    # Check if this creates a proper coloring violation
                    valid_move = True
                    for u in adj_list[v]:
                        if coloring[u] == new_color:
                            valid_move = False
                            break
                    
                    if not valid_move:
                        continue
                    
                    # Try the move and check if it reduces violations
                    # For simplicity, just accept if it doesn't create obvious problems
                    # This is a simplified local search
                    if random.random() < 0.3:  # explore
                        best_new_color = new_color
                        best_improvement = 1
                        break
                
                if best_improvement > 0:
                    coloring[v] = best_new_color
        
        if found:
            break
    
    # If no valid coloring found, fall back to a safe coloring
    if best_coloring is None:
        # Use a greedy coloring with many colors to ensure validity
        # Simple: use vertex index as color, but that's 50 colors
        # Better: try a systematic approach
        coloring = list(range(n))
        # This is always valid (proper, and any two colors induce at most one edge, so forest)
        best_coloring = coloring
        best_num_colors = n
    
    # Ensure we return Python ints in a list
    result = [int(c) for c in best_coloring]
    return result
# EVOLVE-BLOCK-END
