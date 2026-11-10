"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    edges = P['edges']
    n = P['n']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute edge list for fast checking
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    def count_distinct_colors(colors):
        return len(set(colors))
    
    def is_valid_coloring(colors):
        """Check if coloring is valid: adjacent vertices differ and no bicolored cycles."""
        # Check adjacent vertices differ
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        
        # Check no bicolored cycles using union-find
        # For each pair of colors (c1, c2), check if the subgraph induced by vertices
        # with colors c1 or c2 is a forest
        color_pairs = set()
        for u, v in edges:
            c1, c2 = colors[u], colors[v]
            if c1 != c2:
                pair = (min(c1, c2), max(c1, c2))
                color_pairs.add(pair)
        
        for c1, c2 in color_pairs:
            # Build union-find for vertices with color c1 or c2
            parent = {}
            def find(x):
                if x not in parent:
                    parent[x] = x
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False
                parent[rx] = ry
                return True
            
            for u, v in edges:
                cu, cv = colors[u], colors[v]
                if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                    if not union(u, v):
                        return False
        return True
    
    # Try to find a valid coloring with few colors
    # Start with greedy coloring, then try to reduce colors
    
    def greedy_coloring(max_colors):
        """Greedy coloring using at most max_colors colors."""
        colors = [-1] * n
        for v in range(n):
            used = set()
            for u in adj[v]:
                if colors[u] != -1:
                    used.add(colors[u])
            for c in range(max_colors):
                if c not in used:
                    colors[v] = c
                    break
        return colors
    
    # First, get a valid coloring with more colors
    best_colors = None
    best_count = n + 1
    
    # Try greedy with increasing number of colors until valid
    for max_c in range(2, n + 1):
        if time.time() > deadline:
            break
        colors = greedy_coloring(max_c)
        if colors is not None and all(c != -1 for c in colors):
            if is_valid_coloring(colors):
                best_colors = colors
                best_count = count_distinct_colors(colors)
                break
    
    if best_colors is None:
        # Fallback: use all distinct colors
        return list(range(n))
    
    # Now try to reduce the number of colors
    # Local search: try to recolor vertices to reduce distinct colors
    current_colors = best_colors[:]
    current_count = best_count
    
    # Build list of colors used
    def get_used_colors(colors):
        return sorted(set(colors))
    
    used_colors = get_used_colors(current_colors)
    
    # Local search: try to eliminate one color at a time
    improved = True
    while improved and time.time() < deadline:
        improved = False
        used_colors = get_used_colors(current_colors)
        
        # Try to eliminate each color
        for target_color in used_colors[:]:
            if len(used_colors) <= 1:
                break
            # Try to recolor all vertices with target_color to other colors
            vertices_with_target = [v for v in range(n) if current_colors[v] == target_color]
            
            # Try different target colors to recolor to
            other_colors = [c for c in used_colors if c != target_color]
            
            for new_color in other_colors:
                # Try to recolor all vertices with target_color to new_color
                trial_colors = current_colors[:]
                valid = True
                for v in vertices_with_target:
                    # Check if new_color is valid for v
                    for u in adj[v]:
                        if trial_colors[u] == new_color:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Apply the change
                    for v in vertices_with_target:
                        trial_colors[v] = new_color
                    
                    # Check if the new coloring is valid (no bicolored cycles)
                    if is_valid_coloring(trial_colors):
                        new_count = count_distinct_colors(trial_colors)
                        if new_count < current_count:
                            current_colors = trial_colors
                            current_count = new_count
                            improved = True
                            break
        
        if not improved:
            # Try pairwise swaps or other moves
            # Try recoloring a single vertex to a different color
            for v in range(n):
                if time.time() > deadline:
                    break
                old_color = current_colors[v]
                for new_color in range(old_color + 1, max(used_colors) + 1):
                    if new_color == old_color:
                        continue
                    trial_colors = current_colors[:]
                    trial_colors[v] = new_color
                    
                    # Check adjacent vertices differ
                    valid = True
                    for u in adj[v]:
                        if trial_colors[u] == new_color:
                            valid = False
                            break
                    
                    if valid and is_valid_coloring(trial_colors):
                        new_count = count_distinct_colors(trial_colors)
                        if new_count < current_count:
                            current_colors = trial_colors
                            current_count = new_count
                            improved = True
    
    # Ensure we return Python ints
    return [int(c) for c in current_colors]

# EVOLVE-BLOCK-END
