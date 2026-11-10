import time
import random

# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def check_bicolored_cycle(colors, c1, c2, edges_subset):
        """Check if edges between vertices of color c1 and c2 form a cycle using union-find."""
        # Find all vertices with color c1 or c2
        vertices = set()
        for i in range(n):
            if colors[i] == c1 or colors[i] == c2:
                vertices.add(i)
        
        if len(vertices) < 3:
            return False
        
        parent = {v: v for v in vertices}
        
        def find(x):
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
        
        for u, v in edges_subset:
            if u in vertices and v in vertices:
                if colors[u] != colors[v]:  # Only count edges between different colors
                    if not union(u, v):
                        return True  # Cycle found
        
        return False
    
    def is_valid_coloring(colors):
        """Check proper coloring and bicolored-forest constraint."""
        # Check proper coloring
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        
        # Check bicolored forest constraint
        distinct_colors = sorted(set(colors))
        for i in range(len(distinct_colors)):
            for j in range(i + 1, len(distinct_colors)):
                c1, c2 = distinct_colors[i], distinct_colors[j]
                if check_bicolored_cycle(colors, c1, c2, edges):
                    return False
        return True
    
    # Try 3 colors with backtracking and pruning
    # Order vertices by degree (all are 4, so use a heuristic order)
    # Use a simple order that might work well for this structured graph
    
    # First try deterministic patterns
    patterns_3 = [
        lambda i: i % 3,
        lambda i: (i * 2) % 3,
        lambda i: (i + i // 7) % 3,
        lambda i: (i + i // 10) % 3,
        lambda i: (i * 7) % 3,
        lambda i: (i + i // 5) % 3,
    ]
    
    for p in patterns_3:
        colors = [p(i) for i in range(n)]
        if is_valid_coloring(colors):
            return [int(c) for c in colors]
    
    # Backtracking search for 3 colors
    # Vertex order: try to use a good ordering
    # Since all degrees are 4, use a greedy heuristic ordering
    
    def backtrack_3colors(order, colors, idx):
        if time.time() > deadline:
            return None
        
        if idx == len(order):
            if is_valid_coloring(colors):
                return colors[:]
            return None
        
        v = order[idx]
        
        # Determine forbidden colors from neighbors
        forbidden = set()
        for nb in adj[v]:
            if colors[nb] != -1:
                forbidden.add(colors[nb])
        
        for c in range(3):
            if c in forbidden:
                continue
            
            # Try assigning color c to v
            colors[v] = c
            
            # Check if this creates a bicolored cycle with any existing color
            valid = True
            for c2 in range(3):
                if c2 == c:
                    continue
                # Check if adding v with color c creates a cycle in the {c, c2} subgraph
                # We need to check if there's a path between two neighbors of v that are color c2
                # through the existing {c, c2} subgraph
                
                # More efficient: check union-find on the induced subgraph
                # Find all vertices with color c or c2
                vertices = set()
                for i in range(n):
                    if colors[i] == c or colors[i] == c2:
                        vertices.add(i)
                
                if len(vertices) < 3:
                    continue
                
                parent = {x: x for x in vertices}
                
                def find(x):
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
                
                # Process edges in the induced subgraph
                for u, w in edges:
                    if u in vertices and w in vertices:
                        if colors[u] != colors[w]:
                            if not union(u, w):
                                valid = False
                                break
                if not valid:
                    break
            
            if valid:
                result = backtrack_3colors(order, colors, idx + 1)
                if result is not None:
                    return result
            
            colors[v] = -1
        
        return None
    
    # Try backtracking with different orderings
    orderings = [
        list(range(n)),
        list(range(n - 1, -1, -1)),
        sorted(range(n), key=lambda x: -len(adj[x])),  # All same degree
    ]
    
    for order in orderings:
        if time.time() > deadline - 5:
            break
        
        colors = [-1] * n
        result = backtrack_3colors(order, colors, 0)
        if result is not None:
            return [int(c) for c in result]
    
    # If 3 colors fail, try 4 colors
    # For 4 colors, use a smarter approach: greedy with restarts
    
    for attempt in range(5000):
        if time.time() > deadline - 5:
            break
        
        order = list(range(n))
        random.shuffle(order)
        
        colors = [-1] * n
        valid = True
        for v in order:
            forbidden = set()
            for nb in adj[v]:
                if colors[nb] != -1:
                    forbidden.add(colors[nb])
            
            available = [c for c in range(4) if c not in forbidden]
            if not available:
                valid = False
                break
            
            colors[v] = random.choice(available)
        
        if valid and is_valid_coloring(colors):
            return [int(c) for c in colors]
    
    # Fallback: 5 colors
    for attempt in range(2000):
        if time.time() > deadline - 2:
            break
        
        order = list(range(n))
        random.shuffle(order)
        
        colors = [-1] * n
        valid = True
        for v in order:
            forbidden = set()
            for nb in adj[v]:
                if colors[nb] != -1:
                    forbidden.add(colors[nb])
            
            available = [c for c in range(5) if c not in forbidden]
            if not available:
                valid = False
                break
            
            colors[v] = random.choice(available)
        
        if valid and is_valid_coloring(colors):
            return [int(c) for c in colors]
    
    # Final fallback: unique colors (always valid)
    return list(range(n))
# EVOLVE-BLOCK-END
