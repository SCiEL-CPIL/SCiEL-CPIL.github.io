"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

# Build adjacency
adj = [[] for _ in range(N)]
for u, v in EDGES:
    adj[u].append(v)
    adj[v].append(u)

def check_bicolored_forest(colors, max_colors):
    """Check that for every pair of colors (a,b), the induced subgraph is a forest."""
    # For each pair of colors, check if the subgraph induced by vertices with those two colors is a forest
    # A graph is a forest iff it has no cycles, i.e., edges <= vertices - components
    # More efficiently: for each pair, do a DFS/BFS to detect cycles
    color_to_verts = {}
    for i, c in enumerate(colors):
        if c < max_colors:
            if c not in color_to_verts:
                color_to_verts[c] = []
            color_to_verts[c].append(i)
    
    colors_used = sorted(color_to_verts.keys())
    
    for idx_a, a in enumerate(colors_used):
        for b in colors_used[idx_a+1:]:
            # Get vertices with color a or b
            verts_a = color_to_verts[a]
            verts_b = color_to_verts[b]
            verts = set(verts_a) | set(verts_b)
            
            # Build subgraph
            # Count edges in the subgraph
            edge_count = 0
            for u in verts:
                for v in adj[u]:
                    if v in verts and u < v:
                        # Check that both u and v have color a or b
                        if colors[u] in (a, b) and colors[v] in (a, b):
                            edge_count += 1
            
            # Count components
            visited = set()
            components = 0
            for start in verts:
                if start not in visited:
                    components += 1
                    stack = [start]
                    visited.add(start)
                    while stack:
                        node = stack.pop()
                        for nb in adj[node]:
                            if nb in verts and nb not in visited:
                                visited.add(nb)
                                stack.append(nb)
            
            # Forest condition: edges <= vertices - components
            if edge_count > len(verts) - components:
                return False
    return True

def count_distinct(colors):
    return len(set(colors))

def valid_coloring(colors):
    """Check all constraints: adjacent differ, and bicolored subgraphs are forests."""
    # Check adjacent vertices differ
    for u, v in EDGES:
        if colors[u] == colors[v]:
            return False
    # Check bicolored forests
    return check_bicolored_forest(colors, max(colors) + 1)

def greedy_coloring(seed=None, max_colors=4):
    """Greedy coloring with random order."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    order = list(range(N))
    rng.shuffle(order)
    
    colors = [-1] * N
    for v in order:
        # Find forbidden colors
        forbidden = set()
        for nb in adj[v]:
            if colors[nb] != -1:
                forbidden.add(colors[nb])
        
        # Try to assign a valid color
        assigned = False
        for c in range(max_colors):
            if c in forbidden:
                continue
            colors[v] = c
            # Check if this assignment creates a bicolored cycle
            if valid_coloring_partial(colors, v, c):
                assigned = True
                break
            else:
                colors[v] = -1
        
        if not assigned:
            return None
    return colors

def valid_coloring_partial(colors, v, c):
    """Check if assigning color c to vertex v (with all previous assignments valid) maintains validity."""
    # We need to check that for each color pair involving c, the subgraph is still a forest
    # This is expensive; for now, do a full check but only for pairs involving c
    colors[v] = c
    # Check pairs involving color c
    colors_used = set(colors[i] for i in range(N) if colors[i] != -1)
    for other in colors_used:
        if other == c:
            continue
        # Check subgraph induced by colors c and other
        verts = set(i for i in range(N) if colors[i] in (c, other))
        edge_count = 0
        for u in verts:
            for w in adj[u]:
                if w in verts and u < w and colors[w] in (c, other):
                    edge_count += 1
        visited = set()
        components = 0
        for start in verts:
            if start not in visited:
                components += 1
                stack = [start]
                visited.add(start)
                while stack:
                    node = stack.pop()
                    for nb in adj[node]:
                        if nb in verts and nb not in visited:
                            visited.add(nb)
                            stack.append(nb)
        if edge_count > len(verts) - components:
            colors[v] = -1
            return False
    # Also check adjacent vertices differ
    for nb in adj[v]:
        if colors[nb] == c:
            colors[v] = -1
            return False
    return True

def local_search_eliminate(colors, max_colors, time_limit=30):
    """Try to eliminate colors one by one."""
    start_time = time.time()
    current = list(colors)
    current_max = max(current) + 1
    
    # Try to eliminate colors from highest to lowest
    for target_color in range(current_max - 1, -1, -1):
        if time.time() - start_time > time_limit:
            break
        # Check if target_color is actually used
        if target_color not in set(current):
            continue
        
        # Find all vertices with this color
        verts_to_recolor = [i for i in range(N) if current[i] == target_color]
        if not verts_to_recolor:
            continue
        
        # Try to recolor each vertex to a lower color
        success = True
        new_current = list(current)
        for v in verts_to_recolor:
            recolored = False
            for c in range(target_color):
                new_current[v] = c
                if valid_coloring(new_current):
                    recolored = True
                    break
            if not recolored:
                success = False
                break
        
        if success:
            current = new_current
            current_max = target_color
        else:
            # Try random reordering of vertices to recolor
            rng = random.Random()
            rng.shuffle(verts_to_recolor)
            new_current = list(current)
            success = True
            for v in verts_to_recolor:
                recolored = False
                for c in range(target_color):
                    new_current[v] = c
                    if valid_coloring(new_current):
                        recolored = True
                        break
                if not recolored:
                    success = False
                    break
            if success:
                current = new_current
                current_max = target_color
    
    return current, current_max

def try_coloring_with_k_colors(k, seed, time_per_try=2):
    """Try to find a valid coloring with at most k colors."""
    start_time = time.time()
    rng = random.Random(seed)
    
    # Greedy with random order
    order = list(range(N))
    rng.shuffle(order)
    
    colors = [-1] * N
    for v in order:
        forbidden = set()
        for nb in adj[v]:
            if colors[nb] != -1:
                forbidden.add(colors[nb])
        
        assigned = False
        # Try colors in random order
        color_order = list(range(k))
        rng.shuffle(color_order)
        for c in color_order:
            if c in forbidden:
                continue
            colors[v] = c
            if valid_coloring_partial(colors, v, c):
                assigned = True
                break
            else:
                colors[v] = -1
        
        if not assigned:
            return None
    
    if valid_coloring(colors):
        return colors
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    best_colors = list(range(N))  # Fallback: 50 colors
    best_count = N
    
    # First, try to find a good coloring with greedy + local search
    # Try different numbers of colors from 3 up
    for k in [3, 4, 5, 6, 7, 8]:
        if time.time() > deadline - 5:
            break
        
        found = False
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            colors = try_coloring_with_k_colors(k, seed=42 + attempt * 1000 + k)
            if colors is not None:
                count = count_distinct(colors)
                if count < best_count:
                    best_count = count
                    best_colors = list(colors)
                
                # Try local search to eliminate more colors
                improved, improved_max = local_search_eliminate(colors, k, time_limit=2)
                improved_count = count_distinct(improved)
                if improved_count < best_count:
                    best_count = improved_count
                    best_colors = list(improved)
                
                found = True
                if best_count <= 3:
                    break
        
        if best_count <= 3:
            break
    
    # If we haven't found anything better, try a more systematic approach
    if best_count > 4 and time.time() < deadline - 10:
        # Try structured coloring based on graph structure
        # The graph has a path 0-1-2-...-42, spine 0-7-8-...-36, ring 36-43-44-...-49-0
        # and cross edges (i, i+7) for i=1..35
        
        # Try 4-coloring with a pattern
        # Let's try: color based on position mod 4 with some adjustments
        for pattern_seed in range(50):
            if time.time() > deadline - 3:
                break
            rng = random.Random(pattern_seed)
            
            # Start with a 4-coloring using a base pattern
            base = [i % 4 for i in range(N)]
            
            # Try to fix conflicts
            colors = list(base)
            improved = True
            while improved and time.time() < deadline - 3:
                improved = False
                for v in range(N):
                    # Check if v's color conflicts
                    for nb in adj[v]:
                        if colors[nb] == colors[v]:
                            # Try to change v's color
                            for c in range(4):
                                if c != colors[v]:
                                    old = colors[v]
                                    colors[v] = c
                                    if valid_coloring(colors):
                                        improved = True
                                        break
                                    else:
                                        colors[v] = old
                            break
                
                count = count_distinct(colors)
                if count < best_count:
                    best_count = count
                    best_colors = list(colors)
            
            if best_count <= 3:
                break
    
    # Final verification
    if valid_coloring(best_colors):
        result = [int(c) for c in best_colors]
        return result
    else:
        # Fallback: return 50 distinct colors
        return list(range(N))

# EVOLVE-BLOCK-END
