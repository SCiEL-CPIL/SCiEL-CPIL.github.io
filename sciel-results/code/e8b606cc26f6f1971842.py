import random
import sys

def construct():
    """
    Greedy randomized construction for 5-colourings of K_n without monochromatic C4.
    We try to maximize n up to 29.
    """
    max_n = 29
    num_colors = 5
    
    # Try to build from largest n down
    for n in range(max_n, 10, -1):
        if _build_coloring(n, num_colors):
            # If we can build a valid coloring, return it
            pass  # The function _build_coloring returns the colors if successful
    
    # Fallback to known construction for n=11
    n = 11
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}

def _build_coloring(n, num_colors, max_attempts=100):
    """
    Attempt to build a valid 5-coloring of K_n without monochromatic C4.
    Returns True if successful (and sets global result), else False.
    """
    import random
    import time
    
    start_time = time.time()
    timeout = 115  # Leave buffer for 120s limit
    
    for attempt in range(max_attempts):
        if time.time() - start_time > timeout:
            break
            
        # Initialize data structures
        # colors[i][j] = color of edge (i,j), -1 if unassigned
        colors = [[-1] * n for _ in range(n)]
        
        # For each color c and each pair of vertices (u,v), track count of common neighbors in color c
        # common_neighbors[c][u][v] = number of vertices w such that edge (u,w) and (v,w) both have color c
        # To avoid C4: for any color c, no two vertices u,v should have 2 common neighbors in color c
        # So we track, for each color c and each unordered pair (u,v) with u!=v, the count of common neighbors
        # If this count reaches 2, that color is forbidden for future edges that would create a third common neighbor?
        # Actually, C4 exists if there are two vertices u,v with two common neighbors in color c.
        # So if common_neighbors[c][u][v] >= 2, then a C4 already exists (or will exist).
        # We need to prevent this.
        
        # More precisely: when we assign color c to edge (u,v), we need to check that for every other vertex w,
        # if edge (w, x) is already color c for some x, we might create a C4.
        # 
        # Better approach: For each color c, maintain for each vertex v a set of neighbors in color c.
        # When assigning color c to edge (u,v), we check:
        # For each common neighbor w of u and v in color c (i.e., both (u,w) and (v,w) are color c),
        # if there is already one such w, then adding another would create a C4.
        # Actually, we need to ensure that after assignment, no pair of vertices has 2+ common neighbors in color c.
        
        # Let's use: for each color c, for each pair (a,b), count_c[a][b] = number of common neighbors in color c
        # Initialize all to 0.
        # When we assign color c to edge (u,v):
        #   For each vertex w:
        #     if colors[u][w] == c and colors[v][w] == c: 
        #        this means w is a common neighbor of u and v in color c
        #        We need to check if count_c[u][v] is already >= 1 (meaning there's already a common neighbor)
        #        If count_c[u][v] >= 1, then assigning this edge would make count_c[u][v] >= 2, creating a C4.
        #        So this color is forbidden.
        #   Also, for each vertex w where colors[u][w] == c but colors[v][w] != c (or unassigned):
        #     After assigning (u,v) to c, w becomes a common neighbor of v and some other vertices?
        #     No, let's think differently.
        #
        # When we assign color c to edge (u,v), the new common neighbors relationships are:
        # For any vertex w:
        #   If colors[u][w] == c and colors[v][w] == c, then w is a common neighbor of u and v in color c.
        #   The number of such w is count_c[u][v].
        #   We need count_c[u][v] < 2 for all pairs.
        #
        # But also, assigning (u,v) to color c affects counts for other pairs:
        # For any vertex w with colors[u][w] == c (w != v):
        #   Now v and w both have a common neighbor u in color c? No.
        #   u is connected to v and w in color c. So for pair (v,w), u is a common neighbor in color c.
        #   So count_c[v][w] increases by 1.
        #   We need to ensure count_c[v][w] doesn't reach 2.
        # Similarly for w with colors[v][w] == c (w != u):
        #   count_c[u][w] increases by 1.
        
        # This is getting complex. Let me use a simpler check:
        # For each color c, maintain adjacency lists.
        # When assigning color c to (u,v):
        #   For each w in adj_c[u] (vertices adjacent to u in color c):
        #       if w is also in adj_c[v], then w is a common neighbor of u and v in color c.
        #       If there's already at least one such common neighbor, then adding (u,v) would create a C4.
        #   More precisely: let common = adj_c[u] ∩ adj_c[v]. If |common| >= 1, then C4 exists.
        #   Wait, if |common| >= 2, C4 exists. If |common| == 1, no C4 yet from this pair, but...
        #   Actually, C4 in color c means there exist distinct vertices a,b,c,d such that edges (a,b), (b,c), (c,d), (d,a) all in color c.
        #   This is equivalent to: there exist two vertices with two common neighbors in color c.
        #   So for pair (u,v), if |adj_c[u] ∩ adj_c[v]| >= 2, C4 exists.
        #   When we add edge (u,v) to color c, we add v to adj_c[u] and u to adj_c[v].
        #   This can create new common neighbors for other pairs.
        
        # Let me just check: after tentatively assigning color c to (u,v), does any pair have 2+ common neighbors in color c?
        # That's expensive. Let me use the count approach with updates.
        
        # Simpler: just check the direct condition.
        # For color c, after assigning (u,v) to c:
        #   For pair (u,v): common neighbors in c are vertices w with colors[u][w]==c and colors[v][w]==c.
        #   If count >= 2, invalid.
        #   For each w with colors[u][w]==c: pair (v,w) gains u as a common neighbor in c (since colors[u][v]==c and colors[u][w]==c).
        #     Check if (v,w) now has 2+ common neighbors in c.
        #   For each w with colors[v][w]==c: pair (u,w) gains v as a common neighbor in c.
        #     Check if (u,w) now has 2+ common neighbors in c.
        
        # Let me implement this.
        
        # Precompute edge list
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        # Shuffle edges with a seed for reproducibility within attempt
        rng = random.Random(attempt * 12345 + 67890)
        shuffled_edges = edges[:]
        rng.shuffle(shuffled_edges)
        
        valid = True
        for (u, v) in shuffled_edges:
            if time.time() - start_time > timeout:
                valid = False
                break
            
            # Try each color in random order
            color_order = list(range(num_colors))
            rng.shuffle(color_order)
            
            assigned = False
            for c in color_order:
                if _can_assign(colors, n, u, v, c):
                    colors[u][v] = c
                    colors[v][u] = c
                    assigned = True
                    break
            
            if not assigned:
                valid = False
                break
        
        if valid:
            # Flatten colors in lexicographic order
            flat_colors = [colors[i][j] for i in range(n) for j in range(i+1, n)]
            # Store result
            construct._result = {'n': n, 'colors': flat_colors}
            return True
    
    return False

def _can_assign(colors, n, u, v, c):
    """
    Check if assigning color c to edge (u,v) would create a monochromatic C4 in color c.
    """
    # Check pair (u,v): count common neighbors in color c
    common_count = 0
    for w in range(n):
        if w != u and w != v:
            if colors[u][w] == c and colors[v][w] == c:
                common_count += 1
                if common_count >= 2:
                    return False
    
    # Check pairs involving u and vertices w where colors[u][w] == c
    for w in range(n):
        if w == u or w == v:
            continue
        if colors[u][w] == c:
            # After assignment, u is a common neighbor of v and w in color c
            # Count common neighbors of v and w in color c
            count = 0
            for x in range(n):
                if x != v and x != w:
                    if colors[v][x] == c and colors[w][x] == c:
                        count += 1
                        if count >= 2:
                            return False
    
    # Check pairs involving v and vertices w where colors[v][w] == c
    for w in range(n):
        if w == u or w == v:
            continue
        if colors[v][w] == c:
            # After assignment, v is a common neighbor of u and w in color c
            count = 0
            for x in range(n):
                if x != u and x != w:
                    if colors[u][x] == c and colors[w][x] == c:
                        count += 1
                        if count >= 2:
                            return False
    
    return True
