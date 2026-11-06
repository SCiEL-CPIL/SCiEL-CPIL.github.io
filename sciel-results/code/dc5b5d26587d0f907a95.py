import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    def color_index(n, i, j):
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    def verify_colors(colors, n):
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    if colors[color_index(n, a, b)] == colors[color_index(n, a, c)] == colors[color_index(n, b, c)]:
                        return False
        return True

    # Build the 50-vertex construction based on Z_5 x Z_5
    n = 50
    colors = []
    
    for i in range(n):
        for j in range(i + 1, n):
            u_i, u_j = i // 5, i % 5
            v_i, v_j = j // 5, j % 5
            
            dx = (v_i - u_i) % 5
            dy = (v_j - u_j) % 5
            
            if dx == 0:
                c = 0
            elif dy == 0:
                c = 1
            else:
                inv_dx = pow(dx, -1, 5)
                slope = (dy * inv_dx) % 5
                if slope == 1:
                    c = 2
                elif slope == 2:
                    c = 3
                elif slope == 3:
                    c = 0
                elif slope == 4:
                    c = 1
                else:
                    c = 2
            
            colors.append(c)

    if not verify_colors(colors, n):
        # Fallback to 32-vertex Greenwood-Gleason coloring
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r

        log_map, x = {}, 1
        for e in range(15):
            log_map[x] = e
            x = mul(x, 2)

        n = 32
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if i // 16 == j // 16:
                    colors.append(log_map[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)

        return {'n': n, 'colors': colors}

    # We have a valid 50-vertex coloring. Now try to extend to 51 vertices.
    # For a new vertex 50, we need to assign colors to edges (50, i) for i in 0..49
    # such that no monochromatic triangle is formed.
    #
    # Constraint: for each color c, the set S_c = {i : color(50,i) = c} must be
    # an independent set in the color-c graph of K_50.
    #
    # This is equivalent to: we need to 4-color the vertices 0..49 such that
    # each color class is an independent set in the corresponding color graph.
    #
    # Use backtracking with constraint propagation.

    # Precompute: for each color c, build the adjacency list of the color-c graph
    # adj[c][i] = set of j > i such that color(i,j) = c
    adj = [[] for _ in range(4)]
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[color_index(n, i, j)]
            adj[c][i].append(j)
    
    # Also build reverse: for each color c and vertex i, which vertices j < i have color(c) edge to i
    # Actually, for independence check, we need: if i and j are both in S_c, then edge(i,j) != c
    # So S_c must be an independent set in graph G_c where G_c has edge (i,j) iff color(i,j) == c

    # Let's build G_c adjacency as sets for fast lookup
    G = [set() for _ in range(4)]
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[color_index(n, i, j)]
            G[c].add((i, j))
    
    # For faster independence check, build adjacency lists
    G_adj = [[set() for _ in range(n)] for _ in range(4)]
    for c in range(4):
        for (i, j) in G[c]:
            G_adj[c][i].add(j)
            G_adj[c][j].add(i)

    # Now we need to assign each vertex i a color a[i] in {0,1,2,3} such that
    # for all i < j: if a[i] == a[j] == c, then (i,j) not in G[c], i.e., color(i,j) != c
    #
    # This is a constraint satisfaction problem. Use backtracking with forward checking.

    def try_extend_51():
        """Try to find a valid coloring for the 51st vertex."""
        assignment = [-1] * n
        # Order vertices by degree in the "constraint graph" (most constrained first)
        # The constraint graph for vertex i: edges to j where color(i,j) = a[j] would conflict
        # Actually, let's just try natural order with some heuristics
        
        # Heuristic: process vertices in order of their "constrainedness"
        # A vertex i is more constrained if it has many edges of each color
        # Let's compute for each vertex i, the number of edges of each color
        deg_counts = [[0, 0, 0, 0] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[color_index(n, i, j)]
                deg_counts[i][c] += 1
                deg_counts[j][c] += 1
        
        # Sort by sum of squares (more "spread" = less constrained? or more?)
        # Actually, vertices with high degree in any single color are more constrained
        # because if a[i] = c, then no neighbor in G_adj[c][i] can also have color c
        order = list(range(n))
        # Sort by max degree in any color (descending) - most constrained first
        order.sort(key=lambda i: max(deg_counts[i]), reverse=True)
        
        # For backtracking, we need to check consistency as we assign
        # Let's use a simpler approach: try all 4^50? No, that's too many.
        # Use backtracking with forward checking.
        
        # Actually, let's try a randomized greedy approach first, then backtracking if it fails.
        
        # Randomized greedy: for each vertex in order, try each color, pick one that doesn't
        # immediately conflict with already-assigned vertices.
        
        import random
        random.seed(42)
        
        for attempt in range(50):
            if time.time() > deadline:
                return None
            
            assignment = [-1] * n
            # Shuffle the order slightly for different attempts
            if attempt > 0:
                order_shuffled = order[:]
                random.shuffle(order_shuffled)
                # But keep the most constrained ones first
                # Actually, let's just use the sorted order but randomize tie-breaking
                order_use = order[:]
                if attempt % 2 == 1:
                    # Reverse for odd attempts
                    order_use = order_use[::-1]
            else:
                order_use = order
            
            success = True
            for i in order_use:
                # Determine which colors are available for vertex i
                available = [0, 1, 2, 3]
                for j in order_use:
                    if j == i or assignment[j] == -1:
                        continue
                    # If assignment[j] = c, then color(i,j) cannot be c
                    # because that would form triangle (50, i, j) with all color c
                    # Wait, no. The constraint is: if a[i] = c and a[j] = c, then
                    # edge(i,j) must NOT have color c.
                    # So if assignment[j] = c, then a[i] cannot be c if color(i,j) == c.
                    c_ij = colors[color_index(n, min(i,j), max(i,j))]
                    if assignment[j] == c_ij:
                        if c_ij in available:
                            available.remove(c_ij)
                
                if not available:
                    success = False
                    break
                
                # Pick a color from available, prefer the one that appears least so far
                counts = [0, 0, 0, 0]
                for j in range(n):
                    if assignment[j] != -1:
                        counts[assignment[j]] += 1
                
                # Pick the available color with minimum count
                min_count = min(counts[c] for c in available)
                candidates = [c for c in available if counts[c] == min_count]
                if attempt % 3 == 0:
                    # Deterministic: pick first
                    choice = candidates[0]
                elif attempt % 3 == 1:
                    # Random
                    choice = random.choice(candidates)
                else:
                    # Pick the color that leaves the most options for future vertices
                    # This is expensive, so just pick randomly
                    choice = random.choice(candidates)
                
                assignment[i] = choice
            
            if success:
                # Verify the full assignment
                # Check: for all i < j, if assignment[i] == assignment[j] == c, then color(i,j) != c
                valid = True
                for i in range(n):
                    for j in range(i + 1, n):
                        if assignment[i] == assignment[j]:
                            c = assignment[i]
                            if colors[color_index(n, i, j)] == c:
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid:
                    # Build the 51-vertex coloring
                    new_colors = colors[:]
                    # Add edges from vertex 50 to vertices 0..49
                    # In lexicographic order, edges (50, i) come after all edges within 0..49
                    # Wait, no. Lexicographic order: (0,1), (0,2), ..., (0,49), (1,2), ..., (49,50)
                    # So edges involving vertex 50 are: (0,50), (1,50), ..., (49,50)
                    # These come at the end of the list.
                    # Current colors has n*(n-1)/2 = 50*49/2 = 1225 entries for edges within 0..49
                    # New edges: (0,50), (1,50), ..., (49,50) -> 50 more entries
                    for i in range(n):
                        new_colors.append(assignment[i])
                    
                    # Verify the full 51-vertex coloring
                    if verify_colors(new_colors, 51):
                        return new_colors
                    else:
                        # Should not happen if our check was correct
                        continue
            
        return None

    result = try_extend_51()
    if result is not None:
        return {'n': 51, 'colors': result}
    
    # Fallback to 50-vertex
    return {'n': n, 'colors': colors}
