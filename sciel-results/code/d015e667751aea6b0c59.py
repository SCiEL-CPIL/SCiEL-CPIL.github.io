import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Try greedy constructive approach for n=26 first
    result_26 = greedy_construct(26, time_limit=30.0)
    
    if result_26 is None:
        # Fallback to local search
        result_26 = local_search(26, time_limit=20.0)
        if result_26 is None:
            result_26 = local_search(26, time_limit=30.0)
            if result_26 is None:
                # Absolute fallback
                return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}
    
    # 2. Try to extend to n=27 using greedy approach
    base_colors = result_26['colors']
    n_base = 26
    
    # Precompute adjacency lists for the base graph (n=26) for each color
    adj = [[set() for _ in range(n_base)] for _ in range(5)]
    for i in range(n_base):
        for j in range(i + 1, n_base):
            idx = i * (n_base - 1) - i * (i - 1) // 2 + (j - i - 1)
            c = base_colors[idx]
            adj[c][i].add(j)
            adj[c][j].add(i)
    
    # Try greedy extension with backtracking
    new_vertex = n_base
    new_colors = [None] * n_base
    
    # Order vertices to add: try to add vertices with fewer constraints first
    # But for a single new vertex, we just need to assign colors to (new_vertex, i) for i in 0..n_base-1
    
    def can_add_color(vertex, color, current_assignment):
        """Check if we can assign 'color' to edge (new_vertex, vertex) given current_assignment"""
        # Find all other vertices that already have this color assigned to (new_vertex, v)
        for v in range(n_base):
            if v == vertex:
                continue
            if current_assignment[v] is not None and current_assignment[v] == color:
                # Check if vertex and v share a common neighbor in color 'color'
                common = adj[color][vertex] & adj[color][v]
                if len(common) >= 2:
                    return False
        return True
    
    def greedy_extend_with_backtracking(time_limit=30.0):
        """Try to assign colors to (new_vertex, i) for all i using greedy with backtracking"""
        start = time.time()
        
        # Try multiple random orderings
        for attempt in range(100):
            if time.time() - start > time_limit:
                break
            
            # Random order of vertices to assign
            order = list(range(n_base))
            random.shuffle(order)
            
            assignment = [None] * n_base
            
            def backtrack(idx):
                if time.time() - start > time_limit:
                    return False
                if idx == n_base:
                    return True
                
                vertex = order[idx]
                
                # Try colors in random order
                colors_to_try = [0, 1, 2, 3, 4]
                random.shuffle(colors_to_try)
                
                for color in colors_to_try:
                    if can_add_color(vertex, color, assignment):
                        assignment[vertex] = color
                        if backtrack(idx + 1):
                            return True
                        assignment[vertex] = None
                
                return False
            
            if backtrack(0):
                return assignment
        
        return None
    
    extension = greedy_extend_with_backtracking(time_limit=20.0)
    
    if extension is not None:
        # Construct the full coloring for n=27
        final_colors = [0] * (27 * 26 // 2)
        
        idx = 0
        for i in range(27):
            for j in range(i + 1, 27):
                if j < 26:
                    base_idx = i * 25 - i * (i - 1) // 2 + (j - i - 1)
                    final_colors[idx] = base_colors[base_idx]
                else:
                    # j == 26
                    final_colors[idx] = extension[i]
                idx += 1
        
        return {'n': 27, 'colors': final_colors}
    
    # If extension failed, return n=26
    return result_26

def greedy_construct(n, time_limit=30.0):
    """
    Greedy constructive algorithm that builds the coloring vertex-by-vertex.
    For each new vertex, assign colors to edges to existing vertices while
    maintaining the C4-free constraint.
    """
    start_time = time.time()
    
    # adj[c][v] = set of neighbors of v in color c
    adj = [[set() for _ in range(n)] for _ in range(5)]
    colors = [0] * (n * (n - 1) // 2)
    
    # For each vertex i, track which colors are "available" for edge (i, j)
    # A color c is available for edge (i, j) if adding it doesn't create a C4
    
    def is_color_available(i, j, c, current_adj):
        """Check if color c can be assigned to edge (i, j) without creating a C4"""
        # For each pair of common neighbors u, v of i and j in color c,
        # check if u and v share a common neighbor in color c
        # Actually, we need to check: if we add edge (i,j) with color c,
        # does any pair of vertices (u,v) now have two common neighbors in color c?
        
        # The new edge (i,j) in color c means:
        # - i gains j as a neighbor in color c
        # - j gains i as a neighbor in color c
        
        # This creates a C4 if there exist u, v such that:
        # u and v both have i and j as common neighbors in color c
        # i.e., u is adjacent to both i and j in color c, and v is adjacent to both i and j in color c
        # But wait, that's not quite right. A C4 is u-v-w-x-u.
        
        # Let me think more carefully. Adding edge (i,j) in color c creates a C4 if:
        # There exist vertices u and v (distinct from i,j) such that:
        # - u is adjacent to i in color c
        # - u is adjacent to j in color c  
        # - v is adjacent to i in color c
        # - v is adjacent to j in color c
        # Then u-i-j-v-u is a C4 (or some permutation).
        
        # Actually, the C4 would be u-v-j-i-u if u and v are both common neighbors of i and j in color c.
        # So we need: the number of common neighbors of i and j in color c (excluding the new edge)
        # must be at most 1. If it's >= 2, we'd have a C4.
        
        # Common neighbors of i and j in color c (before adding the edge):
        common = current_adj[c][i] & current_adj[c][j]
        return len(common) <= 1
    
    # Build vertex by vertex
    for new_v in range(n):
        if time.time() - start_time > time_limit:
            return None
        
        # Assign colors to edges (new_v, i) for i < new_v
        # Use backtracking
        assignment = [None] * new_v
        
        def backtrack(idx):
            if time.time() - start_time > time_limit:
                return False
            if idx == new_v:
                return True
            
            i = idx
            # Try colors
            for c in range(5):
                if is_color_available(new_v, i, c, adj):
                    assignment[i] = c
                    # Tentatively add to adj
                    adj[c][new_v].add(i)
                    adj[c][i].add(new_v)
                    
                    if backtrack(idx + 1):
                        return True
                    
                    # Undo
                    adj[c][new_v].remove(i)
                    adj[c][i].remove(new_v)
                    assignment[i] = None
            
            return False
        
        if not backtrack(0):
            # Try with different random color ordering
            for attempt in range(10):
                if time.time() - start_time > time_limit:
                    return None
                
                assignment = [None] * new_v
                
                def backtrack_random(idx):
                    if time.time() - start_time > time_limit:
                        return False
                    if idx == new_v:
                        return True
                    
                    i = idx
                    colors_to_try = list(range(5))
                    random.shuffle(colors_to_try)
                    
                    for c in colors_to_try:
                        if is_color_available(new_v, i, c, adj):
                            assignment[i] = c
                            adj[c][new_v].add(i)
                            adj[c][i].add(new_v)
                            
                            if backtrack_random(idx + 1):
                                return True
                            
                            adj[c][new_v].remove(i)
                            adj[c][i].remove(new_v)
                            assignment[i] = None
                    
                    return False
                
                if backtrack_random(0):
                    break
            else:
                return None
        
        # Commit the assignment
        for i in range(new_v):
            c = assignment[i]
            # Find the index of edge (i, new_v) in the colors array
            # Edge (i, new_v) where i < new_v
            # Index: sum_{k=0}^{i-1} (n-1-k) + (new_v - i - 1)
            # But we're building incrementally, so the "current n" is new_v+1
            # Actually, the final array has n vertices, so we need the index in the full array
            # For edge (i, j) with i < j, index = sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
            # = i*(n-1) - i*(i-1)//2 + (j-i-1)
            idx = i * (n - 1) - i * (i - 1) // 2 + (new_v - i - 1)
            colors[idx] = c
    
    return {'n': n, 'colors': colors}

def local_search(n, time_limit=10.0):
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Initial random coloring
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    def get_violations(colors):
        neighbors = [[0] * n for _ in range(5)]
        for idx, (i, j) in enumerate(edges):
            c = colors[idx]
            neighbors[c][i] |= (1 << j)
            neighbors[c][j] |= (1 << i)
            
        violations = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[c][u] & neighbors[c][v]
                    k = common.bit_count()
                    if k >= 2:
                        violations += k * (k - 1) // 2
        return violations

    current_violations = get_violations(colors)
    
    if current_violations == 0:
        return {'n': n, 'colors': colors}
        
    best_violations = current_violations
    best_colors = colors[:]
    
    start_time = time.time()
    
    temperature = 1.0
    cooling_rate = 0.999
    
    while time.time() < start_time + time_limit:
        e_idx = random.randint(0, num_edges - 1)
        old_color = colors[e_idx]
        
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
            
        colors[e_idx] = new_color
        
        new_violations = get_violations(colors)
        
        delta = new_violations - current_violations
        
        if delta < 0 or random.random() < pow(2.71828, -delta / temperature):
            current_violations = new_violations
            if new_violations < best_violations:
                best_violations = new_violations
                best_colors = colors[:]
                if best_violations == 0:
                    return {'n': n, 'colors': colors}
        else:
            colors[e_idx] = old_color
            
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
            
    if best_violations == 0:
        return {'n': n, 'colors': best_colors}
    return None
