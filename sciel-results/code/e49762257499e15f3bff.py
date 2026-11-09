import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin

    def has_mono_k222_fast(n, colors):
        """
        Fast check for monochromatic K2,2,2 using bitmasks.
        colors: list of 0/1 of length n*(n-1)/2
        """
        # Build adjacency bitmasks for each color
        adj = [[0] * n for _ in range(2)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        for c in [0, 1]:
            for u in range(n):
                for v in range(u + 1, n):
                    mask_c = adj[c][u] & adj[c][v]
                    if mask_c.bit_count() < 4:
                        continue
                    
                    verts = [i for i in range(n) if (mask_c >> i) & 1]
                    
                    found = False
                    for i_idx in range(len(verts)):
                        x = verts[i_idx]
                        mask_x = adj[c][x]
                        for j_idx in range(i_idx + 1, len(verts)):
                            y = verts[j_idx]
                            common_xy = mask_x & adj[c][y] & mask_c
                            if common_xy.bit_count() >= 2:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        return True
        return False

    def generate_paley(n):
        if n % 4 != 1:
            return None
        squares = {x * x % n for x in range(1, n)}
        colors = [0 if (j - i) % n in squares else 1 for i in range(n) for j in range(i + 1, n)]
        return colors

    # Try n=29 first (Paley)
    colors_29 = generate_paley(29)
    if colors_29:
        if not has_mono_k222_fast(29, colors_29):
            return {'n': 29, 'colors': colors_29}

    # Try n=17 (Paley)
    colors_17 = generate_paley(17)
    if colors_17:
        if not has_mono_k222_fast(17, colors_17):
            return {'n': 17, 'colors': colors_17}

    # Greedy backtracking search for n=30 with symmetry breaking
    if time.time() < start_time + 120:
        n_target = 30
        total_edges = n_target * (n_target - 1) // 2
        
        # Symmetry breaking: fix first few edges to color 0
        # Edge (0,1) -> index 0, (0,2) -> index 1, (0,3) -> index 2
        # We fix these to reduce the search space
        fixed_edges = {0: 0, 1: 0, 2: 0}
        
        # Build edge index mapping
        edge_idx = {}
        idx = 0
        for i in range(n_target):
            for j in range(i + 1, n_target):
                edge_idx[(i, j)] = idx
                idx += 1
        
        # For efficient K2,2,2 detection during search, we maintain adjacency sets
        # and check incrementally
        
        def check_k222_partial(adj_sets, vertex_set):
            """Check if any color class has K2,2,2 among the given vertices."""
            for c in [0, 1]:
                # For each pair (u,v) in vertex_set, check if they have >=4 common neighbors
                # in the same color, and among those, if any pair has >=2 common neighbors
                vertices = list(vertex_set)
                for u_idx in range(len(vertices)):
                    u = vertices[u_idx]
                    for v_idx in range(u_idx + 1, len(vertices)):
                        v = vertices[v_idx]
                        # Common neighbors of u and v in color c
                        common = adj_sets[c][u] & adj_sets[c][v]
                        common_in_set = common & vertex_set
                        if common_in_set.bit_count() < 4:
                            continue
                        
                        # Check if any two vertices in common_in_set have >=2 common neighbors
                        verts = [i for i in range(n_target) if (common_in_set >> i) & 1]
                        found = False
                        for i_idx in range(len(verts)):
                            x = verts[i_idx]
                            for j_idx in range(i_idx + 1, len(verts)):
                                y = verts[j_idx]
                                common_xy = adj_sets[c][x] & adj_sets[c][y] & common_in_set
                                if common_xy.bit_count() >= 2:
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            return True
            return False
        
        # Backtracking search
        def backtrack(edge_pos, colors, adj_sets, vertex_set):
            """
            Backtracking search for valid coloring.
            edge_pos: current edge position in lexicographic order
            colors: list of colors assigned so far (None for unassigned)
            adj_sets: adjacency bitmasks for each color
            vertex_set: bitmask of vertices that have been fully processed
            """
            if time.time() > deadline:
                return None
            
            if edge_pos == total_edges:
                return colors[:]
            
            i = edge_pos // (n_target - 1)
            j = i + 1 + (edge_pos % (n_target - 1))
            # More accurate: find (i,j) from edge_pos
            # edge_pos = i*(n-1) - i*(i-1)//2 + (j-i-1)
            # Let's use a different approach: iterate through edges in order
            
            # Actually, let's use a simpler edge enumeration
            # For edge_pos, find which (i,j) it corresponds to
            # We'll precompute this
            pass
        
        # Simpler approach: use random restarts with local search
        # but with smarter initialization based on Paley-like structure
        
        # Try to extend n=29 Paley to n=30
        if colors_29 and not has_mono_k222_fast(29, colors_29):
            # Try adding vertex 29 with various colorings
            for attempt in range(50):
                if time.time() > deadline:
                    break
                
                # Randomly assign colors for edges involving vertex 29
                new_colors = colors_29[:]
                for i in range(29):
                    new_colors.append(random.randint(0, 1))
                
                if not has_mono_k222_fast(30, new_colors):
                    return {'n': 30, 'colors': new_colors}
                
                # Local search: try flipping edges involving vertex 29
                for _ in range(200):
                    if time.time() > deadline:
                        break
                    # Pick a random edge involving vertex 29
                    i = random.randint(0, 28)
                    edge_idx_val = 29 * 28 // 2 + i  # edges (i,29) for i=0..28
                    new_colors[edge_idx_val] ^= 1
                    if not has_mono_k222_fast(30, new_colors):
                        return {'n': 30, 'colors': new_colors}
        
        # Pure random restarts with local search
        for restart in range(20):
            if time.time() > deadline:
                break
            
            colors = [random.randint(0, 1) for _ in range(total_edges)]
            if not has_mono_k222_fast(n_target, colors):
                return {'n': n_target, 'colors': colors}
            
            # Local search
            for step in range(500):
                if time.time() > deadline:
                    break
                idx = random.randint(0, total_edges - 1)
                colors[idx] ^= 1
                if not has_mono_k222_fast(n_target, colors):
                    return {'n': n_target, 'colors': colors}
    
    # Fallback
    n = 17
    colors = generate_paley(17)
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
