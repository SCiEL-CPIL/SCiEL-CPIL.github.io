"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    # Greenwood-Gleason colouring of K16 by cubic-residue classes of GF(16) (the Clebsch graph three times),
    # doubled: two copies with colour 3 on all edges between them (a bipartite colour class).
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    
    # Build base 32-vertex coloring
    base_colors = {}
    for i in range(32):
        for j in range(i + 1, 32):
            base_colors[(i, j)] = log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3
    
    def get_color(colors_dict, i, j):
        if i > j:
            i, j = j, i
        return colors_dict[(i, j)]
    
    def is_valid_extension(colors_dict, n, new_vertex, new_colors):
        """Check if assigning new_colors to edges from new_vertex to 0..n-1 creates no monochromatic triangle."""
        for c in range(4):
            # Find all vertices connected to new_vertex with color c
            neighbors_c = [v for v in range(n) if new_colors[v] == c]
            # Check if any pair among neighbors_c is connected with color c
            for idx1 in range(len(neighbors_c)):
                for idx2 in range(idx1 + 1, len(neighbors_c)):
                    v1, v2 = neighbors_c[idx1], neighbors_c[idx2]
                    if get_color(colors_dict, v1, v2) == c:
                        return False
        return True
    
    def try_add_vertex(colors_dict, n, deadline):
        """Try to add vertex n by backtracking over color assignments."""
        def backtrack(v_idx, current_assignment):
            if time.time() > deadline:
                return None
            if v_idx == n:
                return current_assignment[:]
            
            # Try each color for edge (v_idx, n)
            for c in range(4):
                current_assignment.append(c)
                if is_valid_extension_partial(colors_dict, n, v_idx, current_assignment):
                    result = backtrack(v_idx + 1, current_assignment)
                    if result is not None:
                        return result
                current_assignment.pop()
            return None
        
        return backtrack(0, [])
    
    def is_valid_extension_partial(colors_dict, n, v_idx, assignment):
        """Check if the partial assignment is consistent (no monochromatic triangles involving new vertex)."""
        for c in range(4):
            # Find all vertices assigned color c so far
            neighbors_c = [v for v in range(v_idx + 1) if assignment[v] == c]
            # Check if any pair among neighbors_c is connected with color c
            for idx1 in range(len(neighbors_c)):
                for idx2 in range(idx1 + 1, len(neighbors_c)):
                    v1, v2 = neighbors_c[idx1], neighbors_c[idx2]
                    if get_color(colors_dict, v1, v2) == c:
                        return False
        return True
    
    # Start with base 32-vertex coloring
    colors_dict = dict(base_colors)
    n = 32
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Try to extend the coloring
    while time.time() < deadline:
        new_assignment = try_add_vertex(colors_dict, n, deadline)
        if new_assignment is None:
            break
        
        # Add the new vertex
        for v in range(n):
            colors_dict[(v, n)] = new_assignment[v]
        n += 1
    
    # Convert to list format
    colors_list = []
    for i in range(n):
        for j in range(i + 1, n):
            colors_list.append(colors_dict[(i, j)])
    
    return {'n': n, 'colors': colors_list}
# EVOLVE-BLOCK-END
