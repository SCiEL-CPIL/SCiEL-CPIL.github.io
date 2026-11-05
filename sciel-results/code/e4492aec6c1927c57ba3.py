import random

N = 16

def construct():
    # Initialize all colors to 0
    colors = [0] * (N * (N - 1) // 2)
    
    # Helper to get index
    def get_idx(i, j):
        if i > j:
            i, j = j, i
        return i * N - i * (i + 1) // 2 + (j - i - 1)

    # Helper to check if coloring edge (u, v) with color c creates a rainbow C4
    def is_valid(u, v, c):
        # A C4 containing (u, v) must have the form u-x-v-y-u or u-x-y-v-u?
        # No, a C4 is a cycle of length 4.
        # If we add edge (u, v) with color c, we need to check all C4s that include (u, v).
        # Such a C4 is u - a - b - v - u.
        # Edges: (u,a), (a,b), (b,v), (v,u).
        # Colors: col(u,a), col(a,b), col(b,v), c.
        # We need to ensure these 4 are not all distinct.
        
        for a in range(N):
            if a == u or a == v: continue
            for b in range(N):
                if b == u or b == v or b == a: continue
                # Check cycle u-a-b-v-u
                # Edges: (u,a), (a,b), (b,v), (v,u)
                # Note: (v,u) is the same as (u,v)
                idx_ua = get_idx(u, a)
                idx_ab = get_idx(a, b)
                idx_bv = get_idx(b, v)
                idx_vu = get_idx(u, v) # This is the edge we are coloring
                
                # Current colors of the other 3 edges
                c_ua = colors[idx_ua]
                c_ab = colors[idx_ab]
                c_bv = colors[idx_bv]
                
                # The 4 colors would be: c_ua, c_ab, c_bv, c
                # If these 4 are distinct, it's invalid.
                if len({c_ua, c_ab, c_bv, c}) == 4:
                    return False
        return True

    # Greedy assignment
    # We try to assign a new unique color to each edge.
    # We iterate through edges in a fixed order (lexicographic).
    current_max_color = 0
    
    # To improve performance, we can shuffle the order of edges or use a specific order.
    # Let's try lexicographic order first.
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            edges.append((i, j))
            
    for (u, v) in edges:
        idx = get_idx(u, v)
        # Try to assign the next new color
        next_color = current_max_color + 1
        if is_valid(u, v, next_color):
            colors[idx] = next_color
            current_max_color = next_color
        else:
            # If we can't assign a new unique color, maybe we can assign an existing color?
            # But we want to MAXIMIZE distinct colors.
            # If we can't add a new one, we leave it as 0 (or existing).
            # Actually, if it's 0, and 0 is already used, it doesn't increase the count.
            # But it might block future edges?
            # Leaving it 0 is safest for validity.
            pass
            
    return colors
