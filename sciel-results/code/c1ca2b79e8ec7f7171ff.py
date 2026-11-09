"""Constructive backtracking for 4-coloring K_n without C6 in any color class."""
import itertools
import sys
sys.setrecursionlimit(10000)

def construct():
    # We'll do a backtracking search to find a valid 4-coloring of K_n
    # For each color c, maintain forbidden_pairs[c] = set of (u1, u5) 
    # such that there's a path of length 4 (5 vertices, 4 edges) in color c from u1 to u5
    # When adding vertex v, if we color (v,u1)=c and (v,u5)=c and (u1,u5) in forbidden_pairs[c], C6 formed.
    
    # Path of length 4 in color c: u1-a-b-d-u5, edges (u1,a),(a,b),(b,d),(d,u5) all color c
    # So forbidden_pairs[c] contains (u1, u5) if such a path exists.
    
    # We build vertices 0..n-1. When adding vertex v, we assign colors to edges (v,0), (v,1), ..., (v,v-1)
    # Then update forbidden_pairs for each color.
    
    # To update forbidden_pairs[c] after adding v with some colors assigned:
    # New paths of length 4 in color c that involve v:
    # v can be an endpoint: v-a-b-d-u5 or u1-a-b-d-v
    # v can be internal: u1-a-v-b-d or u1-v-a-b-d or u1-a-b-v-d
    # But we only care about pairs (u1,u5) where both are < v (existing vertices) for the constraint on future edges.
    # Actually, forbidden_pairs is used when adding a NEW vertex w > v, where we check edges (w, u1) and (w, u5) for u1,u5 < w.
    # So forbidden_pairs[c] should contain pairs (u1, u5) with u1, u5 < current_v where path of length 4 exists.
    
    # Let me define: after building vertices 0..v-1, forbidden[c] = set of (u1, u5) with 0<=u1<u5<v 
    # such that there's a path of length 4 in color c from u1 to u5 using only vertices < v.
    
    # When we add vertex v and assign colors to (v, u) for u in 0..v-1:
    # New forbidden pairs for color c that include v as endpoint or internal:
    # For path u1-a-b-d-u5 of length 4 in color c:
    # Case 1: v is u1: then path is v-a-b-d-u5, so (v,u5) becomes a new "endpoint pair" but v is not < v, so we don't add to forbidden for future.
    # Wait, I need to reconsider. forbidden[c] after adding v should contain pairs (u1,u5) with u1,u5 <= v (i.e., < v+1) for use when adding v+1.
    
    # Let me redefine: after processing vertices 0..v, forbidden[c] = set of (u1, u5) with 0<=u1<u5<=v 
    # such that there's a path of length 4 in color c from u1 to u5 using vertices in 0..v.
    
    # When adding vertex v (so we have vertices 0..v-1 already, forbidden_old[c] for pairs < v):
    # New pairs (u1, u5) with u1,u5 <= v that have a path of length 4 in color c:
    # - Pairs already in forbidden_old[c] (both < v)
    # - Pairs where one endpoint is v: (v, u) for u < v, if there's a path v-a-b-d-u of length 4 in color c
    #   This requires: (v,a)=c, (a,b)=c, (b,d)=c, (d,u)=c for some a,b,d < v
    #   Equivalently, for each u < v, check if there exist a,b,d < v with (v,a)=c, (a,b)=c, (b,d)=c, (d,u)=c
    #   This is: for each a with (v,a)=c, for each b with (a,b)=c, for each d with (b,d)=c, if (d,u)=c then (v,u) is a new forbidden pair
    # - Pairs (u1, u5) with both < v but the path uses v: 
    #   Path u1-a-v-b-d-u5 or u1-v-a-b-d or u1-a-b-v-d
    #   u1-a-v-b-d-u5: (u1,a)=c, (a,v)=c, (v,b)=c, (b,d)=c, (d,u5)=c
    #   u1-v-a-b-d: (u1,v)=c, (v,a)=c, (a,b)=c, (b,d)=c -> this gives pair (u1, d)
    #   u1-a-b-v-d: (u1,a)=c, (a,b)=c, (b,v)=c, (v,d)=c -> this gives pair (u1, d)
    
    # This is getting complex. Let me just compute forbidden[c] from scratch after each step, or maintain it incrementally.
    
    # Given time, let me do a simpler approach: for each candidate coloring of edges from v, check if it creates a C6 
    # by checking all 6-cycles involving v.
    
    # A 6-cycle involving v: v, u1, u2, u3, u4, u5 (in some order) where all edges are same color.
    # For a fixed color c, a C6 in color c containing v exists iff there exist distinct u1,u2,u3,u4,u5 < v 
    # such that (v,u1)=c, (u1,u2)=c, (u2,u3)=c, (u3,u4)=c, (u4,u5)=c, (u5,v)=c.
    
    # So for each color c, let S_c = {u < v : color(v,u) = c}. Then we need a path of length 4 in color c 
    # from some u1 in S_c to some u5 in S_c (with u1 != u5, and the path uses 4 intermediate vertices).
    
    # I'll precompute for the current graph (vertices 0..v-1) for each color c, the reachability by path of length 4.
    
    def has_path_length_4(color_c, u1, u5, edge_colors, v):
        # Check if there's a path u1-a-b-d-u5 of length 4 all in color_c, using vertices < v
        # u1, u5 < v
        for a in range(v):
            if a == u1 or a == u5:
                continue
            if edge_colors[(min(u1,a), max(u1,a))] != color_c:
                continue
            for b in range(v):
                if b == u1 or b == u5 or b == a:
                    continue
                if edge_colors[(min(a,b), max(a,b))] != color_c:
                    continue
                for d in range(v):
                    if d == u1 or d == u5 or d == a or d == b:
                        continue
                    if edge_colors[(min(b,d), max(b,d))] != color_c:
                        continue
                    if edge_colors[(min(d,u5), max(d,u5))] == color_c:
                        return True
        return False
    
    def check_no_c6_with_v(v, new_colors, edge_colors):
        # new_colors: dict mapping u -> color for edge (v,u), u < v
        # Check for each color c if there's a C6 in color c containing v
        for c in range(4):
            S_c = [u for u in range(v) if new_colors[u] == c]
            if len(S_c) < 2:
                continue
            for u1_idx in range(len(S_c)):
                for u5_idx in range(u1_idx+1, len(S_c)):
                    u1 = S_c[u1_idx]
                    u5 = S_c[u5_idx]
                    if has_path_length_4(c, u1, u5, edge_colors, v):
                        return False
        return True
    
    # Backtracking search
    # edge_colors: dict mapping (i,j) with i<j to color
    # We build vertex by vertex.
    
    def backtrack(v, edge_colors):
        # v is the next vertex to add (0-indexed, so we're adding vertex v, which connects to 0..v-1)
        if v == 0:
            return True
        if v == 1:
            edge_colors[(0,1)] = 0
            return True
        
        # Try all color assignments for edges (v, 0), (v, 1), ..., (v, v-1)
        # Use symmetry breaking: color(v,0) = 0 (WLOG, since colors are symmetric)
        # For v >= 2, we can fix color(v, 0) = 0
        
        def assign_colors(idx, current_assignment):
            if idx == v:
                # All edges assigned, check for C6
                if check_no_c6_with_v(v, current_assignment, edge_colors):
                    # Commit
                    for u in range(v):
                        edge_colors[(min(u,v), max(u,v))] = current_assignment[u]
                    return True
                return False
            
            # Symmetry breaking
            if idx == 0:
                colors_to_try = [0]
            else:
                colors_to_try = [0, 1, 2, 3]
            
            for c in colors_to_try:
                current_assignment[idx] = c
                if assign_colors(idx + 1, current_assignment):
                    return True
            return False
        
        return assign_colors(0, [0]*v)
    
    # Try to build up to n=17
    for target_n in range(17, 8, -1):
        edge_colors = {}
        if backtrack(target_n, edge_colors):
            n = target_n
            # Extract colors in lexicographic order
            colors = []
            for i in range(n):
                for j in range(i+1, n):
                    colors.append(edge_colors[(i,j)])
            return {'n': n, 'colors': colors}
    
    # Fallback: return the seed solution (n=9)
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

