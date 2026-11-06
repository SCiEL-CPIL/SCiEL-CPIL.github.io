import time
import random

def construct():
    start_time = time.time()
    deadline = 140  # 10s margin
    
    n = 16
    # Generate all pairs
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Helper to check validity
    def is_valid(colors):
        # colors is a dict or list mapping pair index to color
        # We need to check every K4
        # K4 is defined by 4 vertices
        # There are C(16,4) = 1820 K4s
        # For each K4, check if number of distinct colors >= 5
        # To speed up, we can precompute or just iterate.
        # 1820 * (check 6 edges) is fast.
        
        # Map (i,j) to color
        # colors is a list of 120 ints.
        # We need a fast lookup.
        # Let's create a 16x16 matrix
        mat = [[-1]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                mat[i][j] = colors[idx]
                mat[j][i] = colors[idx]
                idx += 1
        
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        # Vertices a, b, c, d
                        # Edges: (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
                        cols = {
                            mat[a][b], mat[a][c], mat[a][d],
                            mat[b][c], mat[b][d],
                            mat[c][d]
                        }
                        if len(cols) < 5:
                            return False
        return True

    # Try Circle Method 1-Factorization
    # Vertices 0..14, 15 is infinity
    # Colors 0..14
    def circle_method_perm(perm):
        # perm is a permutation of 0..15
        # We apply perm to vertices
        colors = [0]*120
        # Build the base factorization on 0..15
        # Then apply perm
        # Actually, easier to just generate colors for permuted vertices
        # c(perm(i), perm(j)) = base_color(i, j) ?
        # No, the structure depends on the labels.
        # Let's just generate the base one and then permute the vertex indices in the output list?
        # The output must be for pairs (i,j) in lex order of original indices.
        # So if we permute vertices, we need to map the colors back.
        
        # Base coloring on 0..15
        base_colors = [0]*120
        idx = 0
        for i in range(16):
            for j in range(i+1, 16):
                # Determine color of (i,j) in circle method
                if i == 15 or j == 15:
                    # One is infinity
                    v = i if j == 15 else j
                    # Color is v?
                    # In circle method, F_k contains (inf, k).
                    # So color k is assigned to (15, k).
                    base_colors[idx] = v
                else:
                    # Both in 0..14
                    # Find k such that i,j are in F_k
                    # F_k contains ( (k+i') mod 15, (k-i') mod 15 )
                    # Let u = i, v = j.
                    # We need k + i' = u, k - i' = v (mod 15) ?
                    # 2k = u + v (mod 15)
                    # k = (u + v) * inv(2) mod 15
                    # inv(2) mod 15 is 8 (2*8=16=1)
                    k = (i + j) * 8 % 15
                    base_colors[idx] = k
                idx += 1
        
        # Apply permutation
        # The output list corresponds to pairs (i,j) of the FINAL vertex set 0..15.
        # If we permute vertices, the edge (i,j) in the final set corresponds to edge (perm_inv(i), perm_inv(j)) in the base set?
        # No, usually we say "coloring of K_n".
        # If we have a valid coloring on vertices 0..15, any isomorphic copy is valid.
        # The evaluator checks the coloring on the fixed vertex set 0..15.
        # So we can just generate a random permutation of the 16 vertices, apply it to the base coloring, and check.
        
        # Let's do this:
        # base_col[ (u,v) ] = c
        # new_col[ (i,j) ] = base_col[ (perm_inv(i), perm_inv(j)) ]
        
        inv_perm = [0]*16
        for i in range(16):
            inv_perm[perm[i]] = i
            
        new_colors = [0]*120
        idx = 0
        for i in range(16):
            for j in range(i+1, 16):
                u = inv_perm[i]
                v = inv_perm[j]
                if u > v:
                    u, v = v, u
                # Find index of (u,v) in base_colors
                # base index for (u,v)
                base_idx = 0
                for uu in range(16):
                    for vv in range(uu+1, 16):
                        if uu == u and vv == v:
                            new_colors[idx] = base_colors[base_idx]
                            break
                        base_idx += 1
                idx += 1
        return new_colors

    # Generate base circle method
    base_colors = [0]*120
    idx = 0
    for i in range(16):
        for j in range(i+1, 16):
            if i == 15 or j == 15:
                v = i if j == 15 else j
                base_colors[idx] = v
            else:
                k = (i + j) * 8 % 15
                base_colors[idx] = k
            idx += 1
            
    # Check base
    if is_valid(base_colors):
        return base_colors
        
    # Try random permutations
    # Limit time
    while time.time() < deadline:
        perm = list(range(16))
        random.shuffle(perm)
        cols = circle_method_perm(perm)
        if is_valid(cols):
            return cols
            
    # Fallback to seed (17 colors) which is known valid
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
