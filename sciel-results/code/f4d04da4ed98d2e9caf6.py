import time
import sys

def construct():
    deadline = time.time() + 140
    
    # 1. Construct K32 coloring (Greenwood-Gleason based on GF(16))
    # GF(16) elements 0..15. 
    # Multiplicative group is cyclic order 15.
    # We need a partition of GF(16)* into 4 sets S0, S1, S2, S3 such that no set contains x, y, x+y?
    # Actually, the standard construction for R(3,3,3,3) > 16 uses GF(16).
    # The edges are colored based on the ratio (i-j) in GF(16).
    
    # Let's use the known K32 coloring from the seed program, which is correct.
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
        
    # Inverse log for faster access?
    # The seed program uses log[(i%16) ^ (j%16)] % 3.
    # This maps to colors 0,1,2.
    # The two blocks are connected by color 3.
    
    def base_coloring(n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if i // 16 == j // 16:
                    # Same block
                    colors.append(log[(i % 16) ^ (j % 16)] % 3)
                else:
                    # Different blocks
                    colors.append(3)
        return colors

    def edge_index(n, i, j):
        idx = 0
        for k in range(i):
            idx += (n - 1 - k)
        idx += (j - i - 1)
        return idx

    # Start with K32
    n = 32
    colors = base_coloring(n)
    best_n = n
    best_colors = colors[:]

    # Precompute base colors in a 2D array for fast access
    # base_colors[i][j] for i < j
    base_colors = [[-1]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            idx = edge_index(n, i, j)
            base_colors[i][j] = colors[idx]
            base_colors[j][i] = colors[idx]

    def try_extend(colors, n, base_colors):
        """
        Try to add vertex n.
        Returns list of colors for edges (0,n)..(n-1,n) or None.
        """
        # assignment[i] is color of edge (i, n)
        assignment = [0] * n
        
        # Precompute for each vertex i, the set of colors used in edges (i, j) for j < i?
        # No, we need to check triangles (a, b, n).
        # For a fixed i (being assigned color c to edge (i,n)),
        # and a fixed a < i, if base_colors[a][i] == c, then assignment[a] cannot be c.
        
        # We iterate i from 0 to n-1.
        # When assigning assignment[i] = c, we check all a < i.
        # If base_colors[a][i] == c, then assignment[a] must not be c.
        # But assignment[a] is already assigned.
        
        def backtrack(pos):
            if pos == n:
                return True
            
            for c in range(4):
                # Check conflict with all a < pos
                ok = True
                for a in range(pos):
                    if base_colors[a][pos] == c:
                        if assignment[a] == c:
                            ok = False
                            break
                if not ok:
                    continue
                
                assignment[pos] = c
                if backtrack(pos + 1):
                    return True
            return False
        
        if backtrack(0):
            return assignment
        return None

    # Extension loop
    while n < 62:
        if time.time() > deadline - 10:
            break
        
        # Re-initialize base_colors for the new n? 
        # base_colors is currently for K_n.
        # We need to extend base_colors to K_{n+1}.
        # But try_extend only needs the existing base_colors.
        
        ext = try_extend(colors, n, base_colors)
        if ext is not None:
            # Update colors
            colors.extend(ext)
            # Update base_colors
            # Add row/col for new vertex n
            new_row = [-1] * (n + 1)
            for i in range(n):
                new_row[i] = ext[i]
                new_row[n] = ext[i] # Symmetric, but we only use i<j
            base_colors.append(new_row)
            for i in range(n):
                base_colors[i][n] = ext[i]
            
            n += 1
            best_n = n
            best_colors = colors[:]
        else:
            break

    return {'n': best_n, 'colors': best_colors}
