"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
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
        
    # Base K32 coloring
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    
    # Attempt to extend to K33
    # We need to assign colors to edges (32, j) for j in 0..31
    # Let v = 32.
    # V1 = 0..15, V2 = 16..31.
    # Edges in V1 use colors 0,1,2. Edges in V2 use colors 0,1,2. Edges V1-V2 use color 3.
    
    # Build adjacency lists for G0, G1, G2 in V1
    adj = [[set() for _ in range(3)] for _ in range(16)]
    for i in range(16):
        for j in range(i+1, 16):
            c = log[i ^ j] % 3
            adj[i][c].add(j)
            adj[j][c].add(i)
            
    # CSP for V1: Assign c_u in {0,1,2} for u in 0..15
    # Constraint: If c_u == k and c_w == k, then w not in adj[u][k]
    
    def solve_v1():
        # Backtracking
        # Order vertices by degree in some heuristic? Just 0..15
        colors_v1 = [-1] * 16
        def backtrack(idx):
            if time.time() > deadline: return None
            if idx == 16: return colors_v1[:]
            for c in [0, 1, 2]:
                # Check constraints with previous vertices
                valid = True
                for u in range(idx):
                    if colors_v1[u] == c:
                        if u in adj[idx][c]:
                            valid = False
                            break
                if valid:
                    colors_v1[idx] = c
                    res = backtrack(idx + 1)
                    if res: return res
                    colors_v1[idx] = -1
            return None
        return backtrack(0)

    sol_v1 = solve_v1()
    
    if sol_v1 is None:
        return {'n': 32, 'colors': colors}
        
    # Now solve for V2
    # V2 vertices are 16..31. Map to 0..15 for adj reuse (isomorphic)
    # v connects to V2 with colors {0,1,2,3}
    # Constraint: For c in 0,1,2, the set of vertices in V2 with color c must be independent in G_c.
    # Color 3 has no internal constraint in V2 (edges within V2 are 0,1,2).
    # Also, no inter-block triangle constraint issues if S3 (V1 color 3) is empty.
    # In our V1 solution, colors are 0,1,2. So S3 is empty.
    # So we can use color 3 for any subset of V2.
    
    def solve_v2():
        colors_v2 = [-1] * 16
        def backtrack(idx):
            if time.time() > deadline: return None
            if idx == 16: return colors_v2[:]
            for c in [0, 1, 2, 3]:
                valid = True
                if c < 3:
                    for u in range(idx):
                        if colors_v2[u] == c:
                            if u in adj[idx][c]:
                                valid = False
                                break
                if valid:
                    colors_v2[idx] = c
                    res = backtrack(idx + 1)
                    if res: return res
                    colors_v2[idx] = -1
            return None
        return backtrack(0)

    sol_v2 = solve_v2()
    
    if sol_v2 is None:
        return {'n': 32, 'colors': colors}
        
    # Construct K33
    new_colors = colors[:]
    # Add edges from 32 to 0..15
    for i in range(16):
        new_colors.append(sol_v1[i])
    # Add edges from 32 to 16..31
    for i in range(16):
        new_colors.append(sol_v2[i])
        
    return {'n': 33, 'colors': new_colors}

# EVOLVE-BLOCK-END
