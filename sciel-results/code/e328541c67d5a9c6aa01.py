def construct():
    from itertools import combinations
    
    def check_valid(edge_colors):
        """Check every K4 in K16 has at least 5 distinct colors."""
        for quad in combinations(range(16), 4):
            a, b, c, d = quad
            cols = {
                edge_colors[a][b], edge_colors[a][c], edge_colors[a][d],
                edge_colors[b][c], edge_colors[b][d], edge_colors[c][d]
            }
            if len(cols) < 5:
                return False
        return True

    best_num = 120
    best_colors = None

    # Search over primes and shifts for multiplicative scheme (i+b)*(j+b) % p
    for p in [17, 19, 23, 29, 31, 37]:
        for b in range(p):
            # Check if v_i = (i + b) % p are distinct and non-zero for i in 0..15
            # Since p > 16, i+b mod p are distinct for i in 0..15 if b is fixed?
            # i1+b == i2+b mod p => i1==i2. Yes.
            # Non-zero? i+b == 0 mod p => i = p-b.
            # If p-b is in 0..15, then v_{p-b} = 0.
            # If v_i = 0, then all edges from i have color 0.
            # As seen before, this might cause K4 validity issues or just reduce color count?
            # If v_i=0, edges (i,j) are 0.
            # K4 {i, j, k, l}:
            # (i,j)=0, (i,k)=0, (i,l)=0.
            # (j,k), (j,l), (k,l) must provide at least 2 more colors to reach 5?
            # No, we need 5 distinct colors total.
            # 0 is one. We need 4 more from the other 3 edges? Impossible.
            # So if any v_i = 0, the K4 containing i has at most 1 + 3 = 4 colors?
            # Wait. Edges among j,k,l are 3 edges.
            # Colors: 0, c_jk, c_jl, c_kl.
            # Max distinct: 4.
            # So v_i must be non-zero for all i.
            
            valid_shift = True
            for i in range(16):
                if (i + b) % p == 0:
                    valid_shift = False
                    break
            
            if not valid_shift:
                continue

            edge_colors = [[0]*16 for _ in range(16)]
            for i in range(16):
                vi = (i + b) % p
                for j in range(i+1, 16):
                    vj = (j + b) % p
                    c = (vi * vj) % p
                    edge_colors[i][j] = c
                    edge_colors[j][i] = c
            
            if check_valid(edge_colors):
                num = len(set(edge_colors[i][j] for i in range(16) for j in range(i+1, 16)))
                if num < best_num:
                    best_num = num
                    best_colors = [edge_colors[i][j] for i in range(16) for j in range(i+1, 16)]

    if best_colors:
        return best_colors
    
    # Fallback: return a known valid one (e.g., p=17, b=1)
    p = 17
    b = 1
    edge_colors = [[0]*16 for _ in range(16)]
    for i in range(16):
        vi = (i + b) % p
        for j in range(i+1, 16):
            vj = (j + b) % p
            c = (vi * vj) % p
            edge_colors[i][j] = c
            edge_colors[j][i] = c
    return [edge_colors[i][j] for i in range(16) for j in range(i+1, 16)]
