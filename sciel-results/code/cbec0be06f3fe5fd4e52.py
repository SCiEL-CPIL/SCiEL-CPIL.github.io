"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank_per_block = 7
    rank = num_blocks * rank_per_block
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U (A) indices: 00, 01, 10, 11
    # V (B) indices: 00, 01, 10, 11
    # W (C) indices: 00, 01, 10, 11 (k, i)
    
    # M1: (a+d)(e+h)
    # M2: (c+d)e
    # M3: a(f-h)
    # M4: d(g-e)
    # M5: (a+b)h
    # M6: (c-a)(e+f)
    # M7: (b-d)(g+h)
    
    # U factors (row i*m+j)
    U_factors = [
        [(0, 1), (3, 1)],       # M1: a, d
        [(2, 1), (3, 1)],       # M2: c, d
        [(0, 1)],               # M3: a
        [(3, 1)],               # M4: d
        [(0, 1), (1, 1)],       # M5: a, b
        [(0, -1), (2, 1)],      # M6: -a, c
        [(1, 1), (3, -1)]       # M7: b, -d
    ]
    
    # V factors (row j*p+k)
    V_factors = [
        [(0, 1), (3, 1)],       # M1: e, h
        [(0, 1)],               # M2: e
        [(1, 1), (3, -1)],      # M3: f, -h
        [(0, -1), (2, 1)],      # M4: -e, g
        [(3, 1)],               # M5: h
        [(0, 1), (1, 1)],       # M6: e, f
        [(2, 1), (3, 1)]        # M7: g, h
    ]
    
    # W factors (row k*n+i)
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W index: k is 0 or 1 (local), i is 0 or 1
    # Local k=0 -> col 0, k=1 -> col 1
    # W row index = k*n + i
    
    W_factors = [
        [(0, 1), (3, 1)],       # M1: C00+1, C11+1
        [(2, 1), (3, -1)],      # M2: C10+1, C11-1
        [(1, 1), (3, 1)],       # M3: C01+1, C11+1
        [(0, 1), (2, 1)],       # M4: C00+1, C10+1
        [(0, -1), (1, 1)],      # M5: C00-1, C01+1
        [(3, 1)],               # M6: C11+1
        [(0, 1)]                # M7: C00+1
    ]
    
    for l in range(num_blocks):
        base_rank = l * rank_per_block
        base_k = l * 2
        
        for s in range(rank_per_block):
            col = base_rank + s
            
            # Fill U
            for idx, val in U_factors[s]:
                U[idx][col] += val
                
            # Fill V
            for idx, val in V_factors[s]:
                # Local j, k -> Global j, base_k + k
                # idx is j*2 + k_local
                j = idx // 2
                k_local = idx % 2
                global_idx = j * p + (base_k + k_local)
                V[global_idx][col] += val
                
            # Fill W
            for idx, val in W_factors[s]:
                # Local k, i -> Global k, i
                # idx is k_local*n + i
                k_local = idx // n
                i = idx % n
                global_idx = (base_k + k_local) * n + i
                W[global_idx][col] += val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
