"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """Construct the rank-minimal decomposition for 2x4 times 4x6 using Strassen-like decomposition.
    
    Rank = 42.
    """
    n, m, p = DIMENSIONS
    rank = 42
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # M1: (a+d)(e+h) -> C00 += 1, C11 += 1
    # M2: (c+d)e   -> C10 += 1
    # M3: a(f-h)    -> C01 += 1, C11 += 1
    # M4: d(g-e)    -> C00 += 1, C10 += 1
    # M5: (a+b)h    -> C00 -= 1, C01 += 1
    # M6: (c-a)(e+f)-> C11 += 1
    # M7: (b-d)(g+h)-> C00 += 1
    
    # Note: C indices are (i, k) where i in {0,1}, k in {0,1} local.
    # W index is k*n + i.
    
    strassen_data = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U: [(local_i, local_j, coeff), ...]
        # V: [(local_j, local_k, coeff), ...]
        # W: [(local_k, local_i, coeff), ...]
        
        # M1: (a+d)(e+h)
        {
            'U': [(0, 0, 1.0), (1, 0, 1.0)],
            'V': [(0, 0, 1.0), (0, 1, 1.0)],
            'W': [(0, 0, 1.0), (1, 1, 1.0)]
        },
        # M2: (c+d)e
        {
            'U': [(1, 0, 1.0), (1, 1, 1.0)],
            'V': [(0, 0, 1.0)],
            'W': [(0, 1, 1.0)]
        },
        # M3: a(f-h)
        {
            'U': [(0, 0, 1.0)],
            'V': [(0, 1, 1.0), (1, 1, -1.0)],
            'W': [(1, 0, 1.0), (1, 1, 1.0)]
        },
        # M4: d(g-e)
        {
            'U': [(1, 1, 1.0)],
            'V': [(1, 0, 1.0), (0, 0, -1.0)],
            'W': [(0, 0, 1.0), (0, 1, 1.0)]
        },
        # M5: (a+b)h
        {
            'U': [(0, 0, 1.0), (0, 1, 1.0)],
            'V': [(1, 1, 1.0)],
            'W': [(0, 0, -1.0), (1, 0, 1.0)]
        },
        # M6: (c-a)(e+f)
        {
            'U': [(1, 0, 1.0), (0, 0, -1.0)],
            'V': [(0, 0, 1.0), (0, 1, 1.0)],
            'W': [(1, 1, 1.0)]
        },
        # M7: (b-d)(g+h)
        {
            'U': [(0, 1, 1.0), (1, 1, -1.0)],
            'V': [(1, 0, 1.0), (1, 1, 1.0)],
            'W': [(0, 0, 1.0)]
        }
    ]
    
    col = 0
    # Process A1 (cols 0,1) * B1 (rows 0,1)
    # B1 has 6 cols, split into 3 blocks of 2
    for j_offset in [0, 2]:
        for k_offset in [0, 2, 4]:
            for term in strassen_data:
                # Add to U
                for li, lj, coeff in term['U']:
                    global_i = li
                    global_j = j_offset + lj
                    U[global_i * m + global_j][col] += coeff
                
                # Add to V
                for lj, lk, coeff in term['V']:
                    global_j = j_offset + lj
                    global_k = k_offset + lk
                    V[global_j * p + global_k][col] += coeff
                
                # Add to W
                for lk, li, coeff in term['W']:
                    global_k = k_offset + lk
                    global_i = li
                    W[global_k * n + global_i][col] += coeff
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
