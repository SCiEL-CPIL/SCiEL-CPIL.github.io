"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-28 decomposition for M_{2,2,8} using Strassen on 4 disjoint 2x2x2 blocks.
    
    The 2x2 matrix A is multiplied by a 2x8 matrix B, producing a 2x8 matrix C.
    We partition B (and thus C) into 4 blocks of width 2: B[:, 2t:2t+2] for t=0..3.
    Each block multiplication is a 2x2 * 2x2 = 2x2, solved via Strassen (rank 7).
    Total rank = 4 * 7 = 28.
    
    Conventions:
      U has shape (n*m, rank) = (4, 28). Index: i*m + j = 2*i + j.
      V has shape (m*p, rank) = (16, 28). Index: j*p + k = 8*j + k.
      W has shape (p*n, rank) = (16, 28). Index: k*n + i = 2*k + i.
    """
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]   # 4 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 16 x 28
    
    # Strassen formulas for 2x2 * 2x2:
    # Given A = [[a, b], [c, d]] and B = [[e, f], [g, h]] (0-indexed: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1])
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # For each block t (k1 = 2t, k2 = 2t+1), we have 7 terms.
    # A indices: a=U[0], b=U[1], c=U[2], d=U[3] (i.e., 2*0+0, 2*0+1, 2*1+0, 2*1+1)
    # B indices for block t:
    #   e = B[0, k1] -> V[8*0 + k1]
    #   f = B[0, k2] -> V[8*0 + k2]
    #   g = B[1, k1] -> V[8*1 + k1]
    #   h = B[1, k2] -> V[8*1 + k2]
    
    # W indices for block t:
    #   C[0, k1] -> W[2*k1 + 0]
    #   C[0, k2] -> W[2*k2 + 0]
    #   C[1, k1] -> W[2*k1 + 1]
    #   C[1, k2] -> W[2*k2 + 1]
    
    for t in range(4):
        k1 = 2 * t
        k2 = 2 * t + 1
        base = 7 * t  # starting column index for this block's 7 terms
        
        e_idx = 8 * 0 + k1  # V index for B[0, k1]
        f_idx = 8 * 0 + k2  # V index for B[0, k2]
        g_idx = 8 * 1 + k1  # V index for B[1, k1]
        h_idx = 8 * 1 + k2  # V index for B[1, k2]
        
        # W indices
        w_0k1 = 2 * k1 + 0  # C[0, k1]
        w_0k2 = 2 * k2 + 0  # C[0, k2]
        w_1k1 = 2 * k1 + 1  # C[1, k1]
        w_1k2 = 2 * k2 + 1  # C[1, k2]
        
        # M1 = (a+d)(e+h)
        c = base + 0
        U[0][c] += 1
        U[3][c] += 1
        V[e_idx][c] += 1
        V[h_idx][c] += 1
        W[w_0k1][c] += 1
        W[w_1k2][c] += 1
        
        # M2 = (c+d)e
        c = base + 1
        U[2][c] += 1
        U[3][c] += 1
        V[e_idx][c] += 1
        W[w_1k1][c] += 1
        W[w_1k2][c] -= 1
        
        # M3 = a(f-h)
        c = base + 2
        U[0][c] += 1
        V[f_idx][c] += 1
        V[h_idx][c] -= 1
        W[w_0k2][c] += 1
        W[w_1k2][c] += 1
        
        # M4 = d(g-e)
        c = base + 3
        U[3][c] += 1
        V[g_idx][c] += 1
        V[e_idx][c] -= 1
        W[w_0k1][c] += 1
        W[w_1k1][c] += 1
        
        # M5 = (a+b)h
        c = base + 4
        U[0][c] += 1
        U[1][c] += 1
        V[h_idx][c] += 1
        W[w_0k1][c] -= 1
        W[w_0k2][c] += 1
        
        # M6 = (c-a)(e+f)
        c = base + 5
        U[2][c] += 1
        U[0][c] -= 1
        V[e_idx][c] += 1
        V[f_idx][c] += 1
        W[w_1k2][c] += 1
        
        # M7 = (b-d)(g+h)
        c = base + 6
        U[1][c] += 1
        U[3][c] -= 1
        V[g_idx][c] += 1
        V[h_idx][c] += 1
        W[w_0k1][c] += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
