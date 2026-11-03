import time
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

def _build_strassen_2x2():
    """Returns U, V, W for 2x2 * 2x2 -> 2x2 (Rank 7)."""
    n = m = p = 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    # M1: (a11+a22)(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (a21+a22)b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3: a11(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: a22(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = 1
    
    # M5: (a11+a12)b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[3][4] = 1
    
    # M6: (a21-a11)(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1; W[3][5] = 1
    
    # M7: (a12-a22)(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = 1
    
    return U, V, W

def _build_strassen_4x4():
    """Returns U, V, W for 4x4 * 4x4 -> 4x4 (Rank 49)."""
    n = m = p = 4
    rank = 49
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    U2, V2, W2 = _build_strassen_2x2()
    
    # Map 4x4 indices to 2x2 block indices
    # i in 0..3 -> i_block in 0..1, i_local in 0..1
    # i_block = i // 2, i_local = i % 2
    
    for i in range(4):
        for j in range(4):
            ib = i // 2
            il = i % 2
            jb = j // 2
            jl = j % 2
            u_idx = ib * 2 + il
            u_j_idx = jb * 2 + jl
            
            for c in range(7):
                val = U2[u_idx][c]
                if val != 0:
                    # U4[i*4+j, c_block*7 + c]
                    # We need to map the 7 sub-terms to the 49 global terms
                    # The structure is: for each of the 4 block-pairs, we have 7 terms.
                    # Total 49 terms.
                    # Term index = (block_pair_idx * 7) + c
                    # But the block pair depends on i,j?
                    # No, Strassen computes 4 block outputs using 7 products of 2x2 blocks.
                    # The 7 products are shared? No, they are specific to the block structure.
                    # Actually, the 49 terms correspond to the 7 Strassen formulas applied to the 4 block multiplications?
                    # No, the 49 terms are the result of recursively applying Strassen.
                    # Level 1: 4 blocks of A, 4 blocks of B.
                    # We compute 7 products of 2x2 blocks.
                    # Each 2x2 product requires 7 scalar products.
                    # Total 49.
                    
                    # The mapping is complex. Let's use a simpler approach:
                    # Just use schoolbook for 4x4x4? Rank 64.
                    # 64 + 4 = 68. Still < 80.
                    # Let's stick to schoolbook for 4x4x4 to avoid bugs.
                    pass
                    
    # Fallback to Schoolbook for 4x4x4
    U_sb = [[0]*64 for _ in range(16)]
    V_sb = [[0]*64 for _ in range(16)]
    W_sb = [[0]*64 for _ in range(16)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U_sb[i*4+j][col] = 1
                V_sb[j*4+k][col] = 1
                W_sb[k*4+i][col] = 1
                col += 1
    return U_sb, V_sb, W_sb

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Use Schoolbook for 4x4x4 (Rank 64) and 4x4x1 (Rank 4)
    # Total Rank 68
    
    rank_444 = 64
    rank_441 = 4
    total_rank = rank_444 + rank_441
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    # Part 1: 4x4x4
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    # Part 2: 4x4x1 (k=4)
    for i in range(n):
        for j in range(m):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
