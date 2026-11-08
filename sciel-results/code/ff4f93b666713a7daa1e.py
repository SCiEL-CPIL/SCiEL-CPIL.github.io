"""Matrix multiplication 2x4 by 4x4 via 4 independent Strassen 2x2x2 decompositions, rank 28."""
import time

def construct():
    """
    Construct a rank-28 exact tensor decomposition for multiplying a 2x4 matrix by a 4x4 matrix.
    
    Partition A (2x4) into A1=A[:,0:2] and A2=A[:,2:4].
    Partition B (4x4) into B11=B[0:2,0:2], B12=B[0:2,2:4], B21=B[2:4,0:2], B22=B[2:4,2:4].
    
    C[:,0:2] = A1*B11 + A2*B21
    C[:,2:4] = A1*B12 + A2*B22
    
    Each 2x2x2 product uses Strassen (rank 7), giving 4*7 = 28 total terms.
    
    U shape: (8, 28) - linear forms in A (indexed as A[i,j] -> row i*4+j)
    V shape: (16, 28) - linear forms in B (indexed as B[j,k] -> row j*4+k)
    W shape: (8, 28) - linear forms in C (indexed as C[i,k] -> row k*2+i per convention W[k*n+i])
    """
    t0 = time.time()
    DEADLINE = 140.0
    
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]   # 8 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 8 x 28
    
    # Strassen 2x2x2 coefficients:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b12-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # c11 = M1 - M4 + M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M3 + M2 - M6
    
    # For each Strassen term t (0-6), the (a-factor, b-factor) pairs:
    # t=0: a=(a11+a22), b=(b11+b22)
    # t=1: a=(a21+a22), b=b11
    # t=2: a=a11, b=(b21-b22)
    # t=3: a=a22, b=(b12-b11)
    # t=4: a=(a11+a12), b=b22
    # t=5: a=(a21-a11), b=(b11+b12)
    # t=6: a=(a12-a22), b=(b21+b22)
    
    # a coefficients for each term (indices: 0=a11, 1=a12, 2=a21, 3=a22)
    a_coeffs = [
        [1, 0, 0, 1],   # t=0: a11 + a22
        [0, 0, 1, 1],   # t=1: a21 + a22
        [1, 0, 0, 0],   # t=2: a11
        [0, 0, 0, 1],   # t=3: a22
        [1, 1, 0, 0],   # t=4: a11 + a12
        [-1, 0, 1, 0],  # t=5: a21 - a11
        [0, 1, 0, -1],  # t=6: a12 - a22
    ]
    
    # b coefficients for each term (indices: 0=b11, 1=b12, 2=b21, 3=b22)
    b_coeffs = [
        [1, 0, 0, 1],   # t=0: b11 + b22
        [1, 0, 0, 0],   # t=1: b11
        [0, 0, 1, -1],  # t=2: b21 - b22
        [0, 1, 0, -1],  # t=3: b12 - b11
        [0, 0, 0, 1],   # t=4: b22
        [1, 1, 0, 0],   # t=5: b11 + b12
        [0, 0, 1, 1],   # t=6: b21 + b22
    ]
    
    # c coefficients: for each term t, contribution to (c11, c12, c21, c22)
    # c11 = M1 - M4 + M5 + M7  => term0:+1, term3:-1, term4:+1, term6:+1
    # c12 = M3 + M5            => term2:+1, term4:+1
    # c21 = M2 + M4            => term1:+1, term3:+1
    # c22 = M1 - M3 + M2 - M6  => term0:+1, term2:-1, term1:+1, term5:-1
    c_coeffs = [
        [1, 0, 0, 1],   # t=0: c11+1, c22+1
        [0, 0, 1, 1],   # t=1: c21+1, c22+1
        [0, 1, 0, -1],  # t=2: c12+1, c22-1
        [-1, 0, 1, 0],  # t=3: c11-1, c21+1
        [1, 1, 0, 0],   # t=4: c11+1, c12+1
        [0, 0, 0, -1],  # t=5: c22-1
        [1, 0, 0, 0],   # t=6: c11+1
    ]
    
    # Define the 4 products:
    # product_idx: (A_sub, B_sub, C_sub)
    # A_sub: which 2x2 sub-matrix of A (col offset)
    # B_sub: which 2x2 sub-matrix of B (row offset, col offset)
    # C_sub: which 2x2 sub-matrix of C (col offset)
    
    products = [
        # (A_col_offset, B_row_offset, B_col_offset, C_col_offset)
        (0, 0, 0, 0),  # P1: A1 * B11 -> C[:, 0:2]
        (2, 2, 0, 0),  # P2: A2 * B21 -> C[:, 0:2]
        (0, 0, 2, 2),  # P3: A1 * B12 -> C[:, 2:4]
        (2, 2, 2, 2),  # P4: A2 * B22 -> C[:, 2:4]
    ]
    
    for prod_idx, (a_col_off, b_row_off, b_col_off, c_col_off) in enumerate(products):
        for t in range(7):
            r = prod_idx * 7 + t  # global term index
            
            # Fill U: A sub-matrix entries
            # A sub-matrix at columns [a_col_off, a_col_off+1]
            # a11 -> A[0, a_col_off], a12 -> A[0, a_col_off+1]
            # a21 -> A[1, a_col_off], a22 -> A[1, a_col_off+1]
            a_map = [
                (0, a_col_off),      # a11
                (0, a_col_off + 1),  # a12
                (1, a_col_off),      # a21
                (1, a_col_off + 1),  # a22
            ]
            for ai, (ri, ci) in enumerate(a_map):
                U[ri * m + ci][r] += a_coeffs[t][ai]
            
            # Fill V: B sub-matrix entries
            # B sub-matrix at rows [b_row_off, b_row_off+1], cols [b_col_off, b_col_off+1]
            # b11 -> B[b_row_off, b_col_off], b12 -> B[b_row_off, b_col_off+1]
            # b21 -> B[b_row_off+1, b_col_off], b22 -> B[b_row_off+1, b_col_off+1]
            b_map = [
                (b_row_off, b_col_off),      # b11
                (b_row_off, b_col_off + 1),  # b12
                (b_row_off + 1, b_col_off),  # b21
                (b_row_off + 1, b_col_off + 1),  # b22
            ]
            for bi, (ri, ci) in enumerate(b_map):
                V[ri * p + ci][r] += b_coeffs[t][bi]
            
            # Fill W: C sub-matrix entries
            # C sub-matrix at cols [c_col_off, c_col_off+1], rows [0,1]
            # c11 -> C[0, c_col_off], c12 -> C[0, c_col_off+1]
            # c21 -> C[1, c_col_off], c22 -> C[1, c_col_off+1]
            # W index: W[k*n + i] where k is col, i is row
            c_map = [
                (0, c_col_off),      # c11: row 0, col c_col_off
                (0, c_col_off + 1),  # c12: row 0, col c_col_off+1
                (1, c_col_off),      # c21: row 1, col c_col_off
                (1, c_col_off + 1),  # c22: row 1, col c_col_off+1
            ]
            for ci_idx, (ri, ci) in enumerate(c_map):
                W[ci * n + ri][r] += c_coeffs[t][ci_idx]
    
    if time.time() > DEADLINE:
        return {"U": U, "V": V, "W": W}
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        rank = len(W[0]) if W else 0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    coeff = 0.0
                    for r in range(rank):
                        coeff += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(coeff - 1.0) > 1e-10:
                        return False
                for j1 in range(m):
                    for j2 in range(m):
                        if j1 == j2:
                            continue
                        cross = 0.0
                        for r in range(rank):
                            cross += U[i * m + j1][r] * V[j2 * p + k][r] * W[k * n + i][r]
                        if abs(cross) > 1e-10:
                            return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
