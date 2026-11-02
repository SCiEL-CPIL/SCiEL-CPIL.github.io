import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen for the 2x2 block structure (7 steps)
    # Use a Rank 26 decomposition for the 3x3 block multiplication
    
    # Rank of the inner 3x3 multiplication
    inner_rank = 26
    total_rank = 7 * inner_rank
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Define Strassen Coefficients for 2x2 blocks ---
    # beta[s][I][J], gamma[s][K][L], alpha[s][I][K]
    beta = [[ [0]*2 for _ in range(2) ] for _ in range(7)]
    gamma = [[ [0]*2 for _ in range(2) ] for _ in range(7)]
    alpha = [[ [0]*2 for _ in range(2) ] for _ in range(7)]
    
    # s=0: (A11+A22)(B11+B22) -> C11+=M, C22+=M
    beta[0][0][0] = 1; beta[0][1][1] = 1
    gamma[0][0][0] = 1; gamma[0][1][1] = 1
    alpha[0][0][0] = 1; alpha[0][1][1] = 1
    
    # s=1: (A21+A22)B11 -> C21+=M, C22-=M
    beta[1][1][0] = 1; beta[1][1][1] = 1
    gamma[1][0][0] = 1
    alpha[1][1][0] = 1; alpha[1][1][1] = -1
    
    # s=2: A11(B12-B22) -> C12+=M, C22+=M
    beta[2][0][0] = 1
    gamma[2][0][1] = 1; gamma[2][1][1] = -1
    alpha[2][0][1] = 1; alpha[2][1][1] = 1
    
    # s=3: A22(B21-B11) -> C11+=M, C21+=M
    beta[3][1][1] = 1
    gamma[3][1][0] = 1; gamma[3][0][0] = -1
    alpha[3][0][0] = 1; alpha[3][1][0] = 1
    
    # s=4: (A11+A12)B22 -> C11-=M, C12+=M
    beta[4][0][0] = 1; beta[4][0][1] = 1
    gamma[4][1][1] = 1
    alpha[4][0][0] = -1; alpha[4][0][1] = 1
    
    # s=5: (A21-A11)(B11+B12) -> C22+=M
    beta[5][1][0] = 1; beta[5][0][0] = -1
    gamma[5][0][0] = 1; gamma[5][0][1] = 1
    alpha[5][1][1] = 1
    
    # s=6: (A12-A22)(B21+B22) -> C11+=M
    beta[6][0][1] = 1; beta[6][1][1] = -1
    gamma[6][1][0] = 1; gamma[6][1][1] = 1
    alpha[6][0][0] = 1

    # --- Define Rank 26 Decomposition for 3x3 Matmul ---
    # Indices for 3x3: 0,1,2
    # 2x2 block: (0,1)x(0,1)
    # Vectors/Scalars: (0,1)x2, 2x(0,1), 2x2
    
    inner_U = [[0]*inner_rank for _ in range(9)]
    inner_V = [[0]*inner_rank for _ in range(9)]
    inner_W = [[0]*inner_rank for _ in range(9)]
    
    k = 0
    
    # 1. A00 B00 (2x2 * 2x2) using Strassen (Rank 7)
    # Strassen coeffs for 2x2
    s_beta = [[1,1,1,1,1,-1,1], [1,-1,-1,1,1,1,-1], [1,1,1,-1,-1,-1,1], [1,-1,1,1,1,-1,1]] # Not quite, need standard
    # Standard Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Indices in 2x2: 0,1
    # Map to 3x3: 0->0, 1->1
    
    strassen_U = [
        [1,0, 1,0, 1,0, 0,0, 1,0, 0,0, 0,0], # A11+A22 etc? No, U is for A.
        # U[0][0]=1, U[1][1]=1 for M1?
        # Let's just hardcode the 7 terms for 2x2 A matrix (4 entries)
    ]
    # A entries: A00, A01, A10, A11 (local 2x2)
    # Term 1: A00 + A11
    inner_U[0][k] = 1; inner_U[1][k] = 0; inner_U[2][k] = 0; inner_U[3][k] = 1 # A00, A01, A10, A11? No, flat index.
    # Flat index 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    inner_U[0][k] = 1; inner_U[3][k] = 1
    inner_V[0][k] = 1; inner_V[3][k] = 1 # B00 + B11
    inner_W[0][k] = 1; inner_W[3][k] = 1 # C00 += M, C11 += M (local)
    k += 1
    
    # Term 2: (A10+A11) B00
    inner_U[2][k] = 1; inner_U[3][k] = 1
    inner_V[0][k] = 1
    inner_W[2][k] = 1; inner_W[3][k] = -1 # C10 += M, C11 -= M
    k += 1
    
    # Term 3: A00 (B01 - B11)
    inner_U[0][k] = 1
    inner_V[1][k] = 1; inner_V[3][k] = -1
    inner_W[1][k] = 1; inner_W[3][k] = 1 # C01 += M, C11 += M
    k += 1
    
    # Term 4: A11 (B10 - B00)
    inner_U[3][k] = 1
    inner_V[2][k] = 1; inner_V[0][k] = -1
    inner_W[0][k] = 1; inner_W[2][k] = 1 # C00 += M, C10 += M
    k += 1
    
    # Term 5: (A00 + A01) B11
    inner_U[0][k] = 1; inner_U[1][k] = 1
    inner_V[3][k] = 1
    inner_W[0][k] = -1; inner_W[1][k] = 1 # C00 -= M, C01 += M
    k += 1
    
    # Term 6: (A10 - A00) (B00 + B01)
    inner_U[2][k] = 1; inner_U[0][k] = -1
    inner_V[0][k] = 1; inner_V[1][k] = 1
    inner_W[3][k] = 1 # C11 += M
    k += 1
    
    # Term 7: (A01 - A11) (B10 + B11)
    inner_U[1][k] = 1; inner_U[3][k] = -1
    inner_V[2][k] = 1; inner_V[3][k] = 1
    inner_W[0][k] = 1 # C00 += M
    k += 1
    
    # 2. A01 B10 (2x1 * 1x2) -> Rank 4
    # A01 is col 2, rows 0,1. B10 is row 2, cols 0,1.
    # C00 += A00 B00? No. C00 += A01_0 B10_0.
    inner_U[4][k] = 1 # A(0,2)
    inner_V[4][k] = 1 # B(0,2) ? No, B10 is B(2,0). Index 4 is (2,0).
    inner_W[0][k] = 1 # C(0,0)
    k += 1
    inner_U[4][k] = 1 # A(0,2)
    inner_V[5][k] = 1 # B(1,2)? No, B10 is row 2. Cols 0,1. So B(2,1) is index 5?
    # 3x3 flat: 0,1,2 / 3,4,5 / 6,7,8
    # A01: (0,2)=2, (1,2)=5.
    # B10: (2,0)=6, (2,1)=7.
    # C00 += A(0,2)B(2,0)
    # C01 += A(0,2)B(2,1)
    # C10 += A(1,2)B(2,0)
    # C11 += A(1,2)B(2,1)
    
    # Redo A01 B10
    inner_U[2][k] = 1 # A(0,2)
    inner_V[6][k] = 1 # B(2,0)
    inner_W[0][k] = 1 # C(0,0)
    k += 1
    inner_U[2][k] = 1 # A(0,2)
    inner_V[7][k] = 1 # B(2,1)
    inner_W[1][k] = 1 # C(0,1)
    k += 1
    inner_U[5][k] = 1 # A(1,2)
    inner_V[6][k] = 1 # B(2,0)
    inner_W[3][k] = 1 # C(1,0)
    k += 1
    inner_U[5][k] = 1 # A(1,2)
    inner_V[7][k] = 1 # B(2,1)
    inner_W[4][k] = 1 # C(1,1)
    k += 1
    
    # 3. A00 B01 (2x2 * 2x1) -> Rank 4
    # A00: (0,0)=0, (0,1)=1, (1,0)=3, (1,1)=4.
    # B01: (0,1)=1, (1,1)=4.
    # C01 += A00 B01.
    # C01(0,1) += A00(0,0)B01(0,1) + A00(0,1)B01(1,1)
    # This is standard 2x2 * 2x1.
    # Terms:
    # C(0,1) += A(0,0)B(0,1)
    inner_U[0][k] = 1; inner_V[1][k] = 1; inner_W[1][k] = 1
    k += 1
    # C(0,1) += A(0,1)B(1,1)
    inner_U[1][k] = 1; inner_V[4][k] = 1; inner_W[1][k] = 1
    k += 1
    # C(1,1) += A(1,0)B(0,1)
    inner_U[3][k] = 1; inner_V[1][k] = 1; inner_W[4][k] = 1
    k += 1
    # C(1,1) += A(1,1)B(1,1)
    inner_U[4][k] = 1; inner_V[4][k] = 1; inner_W[4][k] = 1
    k += 1
    
    # 4. A01 B11 (2x1 * 1x1) -> Rank 2
    # A01: (0,2)=2, (1,2)=5. B11: (2,2)=8.
    # C02 += A(0,2)B(2,2)
    inner_U[2][k] = 1; inner_V[8][k] = 1; inner_W[2][k] = 1
    k += 1
    # C12 += A(1,2)B(2,2)
    inner_U[5][k] = 1; inner_V[8][k] = 1; inner_W[5][k] = 1
    k += 1
    
    # 5. A10 B00 (1x2 * 2x2) -> Rank 4
    # A10: (2,0)=6, (2,1)=7. B00: (0,0)=0, (0,1)=1, (1,0)=3, (1,1)=4.
    # C20 += A(2,0)B(0,0)
    inner_U[6][k] = 1; inner_V[0][k] = 1; inner_W[6][k] = 1
    k += 1
    # C20 += A(2,1)B(1,0)
    inner_U[7][k] = 1; inner_V[3][k] = 1; inner_W[6][k] = 1
    k += 1
    # C21 += A(2,0)B(0,1)
    inner_U[6][k] = 1; inner_V[1][k] = 1; inner_W[7][k] = 1
    k += 1
    # C21 += A(2,1)B(1,1)
    inner_U[7][k] = 1; inner_V[4][k] = 1; inner_W[7][k] = 1
    k += 1
    
    # 6. A11 B10 (1x1 * 1x2) -> Rank 2
    # A11: (2,2)=8. B10: (2,0)=6, (2,1)=7.
    # C20 += A(2,2)B(2,0)
    inner_U[8][k] = 1; inner_V[6][k] = 1; inner_W[6][k] = 1
    k += 1
    # C21 += A(2,2)B(2,1)
    inner_U[8][k] = 1; inner_V[7][k] = 1; inner_W[7][k] = 1
    k += 1
    
    # 7. A10 B01 (1x2 * 2x1) -> Rank 2
    # A10: (2,0)=6, (2,1)=7. B01: (0,1)=1, (1,1)=4.
    # C22 += A(2,0)B(0,1) + A(2,1)B(1,1)
    # Wait, C22 is (2,2).
    # C(2,2) += A(2,0)B(0,1)
    inner_U[6][k] = 1; inner_V[1][k] = 1; inner_W[8][k] = 1
    k += 1
    # C(2,2) += A(2,1)B(1,1)
    inner_U[7][k] = 1; inner_V[4][k] = 1; inner_W[8][k] = 1
    k += 1
    
    # 8. A11 B11 (1x1 * 1x1) -> Rank 1
    # C22 += A(2,2)B(2,2)
    inner_U[8][k] = 1; inner_V[8][k] = 1; inner_W[8][k] = 1
    k += 1
    
    if k != inner_rank:
        raise ValueError(f"Inner rank mismatch: {k} vs {inner_rank}")

    # --- Combine Strassen (2x2) and Inner (3x3) ---
    for s in range(7):
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    # This loop structure in the original code was for 27 terms.
                    # Here we iterate over the 26 inner terms.
                    pass
    
    # Correct Combination Logic:
    # For each Strassen step s (0..6):
    # For each inner term t (0..25):
    #   Global Rank Index r = s * 26 + t
    #   U: For each I,J in 2x2 block:
    #       if beta[s][I][J] != 0:
    #           global_i = I*3 + a? No, a,b,c are not defined here.
    #           The inner term t defines the local indices in the 3x3 block.
    #           We need to map the local 3x3 indices to global 6x6 indices.
            
    # Let's re-fill U,V,W properly.
    for s in range(7):
        for t in range(inner_rank):
            r = s * inner_rank + t
            
            # Extract local indices from inner_U[t], inner_V[t], inner_W[t]
            # inner_U[t] is a list of 9 values.
            local_U = inner_U[t]
            local_V = inner_V[t]
            local_W = inner_W[t]
            
            # Map to Global
            for I in range(2):
                for J in range(2):
                    if beta[s][I][J] != 0:
                        # Local A index is determined by which entries are non-zero in local_U
                        # But local_U is fixed for term t.
                        # We need to add beta[s][I][J] * local_U[local_idx] to U[global_idx][r]
                        
                        # local_U is a vector of 9.
                        # global_i = I * 3 + local_i
                        # global_j = J * 3 + local_j
                        
                        for li in range(3):
                            for lj in range(3):
                                l_idx = li * 3 + lj
                                if local_U[l_idx] != 0:
                                    g_i = I * 3 + li
                                    g_j = J * 3 + lj
                                    U[g_i * m + g_j][r] += beta[s][I][J] * local_U[l_idx]
            
            for K in range(2):
                for L in range(2):
                    if gamma[s][K][L] != 0:
                        for lk in range(3):
                            for ll in range(3):
                                l_idx = lk * 3 + ll
                                if local_V[l_idx] != 0:
                                    g_k = K * 3 + lk
                                    g_l = L * 3 + ll
                                    V[g_k * p + g_l][r] += gamma[s][K][L] * local_V[l_idx]
                                    
            for I in range(2):
                for K in range(2):
                    if alpha[s][I][K] != 0:
                        for li in range(3):
                            for lk in range(3):
                                l_idx = li * 3 + lk
                                if local_W[l_idx] != 0:
                                    g_i = I * 3 + li
                                    g_k = K * 3 + lk
                                    W[g_k * n + g_i][r] += alpha[s][I][K] * local_W[l_idx]
                                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
