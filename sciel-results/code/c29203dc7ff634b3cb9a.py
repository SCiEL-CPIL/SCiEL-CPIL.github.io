"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Rank-189 decomposition of 6x6 matrix multiplication using
    tensor product of Strassen (2x2, rank 7) and schoolbook (3x3, rank 27).
    
    The 6x6 matrix is viewed as a 2x2 block matrix with 3x3 blocks.
    The tensor product decomposition gives rank = 7 * 27 = 189.
    """
    n, m, p = DIMENSIONS
    rank = 189
    
    # Strassen's 7 multiplication formulas for 2x2 block matrices
    # For 2x2 matrices A, B, C where C = A*B:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Strassen factors for 2x2 case
    # u2[t][i,j]: input A factor for Strassen step t
    # v2[t][j,k]: input B factor for Strassen step t  
    # w2[t][k,i]: output C factor for Strassen step t
    
    # i,j,k are block indices (0 or 1)
    # u2[t][i,j] is coefficient for A[i,j] in Strassen multiplication t
    u2 = [
        # M1: (A11+A22)*(B11+B22)
        [[1,0],[1,0]],  # A11 coeff: 1, A12 coeff: 0; A21 coeff: 0, A22 coeff: 1
        # M2: (A21+A22)*B11
        [[0,0],[1,1]],
        # M3: A11*(B12-B22)
        [[1,0],[0,0]],
        # M4: A22*(B21-B11)
        [[0,0],[0,1]],
        # M5: (A11+A12)*B22
        [[1,1],[0,0]],
        # M6: (A21-A11)*(B11+B12)
        [[-1,0],[1,0]],
        # M7: (A12-A22)*(B21+B22)
        [[0,1],[0,-1]],
    ]
    
    # v2[t][j,k] is coefficient for B[j,k] in Strassen multiplication t
    v2 = [
        # M1: (A11+A22)*(B11+B22)
        [[1,1],[0,1]],  # B11 coeff: 1, B12 coeff: 1; B21 coeff: 0, B22 coeff: 1
        # M2: (A21+A22)*B11
        [[1,0],[0,0]],
        # M3: A11*(B12-B22)
        [[0,1],[0,-1]],
        # M4: A22*(B21-B11)
        [[-1,0],[1,0]],
        # M5: (A11+A12)*B22
        [[0,0],[0,1]],
        # M6: (A21-A11)*(B11+B12)
        [[1,1],[0,0]],
        # M7: (A12-A22)*(B21+B22)
        [[0,0],[1,1]],
    ]
    
    # w2[t][k,i] is coefficient for output C[i,k] from Strassen multiplication t
    # Note: w2[t][i,k] gives contribution to C[i,k]
    w2 = [
        # M1: contributes to C11 (+1) and C22 (+1)
        [[1,0],[0,1]],
        # M2: contributes to C21 (+1)
        [[0,0],[1,0]],
        # M3: contributes to C12 (+1)
        [[0,1],[0,0]],
        # M4: contributes to C11 (+1) and C21 (+1)
        [[1,0],[1,0]],
        # M5: contributes to C11 (-1) and C12 (+1)
        [[-1,1],[0,0]],
        # M6: contributes to C22 (+1)
        [[0,0],[0,1]],
        # M7: contributes to C11 (+1) and C22 (+1)
        [[1,0],[0,1]],
    ]
    
    # Schoolbook decomposition for 3x3 case
    # For 3x3 matrices, rank = 27
    # Each term (i,j,k) has u3=1 for A[i,j], v3=1 for B[j,k], w3=1 for C[i,k]
    
    # Now construct the full 6x6 decomposition
    # i_full = 2*I + I3, j_full = 2*J + J3, k_full = 2*K + K3
    # where I,J,K in {0,1} are block indices, I3,J3,K3 in {0,1,2} are within-block indices
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(7):  # Strassen step
        for I3 in range(3):  # within-block row of A
            for J3 in range(3):  # within-block col of A / row of B
                for K3 in range(3):  # within-block col of B
                    # For each (t, I3, J3, K3), we have one rank-1 term
                    # U index: i_full * m + j_full = (2*I + I3) * 6 + (2*J + J3)
                    # V index: j_full * p + k_full = (2*J + J3) * 6 + (2*K + K3)
                    # W index: k_full * n + i_full = (2*K + K3) * 6 + (2*I + I3)
                    
                    for I in range(2):  # block row of A
                        for J in range(2):  # block col of A / row of B
                            for K in range(2):  # block col of B
                                # U coefficient: u2[t][I,J] * 1 (schoolbook within-block is 1)
                                u_val = u2[t][I][J]
                                if u_val == 0:
                                    continue
                                # V coefficient: v2[t][J,K] * 1
                                v_val = v2[t][J][K]
                                if v_val == 0:
                                    continue
                                # W coefficient: w2[t][K,I] * 1
                                w_val = w2[t][K][I]
                                if w_val == 0:
                                    continue
                                
                                i_full = 2 * I + I3
                                j_full = 2 * J + J3
                                k_full = 2 * K + K3
                                
                                U[i_full * m + j_full][col] = u_val
                                V[j_full * p + k_full][col] = v_val
                                W[k_full * n + i_full][col] = w_val
                    
                    col += 1
    
    # Verify: col should be 7*3*3*3 = 189
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
