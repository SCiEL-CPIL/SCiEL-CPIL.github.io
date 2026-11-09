"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen rank-7 decomposition for 2x2 x 2x2 matrix multiplication.
    # U indices: (i, j) where i in 0..1, j in 0..1
    # V indices: (j, k) where j in 0..1, k in 0..1
    # W indices: (k, i) where k in 0..1, i in 0..1
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A10)B11
    # M6 = (A11 - A00)(B00 + B10)
    # M7 = (A01 - A11)(B01 + B11)
    
    # Each entry is in {-1, 0, 1}
    # U: rows indexed by (i,j) -> i*2+j, columns 0..6
    # V: rows indexed by (j,k) -> j*2+k, columns 0..6
    # W: rows indexed by (k,i) -> k*2+i, columns 0..6
    
    U_strassen = [
        # i=0, j=0: A00 appears in M1(+), M3(+), M5(+), M6(-)
        [1, 0, 1, 0, 1, -1, 0],
        # i=0, j=1: A01 appears in M7(+)
        [0, 0, 0, 0, 0, 0, 1],
        # i=1, j=0: A10 appears in M2(+), M5(+)
        [0, 1, 0, 0, 1, 0, 0],
        # i=1, j=1: A11 appears in M1(+), M2(+), M4(+), M6(+), M7(-)
        [1, 1, 0, 1, 0, 1, -1],
    ]
    
    V_strassen = [
        # j=0, k=0: B00 appears in M1(+), M2(+), M4(-), M6(+)
        [1, 1, 0, -1, 0, 1, 0],
        # j=0, k=1: B01 appears in M4(-), M7(+)
        [0, 0, 0, -1, 0, 0, 1],
        # j=1, k=0: B10 appears in M3(+), M6(+)
        [0, 0, 1, 0, 0, 1, 0],
        # j=1, k=1: B11 appears in M1(+), M3(-), M5(+), M7(+)
        [1, 0, -1, 0, 1, 0, 1],
    ]
    
    W_strassen = [
        # k=0, i=0: C00 = M1 + M4 - M6 - M7
        [1, 0, 0, 1, 0, -1, -1],
        # k=0, i=1: C01 = M5 + M7
        [0, 0, 0, 0, 1, 0, 1],
        # k=1, i=0: C10 = M2 + M3
        [0, 1, 1, 0, 0, 0, 0],
        # k=1, i=1: C11 = M1 - M3 + M5 + M6
        [1, 0, -1, 0, 1, 1, 0],
    ]
    
    # Now build the 2x4 x 4x4 decomposition using block structure.
    # n=2, m=4, p=4
    # View A as 2x4 = (1x2) x (2x2) blocks: A[i, j] where i in 0..1, j in 0..3
    # View B as 4x4 = (2x2) x (2x2) blocks: B[j, k] where j in 0..3, k in 0..3
    # View C as 4x2 = (2x2) x (2x1) blocks: C[k, i] where k in 0..3, i in 0..1
    
    # Decomposition: m=4 = 2*2, so we have 2 groups of 2 columns in A
    # p=4 = 2*2, so we have 2 groups of 2 rows in B
    # n=2 = 1*2, so we have 1 group of 2 rows in output
    
    # The Kronecker structure:
    # A is n x m = 2 x 4. Split m into 2 groups of 2: m = 2 * 2
    # B is m x p = 4 x 4. Split m into 2 groups of 2, p into 2 groups of 2: m = 2*2, p = 2*2
    # C is p x n = 4 x 2. Split p into 2 groups of 2, n into 1 group of 2: p = 2*2, n = 1*2
    
    # For each combination of (group_a, group_b) where group_a indexes the m-split of A 
    # and group_b indexes the m-split of B (they must match for the multiplication to work),
    # we do a 2x2 x 2x2 Strassen multiplication.
    
    # Actually, let me think more carefully about the block structure.
    # A[i, j] for i in {0,1}, j in {0,1,2,3}
    # B[j, k] for j in {0,1,2,3}, k in {0,1,2,3}
    # C[k, i] for k in {0,1,2,3}, i in {0,1}
    
    # C[k, i] = sum_j A[i, j] * B[j, k]
    
    # Split j into two groups: j0 in {0,1}, j1 in {2,3}
    # C[k, i] = sum_{j0 in {0,1}} A[i, j0] * B[j0, k] + sum_{j1 in {2,3}} A[i, j1] * B[j1, k]
    
    # For group 0: A0[i, j0] is 2x2, B0[j0, k] is 2x4
    #   B0[j0, k] for j0 in {0,1}, k in {0,1,2,3}
    #   Split k into two groups: k0 in {0,1}, k1 in {2,3}
    #   B0[j0, k0] is 2x2, B0[j0, k1] is 2x2
    #   So group 0 contribution to C: Strassen(A0, B0[:, k0]) gives C0[:, k0] (2x2)
    #                                Strassen(A0, B0[:, k1]) gives C0[:, k1] (2x2)
    #   That's 2 Strassen decompositions for group 0, rank 14.
    
    # Similarly for group 1: A1[i, j1] is 2x2, B1[j1, k] is 2x4
    #   Strassen(A1, B1[:, k0]) gives C1[:, k0] (2x2)
    #   Strassen(A1, B1[:, k1]) gives C1[:, k1] (2x2)
    #   That's 2 Strassen decompositions for group 1, rank 14.
    
    # Total rank: 4 * 7 = 28
    
    # Let me build this explicitly.
    # There are 4 Strassen blocks: (group_j, group_k) where group_j in {0,1}, group_k in {0,1}
    # group_j=0: j in {0,1}, group_j=1: j in {2,3}
    # group_k=0: k in {0,1}, group_k=1: k in {2,3}
    
    rank = 28
    # U has n*m = 8 rows, V has m*p = 16 rows, W has p*n = 8 rows
    
    U = [[0] * rank for _ in range(8)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(8)]
    
    # For each (gj, gk) block, we use 7 columns.
    # Block (gj, gk) uses columns gj*14 + gk*7 + s for s in 0..6
    
    for gj in range(2):  # group of j indices
        for gk in range(2):  # group of k indices
            base_col = gj * 14 + gk * 7
            for s in range(7):
                col = base_col + s
                # U entries: for i in {0,1}, j_local in {0,1}
                # j_global = gj * 2 + j_local
                # U row index: i * 4 + j_global
                for i in range(2):
                    for jl in range(2):
                        j_global = gj * 2 + jl
                        U[i * 4 + j_global][col] = U_strassen[i * 2 + jl][s]
                
                # V entries: for j_local in {0,1}, k_local in {0,1}
                # j_global = gj * 2 + j_local
                # k_global = gk * 2 + k_local
                # V row index: j_global * 4 + k_global
                for jl in range(2):
                    for kl in range(2):
                        j_global = gj * 2 + jl
                        k_global = gk * 2 + kl
                        V[j_global * 4 + k_global][col] = V_strassen[jl * 2 + kl][s]
                
                # W entries: for k_local in {0,1}, i in {0,1}
                # k_global = gk * 2 + k_local
                # W row index: k_global * 2 + i
                for kl in range(2):
                    for i in range(2):
                        k_global = gk * 2 + kl
                        W[k_global * 2 + i][col] = W_strassen[kl * 2 + i][s]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
