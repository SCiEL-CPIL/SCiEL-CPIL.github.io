"""Matrix multiplication 4x4x5 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 50 decomposition: 49 (Strassen for 4x4x4) + 1 (Schoolbook for 4x4x1)
    rank = 50
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 base case
    # Returns U, V, W for 2x2x2 multiplication
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    def strassen_2x2_2x2():
        u = np.zeros((4, 7))
        v = np.zeros((4, 7))
        w = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        u[0, 0] = 1; u[3, 0] = 1
        v[0, 0] = 1; v[3, 0] = 1
        w[0, 0] = 1; w[2, 0] = 1; w[1, 1] = 1; w[3, 1] = 1 
        # w indices: k*2+i
        # C00 (k=0,i=0) -> w[0]
        # C01 (k=1,i=0) -> w[2]
        # C10 (k=0,i=1) -> w[1]
        # C11 (k=1,i=1) -> w[3]
        # M1 contributes to all 4?
        # A00B00 -> C00
        # A00B11 -> C01 (k=1, i=0) -> w[1*2+0] = w[2]
        # A11B00 -> C10 (k=0, i=1) -> w[0*2+1] = w[1]
        # A11B11 -> C11 (k=1, i=1) -> w[1*2+1] = w[3]
        # So w[0]=1, w[1]=1, w[2]=1, w[3]=1.
        w[0, 0] = 1; w[1, 0] = 1; w[2, 0] = 1; w[3, 0] = 1

        # M2 = (A10+A11)B00
        u[2, 1] = 1; u[3, 1] = 1
        v[0, 1] = 1
        # A10B00 -> C10 (k=0, i=1) -> w[1]
        # A11B00 -> C11 (k=1, i=1) -> w[3]
        w[1, 1] = 1; w[3, 1] = 1

        # M3 = A00(B10-B11)
        u[0, 2] = 1
        v[2, 2] = 1; v[3, 2] = -1
        # A00B10 -> C00 (k=0, i=0) -> w[0]
        # A00B11 -> C01 (k=1, i=0) -> w[2]
        w[0, 2] = 1; w[2, 2] = -1

        # M4 = A11(B01-B00)
        u[3, 3] = 1
        v[1, 3] = 1; v[0, 3] = -1
        # A11B01 -> C11 (k=1, i=1) -> w[3]
        # A11B00 -> C10 (k=0, i=1) -> w[1]
        w[3, 3] = 1; w[1, 3] = -1

        # M5 = (A00+A10)B11
        u[0, 4] = 1; u[2, 4] = 1
        v[3, 4] = 1
        # A00B11 -> C01 (k=1, i=0) -> w[2]
        # A10B11 -> C11 (k=1, i=1) -> w[3]
        w[2, 4] = 1; w[3, 4] = 1

        # M6 = (A10-A00)(B00+B01)
        u[2, 5] = 1; u[0, 5] = -1
        v[0, 5] = 1; v[1, 5] = 1
        # A10B00 -> C10 (k=0, i=1) -> w[1]
        # A10B01 -> C11 (k=1, i=1) -> w[3]
        # -A00B00 -> C00 (k=0, i=0) -> w[0]
        # -A00B01 -> C01 (k=1, i=0) -> w[2]
        w[1, 5] = 1; w[3, 5] = 1; w[0, 5] = -1; w[2, 5] = -1

        # M7 = (A01-A11)(B10+B11)
        u[1, 6] = 1; u[3, 6] = -1
        v[2, 6] = 1; v[3, 6] = 1
        # A01B10 -> C00 (k=0, i=0) -> w[0]
        # A01B11 -> C01 (k=1, i=0) -> w[2]
        # -A11B10 -> C10 (k=0, i=1) -> w[1]
        # -A11B11 -> C11 (k=1, i=1) -> w[3]
        w[0, 6] = 1; w[2, 6] = 1; w[1, 6] = -1; w[3, 6] = -1

        return u, v, w

    u_base, v_base, w_base = strassen_2x2_2x2()
    
    # Combine 4 blocks of 2x2x2 for 4x4x4
    # Blocks:
    # A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # B00 (0,0), B01 (0,1), B10 (1,0), B11 (1,1)
    # C00 (0,0), C01 (0,1), C10 (1,0), C11 (1,1)
    
    # Strassen 4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mr is a 2x2x2 multiplication.
    # We need to map the local 2x2 indices to global 4x4 indices.
    
    # Helper to add a 2x2x2 decomposition at specific block offsets
    def add_block(r_start, ai, aj, bi, bj, ci, cj):
        # ai, aj: row/col offset of A block (0 or 1)
        # bi, bj: row/col offset of B block (0 or 1)
        # ci, cj: row/col offset of C block (0 or 1)
        
        for r in range(7):
            col = r_start + r
            for i_local in range(2):
                for j_local in range(2):
                    i_global = ai * 2 + i_local
                    j_global = aj * 2 + j_local
                    U[i_global * m + j_global, col] += u_base[i_local * 2 + j_local, r]
                    
            for j_local in range(2):
                for k_local in range(2):
                    j_global = bi * 2 + j_local
                    k_global = bj * 2 + k_local
                    V[j_global * p + k_global, col] += v_base[j_local * 2 + k_local, r]
                    
            for k_local in range(2):
                for i_local in range(2):
                    k_global = ci * 2 + k_local
                    i_global = cj * 2 + i_local
                    W[k_global * n + i_global, col] += w_base[k_local * 2 + i_local, r]

    # M1: (A00+A11)(B00+B11) -> C00, C11?
    # C00 = A00B00 + A01B10
    # C11 = A10B01 + A11B11
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # This term contributes to C00 (A00B00), C01 (A00B11), C10 (A11B00), C11 (A11B11)?
    # No, the Strassen algorithm computes the 7 products, and then combines them to get C blocks.
    # The decomposition is a sum of rank-1 tensors.
    # The "rank 49" comes from using 7 rank-7 decompositions.
    # Each rank-7 decomposition computes a 2x2x2 product.
    # The linear combinations are handled by the U, V, W coefficients.
    
    # So we just need to instantiate the 7 base decompositions for the specific A, B, C blocks involved.
    
    # M1: A=(0,0)+(1,1), B=(0,0)+(1,1)
    # This is not a single block multiplication.
    # It is (A00+A11) * (B00+B11).
    # Let X = A00+A11, Y = B00+B11.
    # We compute X*Y using the 2x2x2 decomposition.
    # X is 2x2, Y is 2x2.
    # The result Z = X*Y is 2x2.
    # Z contributes to C00 and C11?
    # C00 = M1 - M4 + M6
    # C11 = M1 + M3 - M5 + M7?
    
    # We must map the "virtual" 2x2 multiplication to the global indices.
    # U for X:
    # X_00 = A_00,00 + A_11,00
    # X_01 = A_00,01 + A_11,00? No.
    # X is a 2x2 matrix.
    # X[i,j] = A[2*ai+i, 2*aj+j] + A[2*(1-ai)+i, 2*(1-aj)+j]?
    # No, M1 uses A00 and A11.
    # A00 is top-left, A11 is bottom-right.
    # X = A00 + A11.
    # X[i,j] = A00[i,j] + A11[i,j].
    
    # So for U, we sum the contributions from block (0,0) and (1,1).
    
    def add_virtual_block(r_start, a_blocks, b_blocks, c_blocks):
        # a_blocks: list of (ai, aj)
        # b_blocks: list of (bi, bj)
        # c_blocks: list of (ci, cj)
        
        for r in range(7):
            col = r_start + r
            # U
            for (ai, aj) in a_blocks:
                for i_local in range(2):
                    for j_local in range(2):
                        i_global = ai * 2 + i_local
                        j_global = aj * 2 + j_local
                        U[i_global * m + j_global, col] += u_base[i_local * 2 + j_local, r]
                        
            # V
            for (bi, bj) in b_blocks:
                for j_local in range(2):
                    for k_local in range(2):
                        j_global = bi * 2 + j_local
                        k_global = bj * 2 + k_local
                        V[j_global * p + k_global, col] += v_base[j_local * 2 + k_local, r]
                        
            # W
            for (ci, cj) in c_blocks:
                for k_local in range(2):
                    for i_local in range(2):
                        k_global = ci * 2 + k_local
                        i_global = cj * 2 + i_local
                        W[k_global * n + i_global, col] += w_base[k_local * 2 + i_local, r]

    # M1: (A00+A11)(B00+B11)
    # A blocks: (0,0), (1,1)
    # B blocks: (0,0), (1,1)
    # C blocks: (0,0), (1,1) ?
    # The result of M1 is added to C00 and C11?
    # C00 = M1 - M4 + M6
    # C11 = M1 + M3 - M5 + M7
    # C01 = M4 + M5
    # C10 = M2 + M7
    # So M1 contributes to C00 and C11.
    add_virtual_block(0, [(0,0), (1,1)], [(0,0), (1,1)], [(0,0), (1,1)])

    # M2: (A10+A11)B00
    # A blocks: (1,0), (1,1)
    # B blocks: (0,0)
    # C blocks: (1,0) ?
    # C10 = M2 + M7
    # So M2 contributes to C10.
    add_virtual_block(7, [(1,0), (1,1)], [(0,0)], [(1,0)])

    # M3: A00(B10-B11)
    # A blocks: (0,0)
    # B blocks: (1,0), (1,1) with sign -1 for (1,1)?
    # v_base handles the signs?
    # v_base for M3: v[2]=1, v[3]=-1.
    # v indices: j*2+k.
    # v[2] is j=1, k=0. v[3] is j=1, k=1.
    # So it expects B10 and B11.
    # We just add the blocks.
    add_virtual_block(14, [(0,0)], [(1,0), (1,1)], [(0,0)])
    # C00 = M1 - M4 + M6
    # C11 = M1 + M3 - M5 + M7
    # So M3 contributes to C11.
    # Wait, C00 formula: M1 - M4 + M6.
    # C11 formula: M1 + M3 - M5 + M7.
    # So M3 is in C11.
    # My add_virtual_block for M3 used c_blocks=[(0,0)]. It should be [(1,1)].
    # Let's re-verify C00, C01, C10, C11 formulas.
    # C00 = M1 - M4 + M6
    # C01 = M4 + M5
    # C10 = M2 + M7
    # C11 = M1 + M3 - M5 + M7
    
    # So:
    # M1 -> C00, C11
    # M2 -> C10
    # M3 -> C11
    # M4 -> C00, C01
    # M5 -> C01, C11
    # M6 -> C00
    # M7 -> C10, C11

    # Correcting M3:
    U[:, 14:21] = 0; V[:, 14:21] = 0; W[:, 14:21] = 0
    add_virtual_block(14, [(0,0)], [(1,0), (1,1)], [(1,1)])

    # M4: A11(B01-B00)
    # A blocks: (1,1)
    # B blocks: (0,1), (0,0)
    # v_base for M4: v[1]=1, v[0]=-1.
    # v[1] is j=0, k=1. v[0] is j=0, k=0.
    # So B01 and B00.
    # C blocks: C00, C01
    add_virtual_block(21, [(1,1)], [(0,1), (0,0)], [(0,0), (0,1)])

    # M5: (A00+A10)B11
    # A blocks: (0,0), (1,0)
    # B blocks: (1,1)
    # C blocks: C01, C11
    add_virtual_block(28, [(0,0), (1,0)], [(1,1)], [(0,1), (1,1)])

    # M6: (A10-A00)(B00+B01)
    # A blocks: (1,0), (0,0)
    # u_base for M6: u[2]=1, u[0]=-1.
    # u[2] is i=1, j=0. u[0] is i=0, j=0.
    # So A10 and A00.
    # B blocks: (0,0), (0,1)
    # C blocks: C00
    add_virtual_block(35, [(1,0), (0,0)], [(0,0), (0,1)], [(0,0)])

    # M7: (A01-A11)(B10+B11)
    # A blocks: (0,1), (1,1)
    # u_base for M7: u[1]=1, u[3]=-1.
    # u[1] is i=0, j=1. u[3] is i=1, j=1.
    # So A01 and A11.
    # B blocks: (1,0), (1,1)
    # C blocks: C10, C11
    add_virtual_block(42, [(0,1), (1,1)], [(1,0), (1,1)], [(1,0), (1,1)])

    # Add 5th column (k=4) with rank 1
    # U[i, 49] = 1 for all i
    # V[j, 49] = 1 for all j
    # W[4, i] = 1 for all i
    for i in range(n):
        for j in range(m):
            U[i * m + j, 49] = 1.0
    for j in range(m):
        V[j * p + 4, 49] = 1.0
    for i in range(n):
        W[4 * n + i, 49] = 1.0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
