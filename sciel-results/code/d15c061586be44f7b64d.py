"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36

    # Rank 129 decomposition:
    # 49 terms from Strassen for A(4x4) * B1(4x4) [cols 0-3 of B]
    # 80 terms from Schoolbook for A(4x4) * B2(4x5) [cols 4-8 of B]
    rank_total = 49 + 80
    U = np.zeros((nm, rank_total))
    V = np.zeros((mp, rank_total))
    W = np.zeros((pn, rank_total))

    # --- Part 1: Strassen for first 4 columns ---
    # Define the 7 linear combinations for X_l and Y_l
    # X_l and Y_l are 2x2 matrices of coefficients for A_blocks and B_blocks
    # A_blocks: A00, A01, A10, A11 (indices 0,1,2,3 in 2x2 grid)
    # B_blocks: B00, B01, B10, B11
    
    # Coefficients c_l[i,j] for X_l = sum c_l[i,j] A_ij
    # Coefficients d_l[i,j] for Y_l = sum d_l[i,j] B_ij
    
    # l=0: P1 = (A00+A11)(B00+B11)
    # l=1: P2 = (A10+A11)B00
    # l=2: P3 = A00(B01-B11)
    # l=3: P4 = A11(B10-B00)
    # l=4: P5 = (A00+A01)B11
    # l=5: P6 = (A10-A00)(B00+B01)
    # l=6: P7 = (A01-A11)(B10+B11)
    
    C_X = [
        [[1, 0], [0, 1]], # l=0
        [[0, 0], [1, 1]], # l=1
        [[1, 0], [0, 0]], # l=2
        [[0, 0], [0, 1]], # l=3
        [[1, 1], [0, 0]], # l=4
        [[-1, 0], [1, 0]], # l=5
        [[0, 1], [0, -1]]  # l=6
    ]
    
    C_Y = [
        [[1, 0], [0, 1]], # l=0
        [[1, 0], [0, 0]], # l=1
        [[0, 1], [0, -1]], # l=2
        [[-1, 0], [0, 1]], # l=3
        [[0, 0], [0, 1]], # l=4
        [[1, 1], [0, 0]], # l=5
        [[0, 1], [0, 1]]  # l=6
    ]

    # Strassen rank-1 decomposition for 2x2 multiplication
    # s=0..6
    # L_X_s[u,v] coefficients for x_u,v
    # L_Y_s[u,v] coefficients for y_u,v
    # C_s[u,v] coefficients for z_u,v
    
    L_X = [
        [[1, 0], [0, 1]], # M1: x00+x11
        [[0, 0], [1, 1]], # M2: x10+x11
        [[1, 0], [0, 0]], # M3: x00
        [[0, 0], [0, 1]], # M4: x11
        [[1, 1], [0, 0]], # M5: x00+x01
        [[-1, 0], [1, 0]], # M6: x10-x00
        [[0, 1], [0, -1]]  # M7: x01-x11
    ]
    
    L_Y = [
        [[1, 0], [0, 1]], # M1: y00+y11
        [[1, 0], [0, 0]], # M2: y00
        [[0, 1], [0, -1]], # M3: y01-y11
        [[-1, 0], [0, 1]], # M4: y10-y00
        [[0, 0], [0, 1]], # M5: y11
        [[1, 1], [0, 0]], # M6: y00+y01
        [[0, 1], [0, 1]]  # M7: y10+y11
    ]
    
    C_Z = [
        [[1, 0], [0, 1]], # M1: z00, z11
        [[0, 0], [1, -1]], # M2: z10, -z11
        [[0, 1], [0, 1]], # M3: z01, z11
        [[1, 0], [1, 0]], # M4: z00, z10
        [[-1, 1], [0, 0]], # M5: -z00, z01
        [[0, 0], [0, 1]], # M6: z11
        [[1, 0], [0, 0]]  # M7: z00
    ]

    for l in range(7):
        for s in range(7):
            r = 7 * l + s
            
            # Construct U vector
            # U index: i*m + j. i is row of A, j is col of A.
            # A is 4x4. Blocks A_ib, jb are 2x2.
            # Entry (i,j) of A corresponds to block (i//2, j//2) and local (i%2, j%2).
            # L_X_l_s is a linear form on X_l.
            # X_l[u,v] = sum_{ib,jb} C_X[l][ib][jb] * A_ib_jb[u,v]
            # L_X_s is linear form on X_l entries.
            # L_X_l_s[i,j] = sum_{u,v} L_X[s][u][v] * (X_l)[u,v] 
            #              = sum_{u,v} L_X[s][u][v] * sum_{ib,jb} C_X[l][ib][jb] * A_ib_jb[u,v]
            #              = sum_{ib,jb,u,v} L_X[s][u][v] C_X[l][ib][jb] A_ib_jb[u,v]
            # The coefficient for A[ib*2+u, jb*2+v] is L_X[s][u][v] * C_X[l][ib][jb]
            
            for ib in range(2):
                for jb in range(2):
                    for u in range(2):
                        for v in range(2):
                            coeff = L_X[s][u][v] * C_X[l][ib][jb]
                            if coeff != 0:
                                i_global = ib * 2 + u
                                j_global = jb * 2 + v
                                U[i_global * m + j_global, r] += coeff

            # Construct V vector
            # V index: j*p + k. j is col of A (row of B), k is col of B.
            # B is 4x9. First 4 cols are B1.
            # B1 blocks B_ib, jb are 2x2.
            # Entry (j,k) of B1 corresponds to block (j//2, k//2) and local (j%2, k%2).
            # Note: j in V is row of B. k in V is col of B.
            # In our block notation for B1, row block is ib? No.
            # A is 4x4, B is 4x9.
            # A row i, col j. B row j, col k.
            # Block row of B is j//2. Block col of B is k//2.
            # Y_l[u,v] corresponds to B entry (row_block*2+u, col_block*2+v).
            # Wait, Y_l is the matrix being multiplied by X_l.
            # X_l is from A. Y_l is from B.
            # X_l[u,v] is element (u,v) of the 2x2 block matrix.
            # In A, block (ib, jb) has elements A[ib*2+u, jb*2+v].
            # In B, block (jb, kb) has elements B[jb*2+u, kb*2+v].
            # So Y_l[u,v] corresponds to B entry (jb*2+u, kb*2+v).
            # Here u is local row, v is local col.
            # So global row j_global = jb*2+u. Global col k_global = kb*2+v.
            
            for jb in range(2):
                for kb in range(2):
                    for u in range(2):
                        for v in range(2):
                            coeff = L_Y[s][u][v] * C_Y[l][jb][kb]
                            if coeff != 0:
                                j_global = jb * 2 + u
                                k_global = kb * 2 + v
                                V[j_global * p + k_global, r] += coeff

            # Construct W vector
            # W index: k*n + i. k is col of B, i is row of A.
            # Result C is 4x4 (for first 4 cols).
            # C entry (i, k) comes from block (i//2, k//2) local (i%2, k%2).
            # Z_l[u,v] is element (u,v) of result block.
            # C_Z[s][u][v] is coefficient for Z_l[u,v].
            # Global i = ib*2+u. Global k = kb*2+v.
            
            for ib in range(2):
                for kb in range(2):
                    for u in range(2):
                        for v in range(2):
                            coeff = C_Z[s][u][v]
                            if coeff != 0:
                                i_global = ib * 2 + u
                                k_global = kb * 2 + v
                                W[k_global * n + i_global, r] += coeff

    # --- Part 2: Schoolbook for remaining 5 columns ---
    # Cols 4-8 of B.
    # A (4x4) * B2 (4x5).
    # Rank 80.
    col_offset = 49
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                r = col_offset + (i * m + j) * (p - 4) + (k - 4)
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1

    # Verification
    # T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
    # Should be 1 if i,j,k are valid indices for the multiplication.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
