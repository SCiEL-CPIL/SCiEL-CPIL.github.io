"""Matrix multiplication 4x4x5 tensor decomposition using Strassen + schoolbook."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# Strassen decomposition for 2x2x2 (rank 7)
def strassen_2x2x2():
    """Return U, V, W for 2x2x2 matrix multiplication, rank 7."""
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i*2+j], V[i*2+j], W[i*2+j] for i,j in {0,1}
    # C = A*B, A,B,C are 2x2
    # Strassen's 7 products:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B01-B11)
    # P4 = A11*(B10-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1: (A00+A11)*(B00+B11) -> C00, C11
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # P2: (A10+A11)*B00 -> C10, C11
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10
    W[3, 1] = -1 # C11
    
    # P3: A00*(B01-B11) -> C01, C11
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1 # B11
    W[1, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # P4: A11*(B10-B00) -> C00, C10
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1 # B00
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10
    
    # P5: (A00+A01)*B11 -> C00, C01
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1 # C00
    W[1, 4] = 1  # C01
    
    # P6: (A10-A00)*(B00+B01) -> C11
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    W[3, 5] = 1  # C11
    
    # P7: (A01-A11)*(B10+B11) -> C00
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    W[0, 6] = 1  # C00
    
    return U, V, W


def strassen_4x4x4():
    """Return U, V, W for 4x4x4 matrix multiplication using Strassen, rank 49."""
    # Use recursive Strassen: 4x4x4 = 7 * (2x2x2) with each 2x2x2 having rank 7
    # Total rank = 7*7 = 49
    U2, V2, W2 = strassen_2x2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # The 7 top-level Strassen products, each expanded into 7 sub-products
    # Top-level: A = [[A00, A01], [A10, A11]] where each is 2x2
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B01-B11)
    # P4 = A11*(B10-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    
    # For each top-level product alpha, we have a 2x2x2 Strassen decomposition
    # with 7 terms. Each term maps to specific 2x2 blocks.
    
    # Block indices: for 4x4 matrix, block (I,J) covers rows 2I:2I+2, cols 2J:2J+2
    # U index: (I*2+i)*4 + (J*2+j) = I*8 + J*2 + i*4 + j... 
    # Actually, let's use: U[(I*2+i)*4 + (J*2+j)] for row I*2+i, col J*2+j
    # Wait, U has shape (16, rank). Index is row*4+col for a 4x4 matrix.
    # Row index in 4x4: I*2+i, Col index: J*2+j. So U[(I*2+i)*4 + (J*2+j)]
    
    # V similarly: row J*2+i, col K*2+j for 4x4 matrix -> V[(J*2+i)*4 + (K*2+j)]
    # W: row K*2+i, col L*2+j for 4x4 result -> W[(K*2+i)*4 + (L*2+j)]
    
    # Strassen formula for 4x4 using 2x2 blocks:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # For each top-level product P_alpha, it's a 2x2x2 multiplication of block matrices.
    # The 7 sub-terms of Strassen for 2x2x2 are defined by U2, V2, W2.
    # U2 has shape (4,7): indices 0,1,2,3 for the 2x2 A-block
    # V2 has shape (4,7): indices 0,1,2,3 for the 2x2 B-block
    # W2 has shape (4,7): indices 0,1,2,3 for the 2x2 C-block
    
    # For top-level product alpha, the A-block involved is determined by the Strassen formula.
    # P1: A-blocks 00 and 11, B-blocks 00 and 11
    # P2: A-blocks 10 and 11, B-block 00
    # P3: A-block 00, B-blocks 01 and 11
    # P4: A-block 11, B-blocks 10 and 00
    # P5: A-blocks 00 and 01, B-block 11
    # P6: A-blocks 10 and 00, B-blocks 00 and 01
    # P7: A-blocks 01 and 11, B-blocks 10 and 11
    
    # For each (alpha, beta) where alpha in 0..6, beta in 0..6:
    # The A sub-block is determined by U2[beta, i_a] for i_a in 0..3
    # i_a maps to: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1) in the 2x2 block space
    # But we need to know which 2x2 block of the 4x4 matrix this refers to.
    
    # Let me define the mapping more carefully.
    # For top-level product alpha, we have:
    #   A_alpha = sum of certain 2x2 blocks of A with coefficients
    #   B_alpha = sum of certain 2x2 blocks of B with coefficients
    #   P_alpha = A_alpha * B_alpha (2x2x2 multiplication)
    
    # The 2x2x2 multiplication P_alpha is decomposed into 7 terms using U2,V2,W2.
    # Each term beta has:
    #   U2[beta, i] for i in 0..3: coefficient for A_alpha's 2x2 entry (row i//2, col i%2)
    #   V2[beta, j] for j in 0..3: coefficient for B_alpha's 2x2 entry (row j//2, col j%2)
    #   W2[beta, k] for k in 0..3: coefficient for P_alpha's 2x2 entry (row k//2, col k%2)
    
    # Now A_alpha is a 2x2 matrix of 2x2 blocks. Entry (a_row, a_col) of A_alpha is:
    #   A_alpha[a_row, a_col] = sum_{I,J} coeff_alpha_A[I,J,a_row,a_col] * A_block[I,J]
    # where A_block[I,J] is the 2x2 block at position (I,J) of the 4x4 A.
    
    # This is getting complex. Let me just directly construct the rank-49 decomposition
    # by enumerating all 49 terms.
    
    # Alternative: just use the known Strassen rank-49 decomposition.
    # Let me build it programmatically.
    
    # For each of the 7 top-level products, we need to expand the 2x2x2 Strassen.
    # The 2x2x2 Strassen uses 7 terms. Each term picks specific entries from the 2x2 
    # A-matrix, 2x2 B-matrix, and writes to specific entries of the 2x2 C-matrix.
    
    # For the top-level, the "2x2 A-matrix" is actually a matrix of 2x2 blocks.
    # So when U2 says "pick entry (0,0) of A with coeff c", that means pick the 
    # entire 2x2 block at position (0,0) of the 4x4 A, scaled by c.
    
    # So for term (alpha, beta):
    #   For each i in 0..3: if U2[beta, i] != 0, then for each (ii, jj) in 0..1:
    #     U_global[(I*2+ii)*4 + (J*2+jj), col] += U2[beta, i] * coeff_A[alpha, i, I, J]
    #   where I = i//2, J = i%2, and coeff_A is the coefficient for block (I,J) in A_alpha.
    
    # Let me define the top-level Strassen coefficients.
    # A_alpha coefficients for block (I,J):
    # P1: A00 and A11 -> coeff[0,0]=1, coeff[1,1]=1
    # P2: A10 and A11 -> coeff[1,0]=1, coeff[1,1]=1
    # P3: A00 -> coeff[0,0]=1
    # P4: A11 -> coeff[1,1]=1
    # P5: A00 and A01 -> coeff[0,0]=1, coeff[0,1]=1
    # P6: A10 and A00 -> coeff[1,0]=1, coeff[0,0]=-1
    # P7: A01 and A11 -> coeff[0,1]=1, coeff[1,1]=-1
    
    # B_alpha coefficients for block (K,L):
    # P1: B00 and B11 -> coeff[0,0]=1, coeff[1,1]=1
    # P2: B00 -> coeff[0,0]=1
    # P3: B01 and B11 -> coeff[0,1]=1, coeff[1,1]=-1
    # P4: B10 and B00 -> coeff[1,0]=1, coeff[0,0]=-1
    # P5: B11 -> coeff[1,1]=1
    # P6: B00 and B01 -> coeff[0,0]=1, coeff[0,1]=1
    # P7: B10 and B11 -> coeff[1,0]=1, coeff[1,1]=1
    
    # C_alpha coefficients for block (M,N):
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    A_coeffs = [
        [(0,0,1), (1,1,1)],       # P1
        [(1,0,1), (1,1,1)],       # P2
        [(0,0,1)],                 # P3
        [(1,1,1)],                 # P4
        [(0,0,1), (0,1,1)],       # P5
        [(1,0,1), (0,0,-1)],      # P6
        [(0,1,1), (1,1,-1)],      # P7
    ]
    B_coeffs = [
        [(0,0,1), (1,1,1)],       # P1
        [(0,0,1)],                 # P2
        [(0,1,1), (1,1,-1)],      # P3
        [(1,0,1), (0,0,-1)],      # P4
        [(1,1,1)],                 # P5
        [(0,0,1), (0,1,1)],       # P6
        [(1,0,1), (1,1,1)],       # P7
    ]
    C_coeffs = [
        [(0,0,1), (1,1,1)],       # P1 -> C00, C11
        [(1,0,1), (1,1,-1)],      # P2 -> C10, C11
        [(0,1,1), (1,1,1)],       # P3 -> C01, C11
        [(0,0,1), (1,0,1)],       # P4 -> C00, C10
        [(0,0,-1), (0,1,1)],      # P5 -> C00, C01
        [(1,1,1)],                 # P6 -> C11
        [(0,0,1)],                 # P7 -> C00
    ]
    
    for alpha in range(7):
        for beta in range(7):
            col = alpha * 7 + beta
            # U2[beta, i] for i in 0..3 gives coefficient for 2x2 A entry (i//2, i%2)
            for i in range(4):
                if U2[beta, i] == 0:
                    continue
                a_row = i // 2
                a_col = i % 2
                for (I, J, coeff_a) in A_coeffs[alpha]:
                    if a_row == I and a_col == J:
                        for ii in range(2):
                            for jj in range(2):
                                u_idx = (I*2+ii)*4 + (J*2+jj)
                                U[u_idx, col] += U2[beta, i] * coeff_a
            for j in range(4):
                if V2[beta, j] == 0:
                    continue
                b_row = j // 2
                b_col = j % 2
                for (K, L, coeff_b) in B_coeffs[alpha]:
                    if b_row == K and b_col == L:
                        for ii in range(2):
                            for jj in range(2):
                                v_idx = (K*2+ii)*4 + (L*2+jj)
                                V[v_idx, col] += V2[beta, j] * coeff_b
            for k in range(4):
                if W2[beta, k] == 0:
                    continue
                c_row = k // 2
                c_col = k % 2
                for (M, N, coeff_c) in C_coeffs[alpha]:
                    if c_row == M and c_col == N:
                        for ii in range(2):
                            for jj in range(2):
                                w_idx = (M*2+ii)*4 + (N*2+jj)
                                W[w_idx, col] += W2[beta, k] * coeff_c
    
    return U, V, W


def construct():
    start_time = time.time()
    DEADLINE = start_time + 120  # safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Get Strassen decomposition for 4x4x4 (rank 49)
    U_strassen, V_strassen, W_strassen = strassen_4x4x4()
    
    # Schoolbook decomposition for 4x4x1 (rank 16)
    # A is 4x4, B is 4x1, C is 4x1
    # U: 16x16, V: 4x16, W: 4x16
    # C[i] = sum_j A[i,j] * B[j]
    # Term (i,j): U[i*4+j] = 1, V[j] = 1, W[i] = 1
    
    U_sb = np.zeros((16, 16))
    V_sb = np.zeros((4, 16))
    W_sb = np.zeros((4, 16))
    for i in range(4):
        for j in range(4):
            c = i * 4 + j
            U_sb[i*4+j, c] = 1.0
            V_sb[j, c] = 1.0
            W_sb[i, c] = 1.0
    
    # Combine: total rank = 49 + 16 = 65
    # U: 16x65, V: 20x65 (4*5=20), W: 20x65 (5*4=20)
    # Wait, the problem says:
    # U: (n*m, rank) = (4*4, 65) = (16, 65)
    # V: (m*p, rank) = (4*5, 65) = (20, 65)
    # W: (p*n, rank) = (5*4, 65) = (20, 65)
    # And W[k*n+i, r] convention.
    
    # The Strassen part handles columns 0-3 of B (the 4x4 part).
    # The schoolbook part handles column 4 of B (the 4x1 part).
    
    # For Strassen part:
    # U_strassen: 16x49, corresponds to A (4x4)
    # V_strassen: 16x49, corresponds to B's first 4 columns (4x4)
    # W_strassen: 16x49, corresponds to C's first 4 columns (4x4)
    
    # For schoolbook part:
    # U_sb: 16x16, corresponds to A (4x4)
    # V_sb: 4x16, corresponds to B's 5th column (4x1)
    # W_sb: 4x16, corresponds to C's 5th column (4x1)
    
    U = np.zeros((16, 65))
    V = np.zeros((20, 65))
    W = np.zeros((20, 65))
    
    # Fill Strassen part (columns 0-3 of B, columns 0-3 of C)
    U[:, :49] = U_strassen
    # V_strassen is 16x49, maps to B's rows 0-3, cols 0-3. V index = row*5+col
    for r in range(16):
        row = r // 4
        col = r % 4
        V[row*5+col, :49] = V_strassen[r, :]
    # W_strassen is 16x49, maps to C's rows 0-3, cols 0-3. W index = col*4+row (k*n+i)
    for r in range(16):
        row = r // 4
        col = r % 4
        W[col*4+row, :49] = W_strassen[r, :]
    
    # Fill schoolbook part (column 4 of B, column 4 of C)
    U[:, 49:] = U_sb
    # V_sb is 4x16, maps to B's rows 0-3, col 4. V index = row*5+4
    for r in range(4):
        V[r*5+4, 49:] = V_sb[r, :]
    # W_sb is 4x16, maps to C's rows 0-3, col 4. W index = 4*4+row = 16+row
    for r in range(4):
        W[4*4+r, 49:] = W_sb[r, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
