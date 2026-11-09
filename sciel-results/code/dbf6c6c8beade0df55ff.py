"""Program for 3x4 * 4x7 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2_decomposition():
    """
    Return a Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    This gives rank 7 instead of 8.
    
    The 7 multiplications are:
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B11
    M3 = A11*(B11-B21)
    M4 = A22*(B21-B22)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B21+B22)
    
    Output:
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # For 2x2 * 2x2, indices:
    # A: (i,j) where i,j in {0,1}
    # B: (j,k) where j,k in {0,1}
    # C: (i,k) where i,k in {0,1}
    
    # We'll build U, V, W for the 2x2*2x2 case with rank 7
    # U has shape (2*2, 7) = (4, 7)
    # V has shape (2*2, 7) = (4, 7)
    # W has shape (2*2, 7) = (4, 7)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)*(B11+B22)
    # A11 is at (0,0), A22 at (1,1)
    # B11 is at (0,0), B22 at (1,1)
    U[0*2+0][0] = 1  # A11
    U[1*2+1][0] = 1  # A22
    V[0*2+0][0] = 1  # B11
    V[1*2+1][0] = 1  # B22
    W[0*2+0][0] = 1  # C11
    W[1*2+1][0] = 1  # C22
    
    # M2 = (A21+A22)*B11
    U[1*2+0][1] = 1  # A21
    U[1*2+1][1] = 1  # A22
    V[0*2+0][1] = 1  # B11
    W[1*2+0][1] = 1  # C21
    
    # M3 = A11*(B11-B21)
    U[0*2+0][2] = 1  # A11
    V[0*2+0][2] = 1  # B11
    V[1*2+0][2] = -1  # B21
    W[0*2+1][2] = 1  # C12
    
    # M4 = A22*(B21-B22)
    U[1*2+1][3] = 1  # A22
    V[1*2+0][3] = 1  # B21
    V[1*2+1][3] = -1  # B22
    W[0*2+0][3] = 1  # C11
    W[1*2+0][3] = 1  # C21
    
    # M5 = (A11+A12)*B22
    U[0*2+0][4] = 1  # A11
    U[0*2+1][4] = 1  # A12
    V[1*2+1][4] = 1  # B22
    W[0*2+0][4] = -1  # C11
    W[0*2+1][4] = 1  # C12
    
    # M6 = (A21-A11)*(B11+B12)
    U[1*2+0][5] = 1  # A21
    U[0*2+0][5] = -1  # A11
    V[0*2+0][5] = 1  # B11
    V[0*2+1][5] = 1  # B12
    W[1*2+1][5] = 1  # C22
    
    # M7 = (A12-A22)*(B21+B22)
    U[0*2+1][6] = 1  # A12
    U[1*2+1][6] = -1  # A22
    V[1*2+0][6] = 1  # B21
    V[1*2+1][6] = 1  # B22
    W[0*2+0][6] = 1  # C11
    W[1*2+1][6] = 1  # C22
    
    return {"U": U, "V": V, "W": W}

def _strassen_decomposition(n, m, p):
    """
    Use Strassen-style block decomposition to get a lower-rank factorization
    for 3x4 * 4x7.
    
    Strategy: Pad n=3 to 4, keep m=4, pad p=7 to 8.
    Now we have 4x4 * 4x8.
    
    Split B into B_left (4x4) and B_right (4x4).
    Compute C_left = A * B_left and C_right = A * B_right.
    
    Each 4x4 * 4x4 uses Strassen recursively:
    - Split into 2x2 blocks
    - Use 7 multiplications of 2x2 * 2x2
    - Each 2x2 * 2x2 uses schoolbook rank 4
    
    So each 4x4 * 4x4 has rank 7 * 4 = 28
    Total for both halves: 56
    
    We only need the first 3 rows and first 7 columns of the output.
    """
    n, m, p = 3, 4, 7
    n_padded = 4
    m_padded = 4
    p_padded = 8
    
    # Build the full 4x4 * 4x8 decomposition using Strassen
    # First, get the 2x2 * 2x2 Strassen decomposition
    base = _strassen_2x2_decomposition()
    
    # Now build 4x4 * 4x4 using Strassen with 2x2 blocks
    # We need to combine the base decomposition appropriately
    
    # For 4x4 * 4x4, we split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    # C = [[C11, C12], [C21, C22]] where each is 2x2
    
    # The Strassen multiplications are:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B11-B21)
    # M4 = A22*(B21-B22)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # Each Mi is a 2x2 * 2x2 product, which we decompose using the base Strassen (rank 7)
    # But wait, the base is already rank 7 for 2x2*2x2, so each Mi contributes rank 7
    # Total for 4x4*4x4: 7 * 7 = 49
    
    # Actually, I need to be more careful. The base decomposition I defined above
    # is for 2x2*2x2 with rank 7. When I use it for each Mi in the 4x4*4x4 Strassen,
    # each Mi is a 2x2*2x2 product, so I can use the same rank-7 decomposition.
    
    # Let me build the 4x4 * 4x4 decomposition:
    # U has shape (4*4, rank) = (16, rank)
    # V has shape (4*4, rank) = (16, rank)
    # W has shape (4*4, rank) = (16, rank)
    
    # For each of the 7 Strassen multiplications, I need to map the 2x2*2x2
    # decomposition to the appropriate subblocks.
    
    # Let's build this step by step.
    
    # First, let me define how to index the 4x4 matrices:
    # A[i,j] for i,j in 0..3
    # B[j,k] for j,k in 0..3
    # C[i,k] for i,k in 0..3
    
    # Block indices:
    # A11: A[0:2, 0:2], A12: A[0:2, 2:4]
    # A21: A[2:4, 0:2], A22: A[2:4, 2:4]
    
    # Similarly for B and C.
    
    # I'll build the decomposition by creating entries for each of the 7 Mi.
    
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    # Define the 7 Strassen multiplications for 4x4*4x4 with 2x2 blocks
    # Each Mi = Xi * Yi where Xi is a 2x2 combination of A-blocks, Yi is 2x2 combination of B-blocks
    # The result contributes to C-blocks
    
    # Strassen formulas for 2x2 block matrices:
    # M1 = (A11+A22) * (B11+B22) -> contributes to C11 (+), C22 (+)
    # M2 = (A21+A22) * B11 -> contributes to C21 (+)
    # M3 = A11 * (B11-B21) -> contributes to C12 (+)
    # M4 = A22 * (B21-B22) -> contributes to C11 (+), C21 (+)
    # M5 = (A11+A12) * B22 -> contributes to C11 (-), C12 (+)
    # M6 = (A21-A11) * (B11+B12) -> contributes to C22 (+)
    # M7 = (A12-A22) * (B21+B22) -> contributes to C11 (+), C22 (+)
    
    # For each Mi, we need to:
    # 1. Express Xi as a linear combination of A-blocks
    # 2. Express Yi as a linear combination of B-blocks
    # 3. Use the base 2x2*2x2 decomposition to decompose Xi*Yi
    # 4. Map the result to the appropriate C-blocks
    
    # The base decomposition gives us: for 2x2*2x2, we have U_base, V_base, W_base
    # where U_base[i,j] corresponds to A[i,j], V_base[j,k] to B[j,k], W_base[k,i] to C[k,i]
    # (using the trace convention W[k*n+i,r])
    
    # For each Mi, we'll create 7 new columns (one for each base column)
    # and fill in the appropriate entries
    
    col = 0
    for mi in range(7):
        for base_col in range(7):
            # Get the base decomposition entries for this column
            u_base = base["U"]
            v_base = base["V"]
            w_base = base["W"]
            
            # Now we need to map the 2x2 indices to the 4x4 subblocks
            # This depends on which Mi we're computing
            
            # For Mi, we have Xi (2x2 matrix of A-block combinations) and Yi (2x2 matrix of B-block combinations)
            # The base decomposition decomposes Xi*Yi into 7 rank-1 terms
            # Each term has coefficients from u_base, v_base, w_base
            
            # Let's define the block mappings for each Mi
            
            # A-block indices: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
            # In 4x4 indexing: A11 covers A[0:2,0:2], A12 covers A[0:2,2:4], etc.
            # A[i,j] in 4x4 corresponds to:
            #   If i<2, j<2: A11[i,j]
            #   If i<2, j>=2: A12[i,j-2]
            #   If i>=2, j<2: A21[i-2,j]
            #   If i>=2, j>=2: A22[i-2,j-2]
            
            # Similarly for B and C.
            
            # For each base entry, we need to determine which 4x4 positions it maps to
            
            # Let's define helper functions to get the 4x4 index from 2x2 block index
            
            def a_4x4_idx(block_i, block_j, local_i, local_j):
                """Get 4x4 index from block indices."""
                return (block_i * 2 + local_i) * 4 + (block_j * 2 + local_j)
            
            def b_4x4_idx(block_j, block_k, local_j, local_k):
                """Get 4x4 index from block indices for B."""
                return (block_j * 2 + local_j) * 4 + (block_k * 2 + local_k)
            
            def c_4x4_idx(block_i, block_k, local_i, local_k):
                """Get 4x4 index from block indices for C."""
                return (block_k * 2 + local_k) * 4 + (block_i * 2 + local_i)
            
            # Now define the block combinations for each Mi
            
            # Mi=0 (M1): Xi = A11 + A22, Yi = B11 + B22
            # Contributes to: C11 (+), C22 (+)
            if mi == 0:
                # Xi block coefficients: A11: +1, A22: +1
                # Yi block coefficients: B11: +1, B22: +1
                # C contributions: C11: +1, C22: +1
                
                for li in range(2):
                    for lj in range(2):
                        # A11 contribution
                        idx_a11 = a_4x4_idx(0, 0, li, lj)
                        U[idx_a11][col] += u_base[li*2 + lj][base_col] * 1
                        # A22 contribution
                        idx_a22 = a_4x4_idx(1, 1, li, lj)
                        U[idx_a22][col] += u_base[li*2 + lj][base_col] * 1
                
                for lj in range(2):
                    for lk in range(2):
                        # B11 contribution
                        idx_b11 = b_4x4_idx(0, 0, lj, lk)
                        V[idx_b11][col] += v_base[lj*2 + lk][base_col] * 1
                        # B22 contribution
                        idx_b22 = b_4x4_idx(1, 1, lj, lk)
                        V[idx_b22][col] += v_base[lj*2 + lk][base_col] * 1
                
                for li in range(2):
                    for lk in range(2):
                        # C11 contribution
                        idx_c11 = c_4x4_idx(0, 0, li, lk)
                        W[idx_c11][col] += w_base[lk*2 + li][base_col] * 1
                        # C22 contribution
                        idx_c22 = c_4x4_idx(1, 1, li, lk)
                        W[idx_c22][col] += w_base[lk*2 + li][base_col] * 1
            
            # Mi=1 (M2): Xi = A21 + A22, Yi = B11
            # Contributes to: C21 (+)
            elif mi == 1:
                # Xi block coefficients: A21: +1, A22: +1
                # Yi block coefficients: B11: +1
                # C contributions: C21: +1
                
                for li in range(2):
                    for lj in range(2):
                        # A21 contribution
                        idx_a21 = a_4x4_idx(1, 0, li, lj)
                        U[idx_a21][col] += u_base[li*2 + lj][base_col] * 1
                        # A22 contribution
                        idx_a22 = a_4x4_idx(1, 1, li, lj)
                        U[idx_a22][col] += u_base[li*2 + lj][base_col] * 1
                
                for lj in range(2):
                    for lk in range(2):
                        # B11 contribution
                        idx_b11 = b_4x4_idx(0, 0, lj, lk)
                        V[idx_b11][col] += v_base[lj*2 + lk][base_col] * 1
                
                for li in range(2):
                    for lk in range(2):
                        # C21 contribution
                        idx_c21 = c_4x4_idx(1, 0, li, lk)
                        W[idx_c21][col] += w_base[lk*2 + li][base_col] * 1
            
            # Mi=2 (M3): Xi = A11, Yi = B11 - B21
            # Contributes to: C12 (+)
            elif mi == 2:
                # Xi block coefficients: A11: +1
                # Yi block coefficients: B11: +1, B21: -1
                # C contributions: C12: +1
                
                for li in range(2):
                    for lj in range(2):
                        # A11 contribution
                        idx_a11 = a_4x4_idx(0, 0, li, lj)
                        U[idx_a11][col] += u_base[li*2 + lj][base_col] * 1
                
                for lj in range(2):
                    for lk in range(2):
                        # B11 contribution
                        idx_b11 = b_4x4_idx(0, 0, lj, lk)
                        V[idx_b11][col] += v_base[lj*2 + lk][base_col] * 1
                        # B21 contribution
                        idx_b21 = b_4x4_idx(1, 0, lj, lk)
                        V[idx_b21][col] += v_base[lj*2 + lk][base_col] * (-1)
                
                for li in range(2):
                    for lk in range(2):
                        # C12 contribution
                        idx_c12 = c_4x4_idx(0, 1, li, lk)
                        W[idx_c12][col] += w_base[lk*2 + li][base_col] * 1
            
            # Mi=3 (M4): Xi = A22, Yi = B21 - B22
            # Contributes to: C11 (+), C21 (+)
            elif mi == 3:
                # Xi block coefficients: A22: +1
                # Yi block coefficients: B21: +1, B22: -1
                # C contributions: C11: +1, C21: +1
                
                for li in range(2):
                    for lj in range(2):
                        # A22 contribution
                        idx_a22 = a_4x4_idx(1, 1, li, lj)
                        U[idx_a22][col] += u_base[li*2 + lj][base_col] * 1
                
                for lj in range(2):
                    for lk in range(2):
                        # B21 contribution
                        idx_b21 = b_4x4_idx(1, 0, lj, lk)
                        V[idx_b21][col] += v_base[lj*2 + lk][base_col] * 1
                        # B22 contribution
                        idx_b22 = b_4x4_idx(1, 1, lj, lk)
                        V[idx_b22][col] += v_base[lj*2 + lk][base_col] * (-1)
                
                for li in range(2):
                    for lk in range(2):
                        # C11 contribution
                        idx_c11 = c_4x4_idx(0, 0, li, lk)
                        W[idx_c11][col] += w_base[lk*2 + li][base_col] * 1
                        # C21 contribution
                        idx_c21 = c_4x4_idx(1, 0, li, lk)
                        W[idx_c21][col] += w_base[lk*2 + li][base_col] * 1
            
            # Mi=4 (M5): Xi = A11 + A12, Yi = B22
            # Contributes to: C11 (-), C12 (+)
            elif mi == 4:
                # Xi block coefficients: A11: +1, A12: +1
                # Yi block coefficients: B22: +1
                # C contributions: C11: -1, C12: +1
                
                for li in range(2):
                    for lj in range(2):
                        # A11 contribution
                        idx_a11 = a_4x4_idx(0, 0, li, lj)
                        U[idx_a11][col] += u_base[li*2 + lj][base_col] * 1
                        # A12 contribution
                        idx_a12 = a_4x4_idx(0, 1, li, lj)
                        U[idx_a12][col] += u_base[li*2 + lj][base_col] * 1
                
                for lj in range(2):
                    for lk in range(2):
                        # B22 contribution
                        idx_b22 = b_4x4_idx(1, 1, lj, lk)
                        V[idx_b22][col] += v_base[lj*2 + lk][base_col] * 1
                
                for li in range(2):
                    for lk in range(2):
                        # C11 contribution
                        idx_c11 = c_4x4_idx(0, 0, li, lk)
                        W[idx_c11][col] += w_base[lk*2 + li][base_col] * (-1)
                        # C12 contribution
                        idx_c12 = c_4x4_idx(0, 1, li, lk)
                        W[idx_c12][col] += w_base[lk*2 + li][base_col] * 1
            
            # Mi=5 (M6): Xi = A21 - A11, Yi = B11 + B12
            # Contributes to: C22 (+)
            elif mi == 5:
                # Xi block coefficients: A21: +1, A11: -1
                # Yi block coefficients: B11: +1, B12: +1
                # C contributions: C22: +1
                
                for li in range(2):
                    for lj in range(2):
                        # A21 contribution
                        idx_a21 = a_4x4_idx(1, 0, li, lj)
                        U[idx_a21][col] += u_base[li*2 + lj][base_col] * 1
                        # A11 contribution
                        idx_a11 = a_4x4_idx(0, 0, li, lj)
                        U[idx_a11][col] += u_base[li*2 + lj][base_col] * (-1)
                
                for lj in range(2):
                    for lk in range(2):
                        # B11 contribution
                        idx_b11 = b_4x4_idx(0, 0, lj, lk)
                        V[idx_b11][col] += v_base[lj*2 + lk][base_col] * 1
                        # B12 contribution
                        idx_b12 = b_4x4_idx(0, 1, lj, lk)
                        V[idx_b12][col] += v_base[lj*2 + lk][base_col] * 1
                
                for li in range(2):
                    for lk in range(2):
                        # C22 contribution
                        idx_c22 = c_4x4_idx(1, 1, li, lk)
                        W[idx_c22][col] += w_base[lk*2 + li][base_col] * 1
            
            # Mi=6 (M7): Xi = A12 - A22, Yi = B21 + B22
            # Contributes to: C11 (+), C22 (+)
            elif mi == 6:
                # Xi block coefficients: A12: +1, A22: -1
                # Yi block coefficients: B21: +1, B22: +1
                # C contributions: C11: +1, C22: +1
                
                for li in range(2):
                    for lj in range(2):
                        # A12 contribution
                        idx_a12 = a_4x4_idx(0, 1, li, lj)
                        U[idx_a12][col] += u_base[li*2 + lj][base_col] * 1
                        # A22 contribution
                        idx_a22 = a_4x4_idx(1, 1, li, lj)
                        U[idx_a22][col] += u_base[li*2 + lj][base_col] * (-1)
                
                for lj in range(2):
                    for lk in range(2):
                        # B21 contribution
                        idx_b21 = b_4x4_idx(1, 0, lj, lk)
                        V[idx_b21][col] += v_base[lj*2 + lk][base_col] * 1
                        # B22 contribution
                        idx_b22 = b_4x4_idx(1, 1, lj, lk)
                        V[idx_b22][col] += v_base[lj*2 + lk][base_col] * 1
                
                for li in range(2):
                    for lk in range(2):
                        # C11 contribution
                        idx_c11 = c_4x4_idx(0, 0, li, lk)
                        W[idx_c11][col] += w_base[lk*2 + li][base_col] * 1
                        # C22 contribution
                        idx_c22 = c_4x4_idx(1, 1, li, lk)
                        W[idx_c22][col] += w_base[lk*2 + li][base_col] * 1
            
            col += 1
    
    # Now we have a 4x4 * 4x4 decomposition with rank 49
    # But we need 4x4 * 4x8, so we need to extend B to 4x8 and C to 4x8
    
    # For the right half (columns 4-7 of B, columns 4-7 of C), we use schoolbook
    # Each entry B[j,k] for k in 4..7 and C[i,k] for k in 4..7 needs a separate multiplication
    # That's 4*4 = 16 more rank-1 terms
    
    # Actually, let's be more careful. We need:
    # C_left = A * B_left where B_left is 4x4 (columns 0-3)
    # C_right = A * B_right where B_right is 4x4 (columns 4-7)
    
    # C_left uses the Strassen decomposition above (rank 49)
    # C_right uses schoolbook: for each (i,j,k) with i in 0..3, j in 0..3, k in 4..7
    # That's 4*4*4 = 64 more rank-1 terms
    
    # Total rank: 49 + 64 = 113, which is worse than schoolbook 84
    
    # Let me reconsider. Maybe I should use Strassen for both halves.
    
    # Actually, the issue is that we're padding n from 3 to 4 and p from 7 to 8.
    # We only need the first 3 rows and first 7 columns of the output.
    
    # Let's try a different approach: use Strassen for the 4x4 * 4x4 part (left half),
    # and schoolbook for the 4x4 * 4x4 part (right half), but only keep the needed entries.
    
    # Actually, this is getting too complex. Let me just verify if the 4x4*4x4 Strassen works.
    
    # First, let's check if our 4x4*4x4 decomposition is valid
    if not _verify(U, V, W, 4, 4, 4):
        return None
    
    # Now extend to 4x4 * 4x8
    # We need to add entries for the right half
    
    U_ext = [[0]*113 for _ in range(16)]
    V_ext = [[0]*113 for _ in range(32)]
    W_ext = [[0]*113 for _ in range(32)]
    
    # Copy the left half (columns 0-3 of B, columns 0-3 of C)
    for i in range(16):
        for r in range(49):
            U_ext[i][r] = U[i][r]
    for j in range(16):
        for r in range(49):
            V_ext[j][r] = V[j][r]
    for k in range(16):
        for r in range(49):
            W_ext[k][r] = W[k][r]
    
    # Add schoolbook for the right half (columns 4-7 of B, columns 4-7 of C)
    col = 49
    for i in range(4):
        for j in range(4):
            for k in range(4, 8):
                U_ext[i*4 + j][col] = 1
                V_ext[j*8 + k][col] = 1
                W_ext[k*4 + i][col] = 1
                col += 1
    
    # Now we have a 4x4 * 4x8 decomposition with rank 113
    # But we only need n=3, p=7, so we can trim
    
    # Trim to n=3, p=7:
    # U: keep rows 0..11 (i in 0..2, j in 0..3)
    # V: keep rows 0..27 (j in 0..3, k in 0..6)
    # W: keep rows 0..20 (k in 0..6, i in 0..2)
    
    U_trim = [[U_ext[i*4 + j][r] for r in range(113)] for i in range(3) for j in range(4)]
    V_trim = [[V_ext[j*8 + k][r] for r in range(113)] for j in range(4) for k in range(7)]
    W_trim = [[W_ext[k*4 + i][r] for r in range(113)] for k in range(7) for i in range(3)]
    
    # Verify
    if not _verify(U_trim, V_trim, W_trim, 3, 4, 7):
        return None
    
    return {"U": U_trim, "V": V_trim, "W": W_trim}

def construct():
    t0 = time.time()
    DEADLINE = 140.0
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 84
    
    # Try Strassen-style decomposition
    if time.time() - t0 < DEADLINE - 10:
        try:
            strassen_result = _strassen_decomposition(n, m, p)
            if strassen_result is not None:
                rank = len(strassen_result["U"][0])
                if rank < best_rank:
                    best = strassen_result
                    best_rank = rank
        except Exception:
            pass
    
    return best
# EVOLVE-BLOCK-END
