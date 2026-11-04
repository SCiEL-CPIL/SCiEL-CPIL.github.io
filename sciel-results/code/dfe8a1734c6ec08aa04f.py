"""Matrix multiplication 2x4x7 - rank-minimal exact tensor decomposition."""
import time

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 2, 4, 7
    start_time = time.time()
    
    # We will build a decomposition using a state-space / sliding-window approach
    # that exploits the n=2 structure.
    #
    # The tensor T(A, B, C) = C = A B has indices:
    #   A: (i, j) where i in 0..1, j in 0..3
    #   B: (j, k) where j in 0..3, k in 0..6
    #   C: (i, k) where i in 0..1, k in 0..6
    #
    # The naive rank is n*m*p = 2*4*7 = 56 (schoolbook).
    #
    # Strategy: We decompose the multiplication into overlapping 2-column windows
    # of B. For each pair of adjacent columns (k, k+1) of B, we form a 2x2 sub-matrix
    # B_sub = [[B[j,k], B[j,k+1]] for j in {j0, j0+1}]. 
    #
    # Actually, let's think differently. The key insight for n=2 is that we can
    # write C = A * B as a sum of terms where we process B in 2-column strips.
    #
    # Let's partition B into overlapping 2x2 blocks. Since m=4, we have 2 rows of 2x2 blocks.
    # Since p=7, we have 3 full 2-column strips (k=0-1, 2-3, 4-5) and 1 leftover column (k=6).
    #
    # For each 2x2 block of B (rows j0..j0+1, cols k0..k0+1), and for each 2x2 block of A
    # (rows i0..i0+1, cols j0..j0+1), we can use a 7-term Strassen decomposition.
    #
    # But wait, A is 2x4, so it has 2x2 blocks in columns: (j=0,1) and (j=2,3).
    # B is 4x7. Its 2x2 blocks in columns: (k=0,1), (k=2,3), (k=4,5), and single column k=6.
    #
    # For each pair (A_block, B_block) where A_block is 2x2 and B_block is 2x2,
    # we get a 2x2 product C_sub. We can compute this with 7 Strassen terms.
    #
    # Number of (A_block, B_block) pairs:
    #   A has 2 column-blocks (cols 0-1, 2-3)
    #   B has 3 column-blocks (cols 0-1, 2-3, 4-5) + 1 single column (col 6)
    #
    # For the 2x2 B blocks: 2 * 3 = 6 pairs, each using 7 Strassen terms = 42
    # For the single column B (k=6): 2 * 1 = 2 pairs, each using 4 schoolbook terms = 8
    # Total = 42 + 8 = 50
    #
    # This is the same as the current program. Let me try a different approach.
    #
    # Alternative: Use the fact that n=2 to find a lower-rank decomposition.
    #
    # Let's try to use a rank-45 or lower decomposition by sharing intermediate
    # products across the overlapping 2-column windows.
    #
    # Idea: Process B in 2-column strips with overlap. For columns (0,1), (1,2), (2,3),
    # (3,4), (4,5), (5,6), we have 6 overlapping 2-column windows.
    #
    # For each window (k, k+1), we form B[:, k:k+2] which is 4x2.
    # Then C[:, k:k+2] = A * B[:, k:k+2].
    #
    # A is 2x4, B_window is 4x2, so C_window is 2x2.
    #
    # For each window, we can decompose the 2x4x2 multiplication.
    # The rank of 2x4x2 multiplication: We can split A into two 2x2 blocks A1, A2
    # and B_window into B1 (top 2 rows) and B2 (bottom 2 rows).
    # C = A1*B1 + A2*B2.
    # A1*B1 is 2x2x2 (rank 7 with Strassen), A2*B2 is 2x2x2 (rank 7 with Strassen).
    # Total per window: 14 terms.
    #
    # But the windows overlap! C[:, k] appears in windows (k-1, k) and (k, k+1).
    # So we're double-counting the middle columns.
    #
    # Let's try yet another approach: direct rank minimization via the structure.
    #
    # The tensor for 2x4x7 has 56 entries. Let's try to find a decomposition
    # with rank < 50.
    #
    # I'll construct a decomposition using 45 terms by exploiting the n=2 structure
    # more aggressively.
    
    # Let's use a decomposition based on the following insight:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    #
    # For i=0,1 and k=0..6:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    #
    # Let's define:
    # D[k] = B[0,k] - B[2,k]  (for k=0..6)
    # E[k] = B[1,k] - B[3,k]  (for k=0..6)
    #
    # Then:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    #        = A[0,0]*(B[0,k]) + A[0,1]*(B[1,k]) + A[0,2]*(B[2,k]) + A[0,3]*(B[3,k])
    #
    # Let me try a different decomposition. I'll use 48 terms:
    # 6 Strassen blocks for the 2x2x2 sub-problems, plus 48-42=6 extra terms.
    
    # Actually, let's just build a correct rank-48 decomposition.
    # 
    # Decomposition:
    # Split into 2x2 blocks:
    # A = [A11 A12] where A11 is 2x2 (cols 0-1), A12 is 2x2 (cols 2-3)
    # B = [B1; B2] where B1 is 2x7 (rows 0-1), B2 is 2x7 (rows 2-3)
    # C = A11 * B1 + A12 * B2
    #
    # A11*B1: 2x2 * 2x7 = 2x7. Split B1 into 2x2 blocks: B1a (cols 0-1), B1b (cols 2-3), B1c (cols 4-5), B1d (col 6)
    # A11*B1a: 2x2x2, rank 7
    # A11*B1b: 2x2x2, rank 7
    # A11*B1c: 2x2x2, rank 7
    # A11*B1d: 2x2x1, rank 4
    # Subtotal: 25
    #
    # A12*B2: similarly, 25
    #
    # Total: 50
    #
    # To get 48, I need to save 2 terms. Let's try to merge the 2x2x1 parts.
    #
    # A11*B1d: C[0,6] += A11[0,0]*B1[0,6] + A11[0,1]*B1[1,6]
    #           C[1,6] += A11[1,0]*B1[0,6] + A11[1,1]*B1[1,6]
    #
    # A12*B2d: C[0,6] += A12[0,0]*B2[0,6] + A12[0,1]*B2[1,6]
    #           C[1,6] += A12[1,0]*B2[0,6] + A12[1,1]*B2[1,6]
    #
    # Combined for C[:,6]:
    # C[0,6] = (A11[0,0]+A12[0,0])*B[0,6] + (A11[0,1]+A12[0,1])*B[1,6] + (A11[1,0]+A12[1,0])*B[2,6] + (A11[1,1]+A12[1,1])*B[3,6]
    #
    # That's still 4 terms for C[:,6].
    #
    # Let me just implement the rank-50 decomposition correctly and verify it.
    
    rank = 50
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A11+A22)(B11+B22)
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M2: (A21+A22)B11
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] -= 1
        col += 1
        
        # M3: A11(B12-B22)
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M4: A22(B21-B11)
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # M5: (A11+A12)B22
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
        # M6: (A21-A11)(B11+B12)
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M7: (A12-A22)(B21+B22)
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1

    # 2x2 blocks of B in columns: (0,1), (2,3), (4,5)
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            add_strassen(0, j0, k0)
    
    # Single column k=6
    for j in range(4):
        for i in range(2):
            U[i*m + j][col] = 1.0
            V[j*p + 6][col] = 1.0
            W[6*n + i][col] = 1.0
            col += 1
    
    if time.time() - start_time > 100:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
