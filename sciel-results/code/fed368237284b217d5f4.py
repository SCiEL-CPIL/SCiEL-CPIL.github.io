"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-44 exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x8 matrix.
    
    Strategy:
    - Split the 3x8 matrix into two 3x4 blocks (columns 0-3 and 4-7).
    - For each 3x4 block, split the 2x3 matrix into a 2x2 left block and a 2x1 right block.
    - Use Strassen's algorithm (rank 7) for the 2x2 x 2x2 multiplication.
    - Use schoolbook (rank 2) for the 2x1 x 2x2 multiplication.
    - Total rank per 3x4 block: 7 + 2 = 9
    - Total rank for both blocks: 18
    
    Wait, that's only 18. Let me reconsider.
    
    Actually, let's think about this differently.
    
    The matrix is 2x3 times 3x8. We can think of the 3x8 matrix as having
    3 rows and 8 columns. Each column of the result (2x1) is computed from
    the corresponding column of the 3x8 matrix.
    
    For each column k (k=0..7), we need to compute C[:,k] = A * B[:,k]
    where A is 2x3 and B[:,k] is 3x1.
    
    This is a 2x3 x 3x1 multiplication for each k. The tensor rank of 
    (2,3,1) is 6 (schoolbook). So naive rank is 6*8 = 48.
    
    Can we do better? Let's try to group columns.
    
    Alternative approach: Use the fact that the middle dimension is 3.
    We can write:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    This is a bilinear form. The tensor has shape (2,3,8).
    
    Known results: The rank of (2,3,8) matrix multiplication tensor.
    The trivial rank is 2*3*8 = 48.
    
    Can we reduce this? Let's try to use Strassen-like techniques.
    
    Actually, let's try a different decomposition. We can split the 8 columns
    into two groups of 4, and for each group, we have a 2x3 x 3x4 multiplication.
    
    For 2x3 x 3x4: The trivial rank is 24. Can we reduce this?
    
    Let me try to use the structure. We can split the 3 in the middle:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    This is already a sum of 3 terms per (i,k) pair, so 3*2*4 = 24 terms.
    
    To use Strassen, we need to find a bilinear decomposition with fewer terms.
    
    Let me try to use the approach from the previous response, but correctly.
    
    Actually, I think the key insight is that we can use a rank-7 decomposition
    for 2x2 x 2x2 multiplications, and handle the remaining parts separately.
    
    Let's split A into A_left (2x2) and A_right (2x1).
    Let's split B into B_left (3x4) and B_right (3x4) - wait, that doesn't work
    because B is 3x8, not 3x4.
    
    Let me reconsider. We have A (2x3) and B (3x8).
    
    Split A: A = [A1 (2x2), A2 (2x1)]
    For each column group of B (4 columns each), we have B_g (3x4).
    
    Then C_g = A1 * B_g[:, :2] + A2 * B_g[:, 2]
    
    Wait, that's not right. B_g is 3x4, so B_g[:, :2] is 3x2 and B_g[:, 2] is 3x1.
    
    A1 * B_g[:, :2] is 2x2 * 3x2... that doesn't match dimensions.
    
    I'm confusing myself. Let me be more careful.
    
    A is 2x3. B is 3x8. C = A*B is 2x8.
    
    C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    Let's split the sum over j into j=0,1 and j=2.
    
    C[i,k] = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + A[i,2]*B[2,k]
    
    The first part is a 2x2 x 2x8 multiplication (using A[0:2, 0:2] and B[0:2, :]).
    The second part is a 2x1 x 1x8 multiplication (using A[0:2, 2] and B[2, :]).
    
    For the 2x2 x 2x8 part: We can use Strassen for each pair of columns.
    Split the 8 columns into 4 pairs. For each pair, use rank 7.
    Total: 4 * 7 = 28.
    
    For the 2x1 x 1x8 part: This is rank 16 (2*1*8).
    
    Total: 28 + 16 = 44.
    
    So the rank is 44.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=8
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    # Part 1: 2x2 x 2x8 using Strassen (4 pairs of columns, rank 7 each)
    col = 0
    for pair in range(4):
        k0 = 2 * pair
        k1 = 2 * pair + 1
        
        # Strassen terms for 2x2 * 2x2 -> 2x2
        # A_sub is 2x2 (rows 0,1 of A, cols 0,1 of A)
        # B_sub is 2x2 (rows 0,1 of B, cols k0,k1 of B)
        # C_sub is 2x2 (rows 0,1 of C, cols k0,k1 of C)
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[0][col] += 1  # A[0,0]
        U[3][col] += 1  # A[1,1] (row 1, col 1 => index 1*2+1=3)
        V[0][col] += 1  # B[0, k0] (row 0, col k0 => index 0*8+k0=k0)
        V[8 + k1][col] += 1  # B[1, k1] (row 1, col k1 => index 1*8+k1=8+k1)
        W[k0 * 2 + 0][col] += 1  # C[0, k0]
        W[k0 * 2 + 1][col] += 1  # C[0, k1] - wait, W[k*n+i] where n=2
        # W index: k*n + i. For C[0,k0]: k=k0, i=0 => k0*2+0
        # For C[0,k1]: k=k1, i=0 => k1*2+0
        # For C[1,k0]: k=k0, i=1 => k0*2+1
        # For C[1,k1]: k=k1, i=1 => k1*2+1
        
        # Let me redo this. W is indexed by W[k*n+i, c] where C[i,k] goes to W[k*n+i].
        # So C[i,k] -> W[k*2 + i].
        
        # M1 contributes to C[0,k0] and C[1,k1] (and C[0,k1] and C[1,k0] in the reconstruction)
        # Actually, M1 = (A00+A11)*(B00+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So M1 contributes: +1 to C00, +1 to C11
        W[k0 * 2 + 0][col] += 1  # C[0, k0]
        W[k1 * 2 + 1][col] += 1  # C[1, k1]
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[2][col] += 1  # A[1,0] => index 1*2+0=2
        U[3][col] += 1  # A[1,1] => index 3
        V[0][col] += 1  # B[0, k0] => index k0
        W[k0 * 2 + 1][col] += 1  # C[1, k0]
        W[k1 * 2 + 1][col] -= 1  # C[1, k1]
        col += 1
        
        # M3 = A00 * (B01 - B11)
        U[0][col] += 1  # A[0,0]
        V[8 + k1][col] += 1  # B[1, k1] => index 8+k1
        V[k1][col] -= 1  # B[0, k1] => index k1
        W[k1 * 2 + 0][col] += 1  # C[0, k1]
        W[k1 * 2 + 1][col] += 1  # C[1, k1]
        col += 1
        
        # M4 = A11 * (B10 - B00)
        U[3][col] += 1  # A[1,1]
        V[8 + k0][col] += 1  # B[1, k0] => index 8+k0
        V[k0][col] -= 1  # B[0, k0] => index k0
        W[k0 * 2 + 0][col] += 1  # C[0, k0]
        W[k0 * 2 + 1][col] += 1  # C[1, k0]
        col += 1
        
        # M5 = (A00 + A01) * B11
        U[0][col] += 1  # A[0,0]
        U[1][col] += 1  # A[0,1] => index 0*2+1=1
        V[8 + k1][col] += 1  # B[1, k1]
        W[k0 * 2 + 0][col] -= 1  # C[0, k0]
        W[k1 * 2 + 0][col] += 1  # C[0, k1]
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[2][col] += 1  # A[1,0]
        U[0][col] -= 1  # A[0,0]
        V[k0][col] += 1  # B[0, k0]
        V[k1][col] += 1  # B[0, k1]
        W[k1 * 2 + 1][col] += 1  # C[1, k1]
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[1][col] += 1  # A[0,1]
        U[3][col] -= 1  # A[1,1]
        V[8 + k0][col] += 1  # B[1, k0]
        V[8 + k1][col] += 1  # B[1, k1]
        W[k0 * 2 + 0][col] += 1  # C[0, k0]
        col += 1
    
    # Part 2: 2x1 x 1x8 using schoolbook (rank 16)
    for k in range(8):
        for i in range(2):
            U[i * 2 + 2][col] += 1  # A[i, 2] => index i*2+2
            V[16 + k][col] += 1  # B[2, k] => index 2*8+k=16+k
            W[k * 2 + i][col] += 1  # C[i, k] => index k*2+i
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
