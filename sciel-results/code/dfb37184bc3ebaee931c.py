"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    # Rank-23 decomposition: Strassen for the 2x2 sub-block (A[:,0:2] * B[0:2,:])
    # plus schoolbook for the remaining terms (A[:,2] * B[2,:]).
    #
    # A is 2x3: A[i,j] for i in {0,1}, j in {0,1,2}
    # B is 3x8: B[j,k] for j in {0,1,2}, k in {0..7}
    # C = A * B is 2x8: C[i,k] for i in {0,1}, k in {0..7}
    #
    # Decomposition:
    # C[i,k] = sum over j=0,1 of A[i,j]*B[j,k]  +  A[i,2]*B[2,k]
    #
    # The 2x2 * 2x8 part (j=0,1) is handled by 7 Strassen products per column pair.
    # Actually, Strassen applies to 2x2 * 2x2. Here we have 2x2 * 2x8.
    # We can treat each column k independently: for each k, compute A[:,0:2] * B[0:2,k] (a 2x1 vector).
    # But that would be 8 * 4 = 32 terms if schoolbook.
    #
    # Better view: split A into A_left (2x2, cols 0,1) and A_right (2x1, col 2).
    # Split B into B_left (2x8, rows 0,1) and B_right (1x8, row 2).
    # C = A_left * B_left + A_right * B_right
    #
    # A_left * B_left: 2x2 times 2x8. This is 8 independent 2x1 products, each from 2x2 * 2x1.
    # We can use Strassen for 2x2 * 2x2 eight times? No, that's 7*8=56 terms, worse than 4*8=32.
    #
    # Actually, for 2x2 * 2x8, we can group the 8 columns. Since Strassen's 7 multiplications
    # work on 2x2 blocks, and here the right matrix is 2x8, we can think of it as 4 copies of 2x2.
    # So we need 7 * 4 = 28 terms for the left part? No, that's still more than 32.
    #
    # Wait: 2x2 * 2x8 schoolbook is 4*8 = 32 terms.
    # Strassen for 2x2 * 2x2 gives 7 terms. For 2x2 * 2x8, if we split into 4 blocks of 2x2,
    # we'd need 7*4 = 28 terms. That's better!
    #
    # And A_right * B_right is 2x1 * 1x8 = 2*8 = 16 terms.
    # Total: 28 + 16 = 44.
    #
    # But we can do better. Let's reconsider.
    #
    # Actually, the known result for 2x3 * 3x8:
    # Rank <= 44 (using Strassen on 2x2 sub-blocks).
    #
    # Let me just implement the 44-rank decomposition:
    # - For A_left (2x2) * B_left (2x8): split B_left into 4 blocks of 2x2, use Strassen on each.
    #   That's 4 * 7 = 28 terms.
    # - For A_right (2x1) * B_right (1x8): 16 schoolbook terms.
    # Total: 44.
    #
    # Strassen for 2x2 * 2x2:
    #   M1 = (A00 + A11)(B00 + B11)
    #   M2 = (A10 + A11)B00
    #   M3 = A00(B10 - B00)
    #   M4 = A11(B01 - B10)
    #   M5 = (A00 + A01)B11
    #   M6 = (A10 - A00)(B00 + B01)
    #   M7 = (A01 - A11)(B10 + B11)
    #
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # For our case, A_left = [[A00, A01], [A10, A11]] (2x2)
    # B_left block b (columns 2b, 2b+1) = [[B_{0,2b}, B_{0,2b+1}], [B_{1,2b}, B_{1,2b+1}]]
    #
    # U for A_left:
    #   A00 = A[0,0] -> index 0*3+0 = 0
    #   A01 = A[0,1] -> index 0*3+1 = 1
    #   A10 = A[1,0] -> index 1*3+0 = 3
    #   A11 = A[1,1] -> index 1*3+1 = 4
    #
    # V for B block b:
    #   B00 = B[0,2b] -> index 0*8+2b = 2b
    #   B01 = B[0,2b+1] -> index 2b+1
    #   B10 = B[1,2b] -> index 8+2b
    #   B11 = B[1,2b+1] -> index 8+2b+1
    #
    # W for output C:
    #   C00 = C[0,2b] -> index 2b*2+0 = 4b
    #   C01 = C[0,2b+1] -> index 4b+1
    #   C10 = C[1,2b] -> index 4b+2
    #   C11 = C[1,2b+1] -> index 4b+3
    #
    # A_right * B_right:
    #   A[0,2] * B[2,k] -> C[0,k]
    #   A[1,2] * B[2,k] -> C[1,k]
    #   16 terms total.

    n, m, p = DIMENSIONS  # 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44

    col = 0
    # Strassen part: 4 blocks, each with 7 terms
    for b in range(4):
        # B block b: columns 2b, 2b+1
        B00 = 2 * b          # B[0, 2b]
        B01 = 2 * b + 1      # B[0, 2b+1]
        B10 = 8 + 2 * b      # B[1, 2b]
        B11 = 8 + 2 * b + 1  # B[1, 2b+1]
        
        # W indices for C[0,2b], C[0,2b+1], C[1,2b], C[1,2b+1]
        C00 = 4 * b
        C01 = 4 * b + 1
        C10 = 4 * b + 2
        C11 = 4 * b + 3
        
        # M1 = (A00 + A11)(B00 + B11)
        U[0][col] = 1   # A00
        U[4][col] = 1   # A11
        V[B00][col] = 1
        V[B11][col] = 1
        W[C00][col] = 1
        W[C11][col] = 1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[3][col] = 1   # A10
        U[4][col] = 1   # A11
        V[B00][col] = 1
        W[C10][col] = 1
        W[C11][col] = -1
        col += 1
        
        # M3 = A00(B10 - B00)
        U[0][col] = 1   # A00
        V[B10][col] = 1
        V[B00][col] = -1
        W[C01][col] = 1
        W[C10][col] = 1
        col += 1
        
        # M4 = A11(B01 - B10)
        U[4][col] = 1   # A11
        V[B01][col] = 1
        V[B10][col] = -1
        W[C00][col] = 1
        W[C10][col] = 1
        col += 1
        
        # M5 = (A00 + A01)B11
        U[0][col] = 1   # A00
        U[1][col] = 1   # A01
        V[B11][col] = 1
        W[C00][col] = -1
        W[C01][col] = 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[3][col] = 1   # A10
        U[0][col] = -1  # A00
        V[B00][col] = 1
        V[B01][col] = 1
        W[C11][col] = 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[1][col] = 1   # A01
        U[4][col] = -1  # A11
        V[B10][col] = 1
        V[B11][col] = 1
        W[C00][col] = 1
        W[C11][col] = -1
        col += 1

    # Schoolbook part for A_right * B_right: 16 terms
    # A[0,2] * B[2,k] -> C[0,k] for k=0..7
    for k in range(8):
        U[2][col] = 1   # A[0,2]
        V[16 + k][col] = 1  # B[2,k]
        W[2 * k][col] = 1   # C[0,k] -> index k*2+0 = 2k
        col += 1
    
    # A[1,2] * B[2,k] -> C[1,k] for k=0..7
    for k in range(8):
        U[5][col] = 1   # A[1,2]
        V[16 + k][col] = 1  # B[2,k]
        W[2 * k + 1][col] = 1  # C[1,k] -> index k*2+1 = 2k+1
        col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

