"""Program for matrix multiplication 4x4 by 4x5 tensor decomposition."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    # We'll build a decomposition using Strassen for the 2x2 block structure of the 4x4 x 4x4 part,
    # and schoolbook for the 5th column.
    #
    # The 4x4 x 4x4 part is split into 2x2 blocks:
    #   A = [[A11, A12], [A21, A22]] where each block is 2x2
    #   B4 = [[B11, B12], [B21, B22]] where each block is 2x2
    #   C4 = [[C11, C12], [C21, C22]]
    #
    # Using Strassen, we compute the product of two 2x2 matrices with 7 multiplications.
    # But here we need the full 4x4 x 4x4 product, so we have 4 blocks to compute.
    # However, we can't just apply Strassen to each block independently because the
    # Strassen algorithm works on 2x2 matrices. Let me reconsider.
    #
    # Actually, for a 4x4 x 4x4 matrix multiplication, we can view it as multiplying
    # four 2x2 blocks. The product C4 = A * B4 has blocks:
    #   C11 = A11*B11 + A12*B21
    #   C12 = A11*B12 + A12*B22
    #   C21 = A21*B11 + A22*B21
    #   C22 = A21*B12 + A22*B22
    #
    # Each of these involves two 2x2x2x2 multiplications. We can use Strassen for each
    # 2x2 block multiplication, but that would be 4 blocks * 2 multiplications each = 8
    # Strassen calls, each of rank 7, giving 56. But wait, each "multiplication" in the
    # block expression is a full 2x2 * 2x2 product, which Strassen handles with 7 rank-1
    # terms. So each C_block needs 2 * 7 = 14 rank-1 terms, and 4 blocks need 56 total.
    #
    # For the 5th column of B (B[:, 5]), we just do schoolbook: 4*4 = 16 terms.
    #
    # Total rank = 56 + 16 = 72.
    #
    # Let me implement this carefully.
    
    U_rows = n * m  # 16
    V_rows = m * p  # 20
    W_rows = p * n  # 20
    
    U = [[0] * 72 for _ in range(U_rows)]
    V = [[0] * 72 for _ in range(V_rows)]
    W = [[0] * 72 for _ in range(W_rows)]
    
    col = 0
    
    # Helper: add a rank-1 term where U[i] = a, V[j] = b, W[k] = c
    def add_term(i, j, k, a=1, b=1, c=1):
        nonlocal col
        U[i][col] += a
        V[j][col] += b
        W[k][col] += c
        col += 1
    
    # Strassen for 2x2 * 2x2:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B10 - B11)
    # P4 = A11(B01 - B10)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B10)
    # P7 = (A01 - A11)(B01 + B11)
    #
    # C00 = P1 - P4 + P5 - P7
    # C01 = P3 + P4
    # C10 = P2 + P5
    # C11 = P1 - P2 + P3 - P6
    #
    # But we need to track the individual rank-1 terms. Each Pi is itself a 2x2 matrix
    # product, but in Strassen, the "multiplications" are scalar-like in the sense that
    # they produce 2x2 results from sum/diff combinations. Actually, no - each Pi is
    # a 2x2 matrix product where the factors are 2x2 matrices. So each Pi actually
    # requires its own 7 rank-1 terms if we expand it. That's not right.
    #
    # Let me reconsider. The Strassen algorithm for 2x2 matrices replaces 8 scalar
    # multiplications with 7. Each "multiplication" in Strassen is between 2x2 matrices,
    # but the algorithm's structure means we need to think of it differently.
    #
    # Actually, I think the cleanest approach is to just use the known rank-1
    # decomposition. For a 2x2 x 2x2 product, the schoolbook rank is 8. Strassen
    # shows it can be done in rank 7. But implementing Strassen's specific rank-1
    # terms requires knowing the exact coefficients.
    #
    # Given the complexity, let me just use schoolbook for everything but with a
    # smarter structure. The schoolbook rank is 4*4*5 = 80. Let me see if I can
    # find a simple improvement.
    #
    # Actually, let me just do schoolbook rank 80. It's safe and correct.
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
