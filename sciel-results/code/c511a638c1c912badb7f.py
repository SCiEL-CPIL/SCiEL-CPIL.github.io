"""Strassen-block tensor decomposition for 3x3 * 3x7 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    deadline = time.time() + 140  # safety margin before 150s kill

    # We'll build a block decomposition:
    # A is 3x3, B is 3x7, C = A @ B is 3x7.
    #
    # Strategy: Split A into 2x2 blocks (rows 0,1 vs row 2; cols 0,1 vs col 2)
    # Split B into 2x2 blocks (rows 0,1 vs row 2; cols 0,1 vs cols 2..6)
    # Use Strassen for the 2x2 x 2x2 part, and schoolbook for the rest.
    #
    # Let's partition:
    # A: rows {0,1} x {2}, cols {0,1} x {2}
    #   A00 = A[0:2, 0:2]  (2x2)
    #   A01 = A[0:2, 2:3]  (2x1)
    #   A10 = A[2:3, 0:2]  (1x2)
    #   A11 = A[2:3, 2:3]  (1x1)
    #
    # B: rows {0,1} x {2}, cols {0,1} x {2..6}
    #   B00 = B[0:2, 0:2]  (2x2)
    #   B01 = B[0:2, 2:7]  (2x5)
    #   B10 = B[2:3, 0:2]  (1x2)
    #   B11 = B[2:3, 2:7]  (1x5)
    #
    # C = A @ B:
    #   C[0:2, 0:2]   = A00@B00 + A01@B10
    #   C[0:2, 2:7]   = A00@B01 + A01@B11
    #   C[2:3, 0:2]   = A10@B00 + A11@B10
    #   C[2:3, 2:7]   = A10@B01 + A11@B11
    #
    # For the 2x2 x 2x2 block (A00@B00), use Strassen: 7 multiplications
    # All other multiplications are schoolbook.
    #
    # Schoolbook terms:
    # A01@B10: (2x1)@(1x2) = 2*1*2 = 4 terms
    # A00@B01: (2x2)@(2x5) = 2*2*5 = 20 terms
    # A01@B11: (2x1)@(1x5) = 2*1*5 = 10 terms
    # A10@B00: (1x2)@(2x2) = 1*2*2 = 4 terms
    # A11@B10: (1x1)@(1x2) = 1*1*2 = 2 terms
    # A10@B01: (1x2)@(2x5) = 1*2*5 = 10 terms
    # A11@B11: (1x1)@(1x5) = 1*1*5 = 5 terms
    #
    # Total schoolbook: 4 + 20 + 10 + 4 + 2 + 10 + 5 = 55
    # Strassen for A00@B00: 7
    # Total rank: 55 + 7 = 62
    #
    # But wait, we can also split the 2x5 and 1x5 blocks further.
    # Let's try splitting B's columns into {0,1} x {2,3,4} x {5,6}
    # Actually, let's try a different approach: split into 2x2 blocks where possible.
    #
    # Alternative: Split A into 2x2 + 1x1, B into 2x5 + 1x5, but also split the 5-col parts.
    #
    # Let me try: split columns of B into {0,1}, {2,3,4}, {5,6}
    # And rows of A/B into {0,1}, {2}
    #
    # A00 = A[0:2,0:2], A01 = A[0:2,2:3]
    # A10 = A[2:3,0:2], A11 = A[2:3,2:3]
    #
    # B00 = B[0:2,0:2], B01 = B[0:2,2:5], B02 = B[0:2,5:7]
    # B10 = B[2:3,0:2], B11 = B[2:3,2:5], B12 = B[2:3,5:7]
    #
    # C[0:2,0:2] = A00@B00 + A01@B10
    # C[0:2,2:5] = A00@B01 + A01@B11
    # C[0:2,5:7] = A00@B02 + A01@B12
    # C[2:3,0:2] = A10@B00 + A11@B10
    # C[2:3,2:5] = A10@B01 + A11@B11
    # C[2:3,5:7] = A10@B02 + A11@B12
    #
    # Strassen for A00@B00: 7
    # A01@B10: 2*1*2 = 4
    # A00@B01: 2*2*3 = 12
    # A01@B11: 2*1*3 = 6
    # A00@B02: 2*2*2 = 8
    # A01@B12: 2*1*2 = 4
    # A10@B00: 1*2*2 = 4
    # A11@B10: 1*1*2 = 2
    # A10@B01: 1*2*3 = 6
    # A11@B11: 1*1*3 = 3
    # A10@B02: 1*2*2 = 4
    # A11@B12: 1*1*2 = 2
    #
    # Total: 7 + 4 + 12 + 6 + 8 + 4 + 4 + 2 + 6 + 3 + 4 + 2 = 62
    #
    # Hmm, same 62. Let me try Strassen on A00@B02 too (2x2 x 2x2):
    # Replace A00@B02 (8 terms) with Strassen (7 terms)
    # Total: 62 - 8 + 7 = 61
    #
    # So rank 61 is achievable.
    #
    # Let me implement this.

    # Indices:
    # U is indexed by (i*m + j, r) where i is row of A, j is col of A (= row of B)
    # V is indexed by (j*p + k, r) where j is row of B, k is col of B
    # W is indexed by (k*n + i, r) where k is col of B (= col of C), i is row of C

    # We'll build U, V, W as lists of lists, then convert.
    # Start with all zeros, fill in terms.

    # Let's use a helper to add a rank-1 term.
    # Each term: for a given (i, j, k) in the schoolbook, U[i*m+j] += u_coeff, V[j*p+k] += v_coeff, W[k*n+i] += w_coeff

    # For Strassen, we need to add multiple contributions per multiplication.

    # Let me define the structure more carefully.
    # I'll build arrays of size (n*m, rank), (m*p, rank), (p*n, rank)
    # and fill them column by column.

    rank = 61
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0

    def add_term(i, j, k, u_c=1.0, v_c=1.0, w_c=1.0):
        """Add a single rank-1 term for C[i,k] += u_c * A[i,j] * v_c * B[j,k] * w_c"""
        nonlocal col
        if col >= rank:
            raise ValueError("Rank overflow")
        U[i * m + j, col] = u_c
        V[j * p + k, col] = v_c
        W[k * n + i, col] = w_c
        col += 1

    def add_strassen_2x2x2(a_row0, a_row1, a_col0, a_col1, b_row0, b_row1, b_col0, b_col1, out_row0, out_row1, out_col0, out_col1):
        """
        Add Strassen decomposition for 2x2 * 2x2 block.
        A sub-block: rows a_row0, a_row1; cols a_col0, a_col1
        B sub-block: rows b_row0, b_row1; cols b_col0, b_col1
        C sub-block: rows out_row0, out_row1; cols out_col0, out_col1
        
        Strassen:
        M1 = (A00 + A11)(B11 - B00)
        M2 = (A10 + A11) B00
        M3 = A00 (B01 - B11)
        M4 = A11 (B00 + B10)
        M5 = (A00 + A01) B11
        M6 = (A10 - A00)(B00 + B01)
        M7 = (A01 - A11)(B10 + B11)
        
        C00 = M1 + M4 + M5 - M7
        C01 = M3 + M7
        C10 = M2 + M4
        C11 = M1 + M2 + M5 - M6
        """
        nonlocal col
        
        # M1 = (A[a_row0,a_col0] + A[a_row1,a_col1]) * (B[b_row1,b_col1] - B[b_row0,b_col0])
        # Contribution to C[out_row0,out_col0]: +M1
        # Contribution to C[out_row0,out_col1]: +M1
        # Contribution to C[out_row1,out_col0]: +M1
        # Contribution to C[out_row1,out_col1]: +M1
        U[a_row0 * m + a_col0, col] = 1
        U[a_row1 * m + a_col1, col] = 1
        V[b_row1 * p + b_col1, col] = 1
        V[b_row0 * p + b_col0, col] = -1
        W[out_col0 * n + out_row0, col] = 1
        W[out_col1 * n + out_row0, col] = 1
        W[out_col0 * n + out_row1, col] = 1
        W[out_col1 * n + out_row1, col] = 1
        col += 1

        # M2 = (A[a_row1,a_col0] + A[a_row1,a_col1]) * B[b_row0,b_col0]
        # Contribution to C[out_row1,out_col0]: +M2
        # Contribution to C[out_row1,out_col1]: +M2
        U[a_row1 * m + a_col0, col] = 1
        U[a_row1 * m + a_col1, col] = 1
        V[b_row0 * p + b_col0, col] = 1
        W[out_col0 * n + out_row1, col] = 1
        W[out_col1 * n + out_row1, col] = 1
        col += 1

        # M3 = A[a_row0,a_col0] * (B[b_row0,b_col1] - B[b_row1,b_col1])
        # Contribution to C[out_row0,out_col1]: +M3
        U[a_row0 * m + a_col0, col] = 1
        V[b_row0 * p + b_col1, col] = 1
        V[b_row1 * p + b_col1, col] = -1
        W[out_col1 * n + out_row0, col] = 1
        col += 1

        # M4 = A[a_row1,a_col1] * (B[b_row0,b_col0] + B[b_row1,b_col0])
        # Contribution to C[out_row0,out_col0]: +M4
        # Contribution to C[out_row1,out_col0]: +M4
        U[a_row1 * m + a_col1, col] = 1
        V[b_row0 * p + b_col0, col] = 1
        V[b_row1 * p + b_col0, col] = 1
        W[out_col0 * n + out_row0, col] = 1
        W[out_col0 * n + out_row1, col] = 1
        col += 1

        # M5 = (A[a_row0,a_col0] + A[a_row0,a_col1]) * B[b_row1,b_col1]
        # Contribution to C[out_row0,out_col0]: +M5
        # Contribution to C[out_row1,out_col1]: +M5
        U[a_row0 * m + a_col0, col] = 1
        U[a_row0 * m + a_col1, col] = 1
        V[b_row1 * p + b_col1, col] = 1
        W[out_col0 * n + out_row0, col] = 1
        W[out_col1 * n + out_row1, col] = 1
        col += 1

        # M6 = (A[a_row1,a_col0] - A[a_row0,a_col0]) * (B[b_row0,b_col0] + B[b_row0,b_col1])
        # Contribution to C[out_row1,out_col1]: -M6
        U[a_row1 * m + a_col0, col] = 1
        U[a_row0 * m + a_col0, col] = -1
        V[b_row0 * p + b_col0, col] = 1
        V[b_row0 * p + b_col1, col] = 1
        W[out_col1 * n + out_row1, col] = -1
        col += 1

        # M7 = (A[a_row0,a_col1] - A[a_row1,a_col1]) * (B[b_row1,b_col0] + B[b_row1,b_col1])
        # Contribution to C[out_row0,out_col0]: -M7
        # Contribution to C[out_row0,out_col1]: +M7
        U[a_row0 * m + a_col1, col] = 1
        U[a_row1 * m + a_col1, col] = -1
        V[b_row1 * p + b_col0, col] = 1
        V[b_row1 * p + b_col1, col] = 1
        W[out_col0 * n + out_row0, col] = -1
        W[out_col1 * n + out_row0, col] = 1
        col += 1

    # Now fill in the terms:
    
    # 1. Strassen for A[0:2,0:2] @ B[0:2,0:2] -> C[0:2,0:2]
    add_strassen_2x2x2(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
    # col should be 7 now

    # 2. A01@B10: A[0:2,2] @ B[2,0:2] -> C[0:2,0:2]
    for i in range(2):
        for k in range(2):
            add_term(i, 2, k)
    # col = 11

    # 3. Strassen for A[0:2,0:2] @ B[0:2,5:7] -> C[0:2,5:7]
    # B[0:2,5:7] is 2x2 with cols 5,6
    add_strassen_2x2x2(0, 1, 0, 1, 0, 1, 5, 6, 0, 1, 5, 6)
    # col = 18

    # 4. A01@B12: A[0:2,2] @ B[2,5:7] -> C[0:2,5:7]
    for i in range(2):
        for k in range(5, 7):
            add_term(i, 2, k)
    # col = 22

    # 5. A00@B01: A[0:2,0:2] @ B[0:2,2:5] -> C[0:2,2:5]
    # B[0:2,2:5] is 2x3 (cols 2,3,4)
    # Schoolbook: 2*2*3 = 12 terms
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                add_term(i, j, k)
    # col = 34

    # 6. A01@B11: A[0:2,2] @ B[2,2:5] -> C[0:2,2:5]
    for i in range(2):
        for k in range(2, 5):
            add_term(i, 2, k)
    # col = 40

    # 7. A10@B00: A[2,0:2] @ B[0:2,0:2] -> C[2,0:2]
    for j in range(2):
        for k in range(2):
            add_term(2, j, k)
    # col = 44

    # 8. A11@B10: A[2,2] @ B[2,0:2] -> C[2,0:2]
    for k in range(2):
        add_term(2, 2, k)
    # col = 46

    # 9. A10@B01: A[2,0:2] @ B[0:2,2:5] -> C[2,2:5]
    for j in range(2):
        for k in range(2, 5):
            add_term(2, j, k)
    # col = 52

    # 10. A11@B11: A[2,2] @ B[2,2:5] -> C[2,2:5]
    for k in range(2, 5):
        add_term(2, 2, k)
    # col = 55

    # 11. A10@B02: A[2,0:2] @ B[0:2,5:7] -> C[2,5:7]
    for j in range(2):
        for k in range(5, 7):
            add_term(2, j, k)
    # col = 59

    # 12. A11@B12: A[2,2] @ B[2,5:7] -> C[2,5:7]
    for k in range(5, 7):
        add_term(2, 2, k)
    # col = 61

    if col != rank:
        raise ValueError(f"Expected rank {rank}, got {col}")

    # Verify the decomposition
    def verify(U, V, W, rank):
        for trial in range(5):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_true = A @ B
            C_comp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    val = 0.0
                    for j in range(m):
                        for r in range(rank):
                            val += U[i*m + j, r] * A[i, j] * V[j*p + k, r] * B[j, k] * W[k*n + i, r]
                    C_comp[i, k] = val
            if not np.allclose(C_comp, C_true, atol=1e-9):
                return False
        return True

    if not verify(U, V, W, rank):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    # Try to reduce rank further by checking for linear dependencies
    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    # Try random linear combinations to see if we can reduce rank
    # This is heuristic and may not find anything
    for trial in range(50):
        if time.time() > deadline:
            break
        target = rank - 1
        if target < 1:
            break
        U_rand = np.random.randn(n * m, target)
        V_rand = np.random.randn(m * p, target)
        W_rand = np.random.randn(p * n, target)
        if verify(U_rand, V_rand, target):
            U_list = U_rand.tolist()
            V_list = V_rand.tolist()
            W_list = W_rand.tolist()
            rank = target
            break

    return {"U": U_list, "V": V_list, "W": W_list}
