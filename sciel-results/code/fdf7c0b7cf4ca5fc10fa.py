"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Output index convention: W[k*n + i, c] where output C[i,k] = sum_j A[i,j]*B[j,k]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]

    # We will try to find a rank-minimal decomposition.
    # Known results:
    # - 2x2x2 (Strassen): rank 7
    # - 2x2x3: rank 9 (this is a known result, see e.g. Bini et al.)
    # - 2x3x3: rank 12
    # - 2x4x5: we can partition

    # Strategy: partition the 4-dim inner product into 2+2, and 5-dim output into 2+3.
    # This gives us:
    # - 2x2x2 subproblem (A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]): rank 7
    # - 2x2x3 subproblem (A[0:2,0:2] * B[0:2,2:5] -> C[0:2,2:5]): rank 9
    # - 2x2x2 subproblem (A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]): rank 7
    # - 2x2x3 subproblem (A[0:2,2:4] * B[2:4,2:5] -> C[0:2,2:5]): rank 9
    # Total rank: 7 + 9 + 7 + 9 = 32

    # But wait, we need to be more careful. The matrix is 2x4 times 4x5.
    # Let's partition A into A1 (2x2) and A2 (2x2) along columns.
    # Let's partition B into B1 (4x2) and B2 (4x3) along columns.
    # Then C = A*B = A1*B1_top + A2*B2_top for first 2 cols, etc.

    # Actually, let me think more carefully.
    # A is 2x4: rows i=0,1; cols j=0,1,2,3
    # B is 4x5: rows j=0,1,2,3; cols k=0,1,2,3,4
    # C is 2x5: rows i=0,1; cols k=0,1,2,3,4
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]

    # Partition j into {0,1} and {2,3}, and k into {0,1} and {2,3,4}.
    # Then:
    # C[i, 0:2] = A[i,0:2] * B[0:2, 0:2] + A[i,2:4] * B[2:4, 0:2]
    # C[i, 2:5] = A[i,0:2] * B[0:2, 2:5] + A[i,2:4] * B[2:4, 2:5]

    # So we have four sub-matrix multiplications:
    # 1. A[0:2,0:2] (2x2) * B[0:2,0:2] (2x2) -> 2x2: rank 7 (Strassen)
    # 2. A[0:2,0:2] (2x2) * B[0:2,2:5] (2x3) -> 2x3: rank 9
    # 3. A[0:2,2:4] (2x2) * B[2:4,0:2] (2x2) -> 2x2: rank 7 (Strassen)
    # 4. A[0:2,2:4] (2x2) * B[2:4,2:5] (2x3) -> 2x3: rank 9

    # Total rank = 7 + 9 + 7 + 9 = 32

    # Now I need the actual Strassen and 2x2x3 decompositions.

    # Strassen for 2x2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For 2x2x3, I need a rank-9 decomposition.
    # Let me think about this. A (2x2) * B (2x3) -> C (2x3).
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i=0,1; k=0,1,2

    # One approach: use Strassen-like ideas.
    # Actually, let me just use the schoolbook for 2x2x3 which has rank 6,
    # but we can do better. The known rank for 2x2x3 is 9? Let me verify.
    # Actually, I recall that the rank of 2x2x3 is 9. But let me try to construct it.

    # Hmm, actually I'm not 100% sure about the 2x2x3 rank. Let me just use
    # schoolbook for the 2x2x3 parts to be safe, which gives rank 6 each.
    # Then total rank = 7 + 6 + 7 + 6 = 26.

    # Wait, that's already better than 40! Let me verify the schoolbook for 2x2x3.
    # A (2x2) * B (2x3): 4*3 = 12 products, but we can share some.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # For fixed i, we compute A[i,0]*B[0,k] and A[i,1]*B[1,k] for k=0,1,2.
    # That's 2*3 = 6 multiplications for each i, so 12 total. No sharing possible
    # in the straightforward way.

    # But actually, we can use the fact that for different k, the B values are different.
    # So schoolbook for 2x2x3 is rank 12? No wait.
    # U has 4 entries (2x2), V has 6 entries (2x3), W has 6 entries (2x3).
    # Schoolbook: for each (i,j,k), one term. That's 2*2*3 = 12 terms.
    # So rank 12 for schoolbook 2x2x3.

    # Hmm, so with Strassen for 2x2x2 (rank 7) and schoolbook for 2x2x3 (rank 12):
    # Total = 7 + 12 + 7 + 12 = 38. Still better than 40!

    # Let me try to find a better 2x2x3 decomposition.
    # Actually, I found references that say rank(2x2x3) = 9. Let me try to construct it.

    # Alternative: maybe I should try a different partition.
    # What if I partition the 4-dim into 2+2 and don't partition the 5-dim?
    # Then I have:
    # A[0:2,0:2] * B[0:2,0:5] -> 2x5: this is 2x2x5
    # A[0:2,2:4] * B[2:4,0:5] -> 2x5: this is 2x2x5
    # Total rank = 2 * rank(2x2x5)

    # What's rank(2x2x5)? Schoolbook is 2*2*5=20.
    # That's worse.

    # Let me stick with the 2+2 / 2+3 partition.
    # Total rank with Strassen + schoolbook = 38.

    # Let me also consider: can I do better on the 2x2x3 part?
    # I'll try to find a rank-9 or rank-10 decomposition for 2x2x3.

    # For now, let me implement the 38-rank solution and verify it works.

    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition.
        U has shape (4, 7) indexed by [i*2+j]
        V has shape (4, 7) indexed by [j*2+k]
        W has shape (4, 7) indexed by [k*2+i]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]

        # M1 = (A00+A11)*(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[2][0] = 1  # W[k*2+i]: k=0,i=0 -> 0; k=1,i=0 -> 2; wait
        # W index: k*n + i where n=2. So W[k*2+i].
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # W[k*2+i, c] contributes to C[i,k]
        # So W[0*2+0]=W[0] for C00, W[1*2+0]=W[2] for C00? No.
        # W[k*n+i] where n=2 (number of rows of A / cols of C).
        # C[i,k] gets contribution from W[k*2+i].
        # C00: i=0,k=0 -> W[0]
        # C01: i=0,k=1 -> W[2]
        # C10: i=1,k=0 -> W[1]
        # C11: i=1,k=1 -> W[3]

        # M1 contributes to C00 and C11
        W[0][0] = 1   # C00
        W[3][0] = 1   # C11

        # M2 = (A10+A11)*B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        # M2 contributes to C10 and -C11
        W[1][1] = 1   # C10
        W[3][1] = -1  # C11

        # M3 = A00*(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        # M3 contributes to C01 and C11
        W[2][2] = 1   # C01
        W[3][2] = 1   # C11

        # M4 = A11*(B10-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        # M4 contributes to C00 and C10
        W[0][3] = 1   # C00
        W[1][3] = 1   # C10

        # M5 = (A00+A01)*B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        # M5 contributes to -C00 and C01
        W[0][4] = -1  # C00
        W[2][4] = 1   # C01

        # M6 = (A10-A00)*(B00+B10)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[2][5] = 1
        # M6 contributes to C11
        W[3][5] = 1   # C11

        # M7 = (A01-A11)*(B01+B11)
        U[1][6] = 1; U[3][6] = -1
        V[1][6] = 1; V[3][6] = 1
        # M7 contributes to C00
        W[0][6] = 1   # C00

        return U, V, W

    def schoolbook_2x2x3():
        """Return U, V, W for 2x2x3 schoolbook decomposition. Rank 12.
        U has shape (4, 12) indexed by [i*2+j]
        V has shape (6, 12) indexed by [j*3+k]
        W has shape (6, 12) indexed by [k*2+i]
        """
        U = [[0]*12 for _ in range(4)]
        V = [[0]*12 for _ in range(6)]
        W = [[0]*12 for _ in range(6)]
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[i*2+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return U, V, W

    # Build the full 2x4x5 decomposition
    # Partition: j = {0,1} (A1) and {2,3} (A2)
    #             k = {0,1} (B1) and {2,3,4} (B2)

    # Sub-problem 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] (2x2x2, Strassen)
    # Sub-problem 2: A[0:2,0:2] * B[0:2,2:5] -> C[0:2,2:5] (2x2x3, schoolbook)
    # Sub-problem 3: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2] (2x2x2, Strassen)
    # Sub-problem 4: A[0:2,2:4] * B[2:4,2:5] -> C[0:2,2:5] (2x2x3, schoolbook)

    total_rank = 7 + 12 + 7 + 12  # = 38

    U_full = [[0]*total_rank for _ in range(8)]  # 2*4 = 8 rows
    V_full = [[0]*total_rank for _ in range(20)] # 4*5 = 20 rows
    W_full = [[0]*total_rank for _ in range(10)] # 5*2 = 10 rows

    col_offset = 0

    # Sub-problem 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    U1, V1, W1 = strassen_2x2x2()
    for r in range(4):
        for c in range(7):
            U_full[r][col_offset + c] = U1[r][c]
    for r in range(4):
        for c in range(7):
            V_full[r][col_offset + c] = V1[r][c]
    for r in range(4):
        for c in range(7):
            W_full[r][col_offset + c] = W1[r][c]
    col_offset += 7

    # Sub-problem 2: A[0:2,0:2] * B[0:2,2:5] -> C[0:2,2:5]
    U2, V2, W2 = schoolbook_2x2x3()
    for r in range(4):
        for c in range(12):
            U_full[r][col_offset + c] = U2[r][c]
    # V2 is indexed by [j*3+k] where j in {0,1}, k in {0,1,2}
    # But in V_full, the index is [j*5+k] where j in {0,1,2,3}, k in {0,1,2,3,4}
    # For sub-problem 2, j in {0,1}, k in {2,3,4} (which is k_local in {0,1,2})
    for j_local in range(2):
        for k_local in range(3):
            j_full = j_local  # 0,1
            k_full = 2 + k_local  # 2,3,4
            for c in range(12):
                V_full[j_full*5 + k_full][col_offset + c] = V2[j_local*3 + k_local][c]
    # W2 is indexed by [k*2+i] where k in {0,1,2}, i in {0,1}
    # In W_full, index is [k*2+i] where k in {0,1,2,3,4}, i in {0,1}
    for k_local in range(3):
        for i in range(2):
            k_full = 2 + k_local
            for c in range(12):
                W_full[k_full*2 + i][col_offset + c] = W2[k_local*2 + i][c]
    col_offset += 12

    # Sub-problem 3: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    U3, V3, W3 = strassen_2x2x2()
    # U3 is indexed by [i*2+j_local] where j_local in {0,1}
    # In U_full, index is [i*4+j] where j in {0,1,2,3}
    for i in range(2):
        for j_local in range(2):
            j_full = 2 + j_local
            for c in range(7):
                U_full[i*4 + j_full][col_offset + c] = U3[i*2 + j_local][c]
    # V3 is indexed by [j_local*2+k] where j_local in {0,1}, k in {0,1}
    # In V_full, index is [j*5+k] where j in {2,3}, k in {0,1}
    for j_local in range(2):
        for k in range(2):
            j_full = 2 + j_local
            for c in range(7):
                V_full[j_full*5 + k][col_offset + c] = V3[j_local*2 + k][c]
    for r in range(4):
        for c in range(7):
            W_full[r][col_offset + c] = W3[r][c]
    col_offset += 7

    # Sub-problem 4: A[0:2,2:4] * B[2:4,2:5] -> C[0:2,2:5]
    U4, V4, W4 = schoolbook_2x2x3()
    for i in range(2):
        for j_local in range(2):
            j_full = 2 + j_local
            for c in range(12):
                U_full[i*4 + j_full][col_offset + c] = U4[i*2 + j_local][c]
    for j_local in range(2):
        for k_local in range(3):
            j_full = 2 + j_local
            k_full = 2 + k_local
            for c in range(12):
                V_full[j_full*5 + k_full][col_offset + c] = V4[j_local*3 + k_local][c]
    for k_local in range(3):
        for i in range(2):
            k_full = 2 + k_local
            for c in range(12):
                W_full[k_full*2 + i][col_offset + c] = W4[k_local*2 + i][c]
    col_offset += 12

    # Verify correctness with a simple test
    import numpy as np
    if time.time() < deadline - 10:
        U_np = np.array(U_full, dtype=float)
        V_np = np.array(V_full, dtype=float)
        W_np = np.array(W_full, dtype=float)

        # Test with random matrices
        np.random.seed(42)
        A_test = np.random.randint(-5, 6, size=(2, 4)).astype(float)
        B_test = np.random.randint(-5, 6, size=(4, 5)).astype(float)
        C_expected = A_test @ B_test

        # Compute C from decomposition
        # C[i,k] = sum_c U[i*4+j, c] * V[j*5+k, c] * W[k*2+i, c]
        # This is a tensor contraction. Let's verify by checking:
        # For each (i,k), C[i,k] = sum_{j} A[i,j] * B[j,k]
        # From decomposition: C[i,k] = sum_c (sum_j U[i*4+j,c] * A[i,j]) * (sum_k' V[j*5+k',c] * B[j,k']) ...
        # Actually, the decomposition means:
        # C[i,k] = sum_c U[i*4+?, c] ... no.
        # The standard form is: C[i,k] = sum_c (sum_j U[i*4+j,c]*A[i,j]) * (sum_k V[j*5+k,c]*B[j,k])
        # Wait, that's not right either. Let me think.
        # The tensor T[i,j,k] = delta (i.e., C[i,k] = sum_j A[i,j]B[j,k])
        # Decomposition: T[i,j,k] = sum_c U[i*4+j,c] * V[j*5+k,c] * W[k*2+i,c]
        # So for the product: sum_j A[i,j] B[j,k] = sum_c W[k*2+i,c] * (sum_j U[i*4+j,c] A[i,j]) * (sum_j V[j*5+k,c] B[j,k])
        # Wait, that's not standard. Let me re-read the task.
        # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
        # So the decomposition is: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # But this doesn't make sense as written because j is summed over.
        # I think the correct interpretation is:
        # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
        # No, that's not right. Let me think again.
        # Actually, I think the convention is that the tensor for matrix multiplication is:
        # T[i,j,k] where C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
        # And T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # But T[i,j,k] should be delta (i.e., T[i,j,k] = 1 for all i,j,k since C[i,k] = sum_j A[i,j]B[j,k])
        # Wait no, T[i,j,k] is not delta. The matrix multiplication tensor is defined such that
        # C[i,k] = sum_{j} A[i,j] B[j,k], so the "tensor" is just the identity in j.
        # I think the decomposition is: for each c, define:
        # A_c[i,j] = U[i*m+j, c]
        # B_c[j,k] = V[j*p+k, c]
        # C_c[i,k] = W[k*n+i, c]
        # Then C = sum_c C_c * (A * A_c) * (B_c * B) ... no.
        # I think the correct interpretation is:
        # C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k]) * W[k*n+i,c]
        # Hmm, that would be a sum of products of two bilinear forms, times a scalar.
        # Let me just verify numerically.

        C_computed = np.zeros((2, 5))
        for i in range(2):
            for k in range(5):
                total = 0.0
                for c in range(total_rank):
                    # Compute sum_j U[i*4+j, c] * A[i, j]
                    sum_U = 0.0
                    for j in range(4):
                        sum_U += U_np[i*4+j, c] * A_test[i, j]
                    # Compute sum_j V[j*5+k, c] * B[j, k]
                    sum_V = 0.0
                    for j in range(4):
                        sum_V += V_np[j*5+k, c] * B_test[j, k]
                    total += W_np[k*2+i, c] * sum_U * sum_V
                C_computed[i, k] = total

        if not np.allclose(C_computed, C_expected, atol=1e-10):
            print(f"Verification failed!")
            print(f"Expected: {C_expected}")
            print(f"Computed: {C_computed}")
            # Fall back to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}

    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
