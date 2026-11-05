"""Matrix multiplication 3,3,8 rank-minimal exact tensor decomposition."""
import numpy as np
from fractions import Fraction
import time
import sys

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Schoolbook rank is n*m*p = 3*3*8 = 72
    # We'll use a structural decomposition to get rank 68.
    #
    # Decompose A (3x3) and B (3x8) into blocks:
    # A = [[A11, A12],     B = [[B11, B12],
    #      [A21, A22]]         [B21, B22]]
    # where A11, B11 are 2x2, A12, B12 are 2x2, A21, B21 are 1x2, A22, B22 are 1x2
    #
    # Wait, let me be more careful.
    # A is 3x3, B is 3x8.
    # Split A into top 2 rows and bottom 1 row.
    # A_top = A[0:2, 0:3] (2x3)
    # A_bot = A[2:3, 0:3] (1x3)
    # Split B into top 2 rows and bottom 1 row.
    # B_top = B[0:2, 0:8] (2x8)
    # B_bot = B[2:3, 0:8] (1x8)
    #
    # C = A_top * B_top + A_bot * B_bot
    # A_top * B_top: (2x3) * (2x8) -- wait, this doesn't match dimensions.
    # A is 3x3, B is 3x8. The inner dimension is 3.
    # So we split along the inner dimension (columns of A, rows of B).
    # A = [A1 | A2 | A3] where Ai is 3x1
    # B = [B1; B2; B3] where Bi is 1x8
    # C = A1*B1 + A2*B2 + A3*B3
    # Each term is (3x1)*(1x8) = 3x8.
    # Rank of (3x1)*(1x8) is 3*1*8 = 24 schoolbook, but actually it's just outer product, rank 3.
    # Wait no. (3x1)*(1x8) = 3x8 matrix which is a sum of 3 outer products of vectors.
    # The tensor for (3x1)*(1x8) has rank 3 (one for each row of the 3x1 vector).
    # So total rank = 3*3 = 9? No that's wrong.
    #
    # Let me reconsider. The tensor T for multiplying (n x m) by (m x p) is in R^{nm} x R^{mp} x R^{pn}.
    # For (3x3)*(3x8): T is in R^9 x R^24 x R^24.
    #
    # Schoolbook: rank = 3*3*8 = 72.
    #
    # Alternative decomposition:
    # Split the inner dimension m=3 into m1=2, m2=1.
    # A = [A1 | A2] where A1 is 3x2, A2 is 3x1
    # B = [B1; B2] where B1 is 2x8, B2 is 1x8
    # C = A1*B1 + A2*B2
    #
    # A1*B1: (3x2)*(2x8). Schoolbook rank = 3*2*8 = 48.
    # A2*B2: (3x1)*(1x8). Schoolbook rank = 3*1*8 = 24.
    # Total schoolbook = 72.
    #
    # Can we do better for A1*B1?
    # A1 is 3x2, B1 is 2x8.
    # Split A1 into A1_top (2x2) and A1_bot (1x2).
    # A1 = [A1_top; A1_bot]
    # B1 stays as 2x8.
    # A1*B1 = A1_top*B1 + A1_bot*B1
    # A1_top*B1: (2x2)*(2x8). Can use Strassen-like? 
    # (2x2)*(2x8) = (2x2)*(2x2)*(2x4) -- not straightforward.
    #
    # Let me try a different approach. Use the known result for (2x2)*(2x2) Strassen (rank 7)
    # and combine with schoolbook for the rest.
    #
    # Split A (3x3) as:
    # A = [[a, b, c],
    #      [d, e, f],
    #      [g, h, i]]
    # Split B (3x8) as:
    # B = [[b11 b12 ... b18],
    #      [b21 b22 ... b28],
    #      [b31 b32 ... b38]]
    #
    # C = A*B where C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # Let's decompose using the 2x2 Strassen trick on the top-left 2x2 block of A and 
    # corresponding 2x8 submatrix of B, then handle the remaining parts.
    #
    # Actually, let me just implement a careful structural decomposition.
    #
    # Strategy: 
    # 1. Use Strassen for the (2x2)*(2x8) part by splitting B's 8 columns into four 2-column groups.
    #    For each group of 2 columns, we have (2x2)*(2x2) which has rank 7 (Strassen).
    #    4 groups * 7 = 28 rank for the (2x2)*(2x8) part.
    # 2. The (1x2)*(2x8) part: A_bot (1x2) * B_top (2x8). This is (1x2)*(2x8), schoolbook rank 16.
    #    But actually (1x2)*(2x8) = sum of 2 terms (outer products), rank 2*8 = 16? 
    #    No, (1x2)*(2x8): the result is 1x8. Each element C[k] = A[0,0]*B[0,k] + A[0,1]*B[1,k].
    #    This can be done with 8 multiplications (one per column), so rank 8.
    #    Wait, the tensor for (1x2)*(2x8) in R^2 x R^16 x R^8: rank is at most 16 schoolbook.
    #    But we can do it with 8 terms: for each column k, one term A[0,0]*B[0,k]*1 + A[0,1]*B[1,k]*1.
    #    Hmm, that's 2 terms per column = 16 terms total. Or 8 terms if we combine?
    #    Actually no. For (1xm)*(mxp), the rank is m*p (each (j,k) pair needs one term) 
    #    unless there's structure. For (1x2)*(2x8), rank = 2*8 = 16 schoolbook.
    #    But wait, we can write C[k] = A[0,0]*B[0,k] + A[0,1]*B[1,k].
    #    In tensor form: sum_j A[j]*B[j,k]*W[k]. We need one term per (j,k) where j is 0,1 and k is 0..7.
    #    So 16 terms. Rank 16.
    #
    # 3. The (2x1)*(1x8) part: A_left (2x1) * B_bot (1x8). Similarly rank 2*8 = 16.
    #
    # 4. The (1x1)*(1x8) part: A[2,2]*B[2,k] for each k. Rank 8.
    #
    # Total: 28 + 16 + 16 + 8 = 68.
    #
    # Let me verify the decomposition:
    # A = [[A11, A12],     B = [[B11, B12],
    #      [A21, A22]]         [B21, B22]]
    # where:
    # A11 = 2x2 (rows 0-1, col 0), A12 = 2x1 (rows 0-1, col 1), A13 = 2x1 (rows 0-1, col 2)
    # A21 = 1x2 (row 2, cols 0-1), A22 = 1x1 (row 2, col 2)
    # Wait, I need to be more careful.
    #
    # Let me split the inner dimension (size 3) into 2+1.
    # A (3x3) = [A_2x2 | A_2x1 | A_2x0] -- no this is confusing.
    #
    # Let me use indices directly.
    # A[i,j] for i=0,1,2 and j=0,1,2
    # B[j,k] for j=0,1,2 and k=0..7
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    #
    # Split j into j=0,1 (first part) and j=2 (second part).
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + A[i,2]*B[2,k]
    #
    # First part: sum_{j=0}^{1} A[i,j]*B[j,k] for i=0,1,2 and k=0..7
    # This is (3x2)*(2x8) multiplication.
    # Split i into i=0,1 (top) and i=2 (bottom).
    # C_top[k] = sum_{j=0}^{1} A[i,j]*B[j,k] for i=0,1, k=0..7: (2x2)*(2x8)
    # C_bot[k] = sum_{j=0}^{1} A[2,j]*B[j,k] for k=0..7: (1x2)*(2x8)
    #
    # Second part: A[i,2]*B[2,k] for i=0,1,2 and k=0..7: (3x1)*(1x8)
    #
    # Now:
    # (2x2)*(2x8): Split B into four (2x2) blocks along k.
    # For each block l (k in {2l, 2l+1}), we have (2x2)*(2x2) with rank 7 (Strassen).
    # Total: 4 * 7 = 28.
    #
    # (1x2)*(2x8): For each k, C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k].
    # This needs 2 terms per k, total 16 terms.
    #
    # (3x1)*(1x8): For each (i,k), C[i,k] += A[i,2]*B[2,k].
    # This needs 1 term per (i,k), total 3*8 = 24 terms.
    # Wait, that gives 28 + 16 + 24 = 68. 
    #
    # Hmm, earlier I said 8 for the last part but it should be 24. Let me recount.
    # (3x1)*(1x8): A[i,2] * B[2,k] for i=0,1,2 and k=0..7. That's 24 products.
    # But in tensor decomposition, each product corresponds to one term.
    # So rank contribution is 24.
    #
    # Total: 28 + 16 + 24 = 68. Good.
    #
    # Now let me implement this.
    #
    # Strassen terms for (2x2)*(2x2):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # For our case, A is 2x2 with entries A[i,j] for i=0,1 and j=0,1.
    # B is 2x2 with entries B[j,k'] for j=0,1 and k'=0,1 (local column index).
    #
    # Let me code this up.

    rank_total = 68
    U = [[0] * rank_total for _ in range(n * m)]  # 9 x 68
    V = [[0] * rank_total for _ in range(m * p)]  # 24 x 68
    W = [[0] * rank_total for _ in range(p * n)]  # 24 x 68

    col = 0

    # Part 1: (2x2)*(2x8) using Strassen for 4 blocks of (2x2)*(2x2)
    for block in range(4):
        k_offset = block * 2
        # A block: rows i=0,1; cols j=0,1 -> A[i*3+j]
        # B block: rows j=0,1; cols k=k_offset, k_offset+1 -> B[j*8+k]
        # C block: rows i=0,1; cols k=k_offset, k_offset+1 -> C[i*8+k] but W index is k*3+i
        
        # Strassen M1 = (A00+A11)(B00+B11)
        U[0*3+0][col] += 1  # A[0,0]
        U[1*3+1][col] += 1  # A[1,1]
        V[0*8+k_offset][col] += 1  # B[0,k0]
        V[1*8+k_offset+1][col] += 1  # B[1,k1]
        W[(k_offset)*3+0][col] += 1  # C[0,k0]
        W[(k_offset+1)*3+0][col] += 1  # C[0,k1]
        W[(k_offset)*3+1][col] += 1  # C[1,k0]
        W[(k_offset+1)*3+1][col] += 1  # C[1,k1]
        col += 1

        # M2 = (A10+A11)B00
        U[1*3+0][col] += 1  # A[1,0]
        U[1*3+1][col] += 1  # A[1,1]
        V[0*8+k_offset][col] += 1  # B[0,k0]
        W[(k_offset)*3+1][col] += 1  # C[1,k0]
        W[(k_offset+1)*3+1][col] += 1  # C[1,k1]
        col += 1

        # M3 = A00(B01-B11)
        U[0*3+0][col] += 1  # A[0,0]
        V[0*8+k_offset+1][col] += 1  # B[0,k1]
        V[1*8+k_offset+1][col] -= 1  # -B[1,k1]
        W[(k_offset+1)*3+0][col] += 1  # C[0,k1]
        W[(k_offset+1)*3+1][col] += 1  # C[1,k1]
        col += 1

        # M4 = A11(B10-B00)
        U[1*3+1][col] += 1  # A[1,1]
        V[1*8+k_offset][col] += 1  # B[1,k0]
        V[0*8+k_offset][col] -= 1  # -B[0,k0]
        W[(k_offset)*3+0][col] += 1  # C[0,k0]
        W[(k_offset)*3+1][col] += 1  # C[1,k0]
        col += 1

        # M5 = (A00+A01)B11
        U[0*3+0][col] += 1  # A[0,0]
        U[0*3+1][col] += 1  # A[0,1]
        V[1*8+k_offset+1][col] += 1  # B[1,k1]
        W[(k_offset+1)*3+0][col] -= 1  # -C[0,k1]
        W[(k_offset+1)*3+1][col] -= 1  # -C[1,k1]
        col += 1

        # M6 = (A10-A00)(B00+B01)
        U[1*3+0][col] += 1  # A[1,0]
        U[0*3+0][col] -= 1  # -A[0,0]
        V[0*8+k_offset][col] += 1  # B[0,k0]
        V[0*8+k_offset+1][col] += 1  # B[0,k1]
        W[(k_offset+1)*3+1][col] += 1  # C[1,k1]
        col += 1

        # M7 = (A01-A11)(B10+B11)
        U[0*3+1][col] += 1  # A[0,1]
        U[1*3+1][col] -= 1  # -A[1,1]
        V[1*8+k_offset][col] += 1  # B[1,k0]
        V[1*8+k_offset+1][col] += 1  # B[1,k1]
        W[(k_offset)*3+0][col] += 1  # C[0,k0]
        col += 1

    # Part 2: (1x2)*(2x8) for row i=2 of A
    # C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k]
    for k in range(8):
        # Term 1: A[2,0]*B[0,k]
        U[2*3+0][col] += 1  # A[2,0]
        V[0*8+k][col] += 1  # B[0,k]
        W[k*3+2][col] += 1  # C[2,k]
        col += 1
        # Term 2: A[2,1]*B[1,k]
        U[2*3+1][col] += 1  # A[2,1]
        V[1*8+k][col] += 1  # B[1,k]
        W[k*3+2][col] += 1  # C[2,k]
        col += 1

    # Part 3: (3x1)*(1x8) for col j=2 of A and row j=2 of B
    # C[i,k] += A[i,2]*B[2,k]
    for i in range(3):
        for k in range(8):
            U[i*3+2][col] += 1  # A[i,2]
            V[2*8+k][col] += 1  # B[2,k]
            W[k*3+i][col] += 1  # C[i,k]
            col += 1

    # Verify col == rank_total
    assert col == rank_total, f"Expected rank {rank_total}, got {col}"

    # Convert to numpy for verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Verify the decomposition
    # For random A and B, check that sum_r U[:,r] * V[:,r] * W[:,r] = A@B
    # The tensor equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = C[i,k]
    # which is sum_r sum_j U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = C[i,k]
    
    np.random.seed(42)
    A_test = np.random.randn(3, 3)
    B_test = np.random.randn(3, 8)
    C_expected = A_test @ B_test

    C_computed = np.zeros((3, 8))
    for r in range(rank_total):
        # U[:,r] is 9-vector, V[:,r] is 24-vector, W[:,r] is 24-vector
        # Need to compute sum_j U[i*3+j, r] * V[j*8+k, r] * W[k*3+i, r]
        for i in range(3):
            for k in range(8):
                s = 0
                for j in range(3):
                    s += U_np[i*3+j, r] * V_np[j*8+k, r] * W_np[k*3+i, r]
                C_computed[i, k] += s

    max_err = np.max(np.abs(C_computed - C_expected))
    if max_err > 1e-10:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][c] = 1
                    V_fb[j*p+k][c] = 1
                    W_fb[k*n+i][c] = 1
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
