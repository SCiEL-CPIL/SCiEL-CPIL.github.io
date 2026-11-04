import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Rank 30 construction using a mixed decomposition:
    # For n=2, m=4, p=6, we decompose as:
    # - 2 Strassen 2x2x2 blocks (for j=0,1) -> rank 14
    # - 2x2x3 convolution for each j block -> rank 4 each, total 8
    # But we need to be careful with the tensor structure.
    #
    # Actually, let's try a different approach:
    # Use the fact that the matmul tensor for (2,4,6) can be decomposed
    # using 30 terms by combining Strassen and direct methods optimally.
    #
    # Let me construct a rank-30 decomposition:
    # Split m=4 into two groups of 2: m1=[0,1], m2=[2,3]
    # Split p=6 into three groups of 2: p1=[0,1], p2=[2,3], p3=[4,5]
    #
    # For each (j_group, k_group) pair, we have a 2x2x2 subproblem in (i, j_local, k_local)
    # There are 2 j_groups and 3 k_groups = 6 subproblems
    # Each 2x2x2 subproblem has rank 7 (Strassen) or rank 8 (schoolbook)
    # Total: 6 * 7 = 42 (current)
    #
    # Alternative: For n=2, we can exploit the small n dimension.
    # Let's try to find a rank-30 construction.
    #
    # Actually, let me use a known result: the rank of (2,4,6) matmul is at most 30.
    # Here's a construction using Winograd's method:
    #
    # A is 2x4, B is 4x6
    # C = A * B is 2x6
    #
    # We can write:
    # C[0, :] = A[0, 0:2] * B[0:2, :] + A[0, 2:4] * B[2:4, :]
    # C[1, :] = A[1, 0:2] * B[0:2, :] + A[1, 2:4] * B[2:4, :]
    #
    # Each 2x4 * 4x6 multiplication can be split into:
    # 2 * (2x2 * 2x6) + 2 * (2x2 * 2x6)
    #
    # For a 2x2 * 2x6 multiplication:
    # This is equivalent to computing [a0, a1; b0, b1] * [B1 B2 B3]
    # = [a0*B1 + a1*B2, a0*B2 + a1*B3, a0*B3 + a1*B4] (each B is 2x2)
    #
    # Actually, let me just construct a valid rank-30 decomposition directly.
    #
    # Known construction for rank 30:
    # Use 5 Strassen-like blocks of rank 6 each? No, that doesn't work.
    #
    # Let me try a systematic approach. The matmul tensor T has components T[i,j,k]
    # where i in [0,2), j in [0,4), k in [0,6).
    # T[i,j,k] = delta_{condition} where the condition is that the (i,j) entry of A
    # and (j,k) entry of B contribute to (i,k) entry of C.
    #
    # Actually, the matmul tensor has T[i, (j, k), (i', k')] = 1 if i=i' and j=k', else 0.
    # Wait, no. The tensor is:
    # T[a, b, c] = 1 if a = i*m + j, b = j'*p + k, c = k'*n + i' and j = j', i = i'
    # So T[i*m+j, j'*p+k, k'*n+i'] = 1 if j=j' and i=i', else 0.
    #
    # Let me just use the rank-42 construction but verify it's correct, and accept that
    # this is the best I can do.
    #
    # Actually, let me try rank 36 by using a mixed approach:
    # - For j=0,1 (first half): use Strassen 2x2x2 for each k-group, 3 * 7 = 21
    # - For j=2,3 (second half): use a different decomposition
    #
    # This is getting complicated. Let me just return the rank-42 construction.

    rank = 42
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)

    def add_strassen_block(i_start, j_start, k_start, offset):
        a00 = (i_start + 0) * m + (j_start + 0)
        a01 = (i_start + 0) * m + (j_start + 1)
        a10 = (i_start + 1) * m + (j_start + 0)
        a11 = (i_start + 1) * m + (j_start + 1)

        b00 = (j_start + 0) * p + (k_start + 0)
        b01 = (j_start + 0) * p + (k_start + 1)
        b10 = (j_start + 1) * p + (k_start + 0)
        b11 = (j_start + 1) * p + (k_start + 1)

        w00 = (k_start + 0) * n + (i_start + 0)
        w01 = (k_start + 1) * n + (i_start + 0)
        w10 = (k_start + 0) * n + (i_start + 1)
        w11 = (k_start + 1) * n + (i_start + 1)

        # M1: (A00+A11)(B00+B11)
        U[a00, offset] += 1.0
        U[a11, offset] += 1.0
        V[b00, offset] += 1.0
        V[b11, offset] += 1.0
        W[w00, offset] += 1.0
        W[w11, offset] += 1.0

        # M2: (A10+A11)B00
        U[a10, offset + 1] += 1.0
        U[a11, offset + 1] += 1.0
        V[b00, offset + 1] += 1.0
        W[w10, offset + 1] += 1.0
        W[w11, offset + 1] -= 1.0

        # M3: A00(B01-B11)
        U[a00, offset + 2] += 1.0
        V[b01, offset + 2] += 1.0
        V[b11, offset + 2] -= 1.0
        W[w01, offset + 2] += 1.0
        W[w11, offset + 2] += 1.0

        # M4: A11(B10-B00)
        U[a11, offset + 3] += 1.0
        V[b10, offset + 3] += 1.0
        V[b00, offset + 3] -= 1.0
        W[w00, offset + 3] += 1.0
        W[w10, offset + 3] += 1.0

        # M5: (A00+A01)B11
        U[a00, offset + 4] += 1.0
        U[a01, offset + 4] += 1.0
        V[b11, offset + 4] += 1.0
        W[w00, offset + 4] -= 1.0
        W[w01, offset + 4] += 1.0

        # M6: (A10-A00)(B00+B01)
        U[a10, offset + 5] += 1.0
        U[a00, offset + 5] -= 1.0
        V[b00, offset + 5] += 1.0
        V[b01, offset + 5] += 1.0
        W[w11, offset + 5] += 1.0

        # M7: (A01-A11)(B10+B11)
        U[a01, offset + 6] += 1.0
        U[a11, offset + 6] -= 1.0
        V[b10, offset + 6] += 1.0
        V[b11, offset + 6] += 1.0
        W[w00, offset + 6] += 1.0

    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_block(i_start, j_start, k_start, offset)
                offset += 7

    # Quick verification
    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    if not verify(U, V, W):
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * (m * p) + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
        return {"U": U, "V": V, "W": W}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
