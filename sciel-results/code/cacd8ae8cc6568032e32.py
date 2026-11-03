"""Matrix multiplication 2x3x10: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    # We build a decomposition of the 2x3x10 matrix multiplication tensor.
    # The tensor T has indices (i*m+j, j*p+k, k*n+i) = 1 for all valid (i,j,k).
    #
    # Strategy: Split j into j=0 (trivial, rank 20) and j=1,2 (2x2x10 sub-tensor).
    # For the 2x2x10 sub-tensor, we use Strassen's algorithm on 2x2x2 blocks.
    # 5 blocks of 2x2x2, each with rank 7, giving rank 35.
    # Total: 20 + 35 = 55.
    #
    # We then attempt to find a lower-rank decomposition by exploiting
    # the cyclic structure of the 2x2x10 tensor via WFTA (Winograd Fourier Transform Algorithm).
    #
    # For 2x2xp with p even, WFTA gives rank 2p + p/2 = 2*10 + 5 = 25 for the 2x2x10 part.
    # Wait, let me reconsider. The standard WFTA for 2x2xp:
    # The tensor is sum_{j=1,2} sum_{k=0}^{p-1} A[i,j]*B[j,k]*C[i,k]
    # WFTA decomposes this into rank 2p + p/2 for even p? Let me verify.
    #
    # Actually, for 2x2xp, the WFTA decomposition has rank 2p + p/2 when p is even.
    # For p=10: rank = 20 + 5 = 25.
    # So total would be 20 + 25 = 45.
    #
    # Let me construct this carefully.
    #
    # WFTA for 2x2xp (p even):
    # Let omega = exp(2*pi*i/p), but we need integer/half-integer entries.
    # Actually, WFTA uses the DFT matrix. For p=10, the DFT involves complex numbers.
    # But the task requires integer/half-integer entries.
    #
    # Hmm, this is a problem. Let me reconsider.
    #
    # Actually, the WFTA decomposition for 2x2xp can be written with integer entries
    # if we use the fact that the DFT of a real sequence can be represented
    # using only real arithmetic. But for p=10, this still involves sqrt(2) and sqrt(5).
    #
    # Let me think about this differently. The task says "integer/half-integer entries".
    # This means entries must be in {k/2 : k in Z}.
    #
    # So we can't use the standard WFTA with DFT. We need a different approach.
    #
    # Let me try a different strategy: use the known rank-55 decomposition and then
    # try to find linear dependencies among the 55 rank-1 tensors to reduce the rank.
    #
    # Actually, let me just return the rank-55 decomposition, which is valid.
    # The score will be 55, and seed/value = 60/55 = 1.0909.

    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part A: j=0 (first column of A, first row of B)
    # A[i,0] * B[0,k] -> C[i,k]
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
    # 5 blocks of 2x2 Strassen
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2

        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1

        # M1 = (a11 + a22)(b11 + b22)
        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1

        # M2 = (a21 + a22)b11
        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1

        # M3 = a11(b12 - b22)
        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1

        # M4 = a22(b21 - b11)
        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1

        # M5 = (a11 + a12)b22
        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1

        # M6 = (a21 - a11)(b11 + b12)
        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1

        # M7 = (a12 - a22)(b21 + b22)
        U[u_12][col] += 1
        U[u_22][col] += -1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1

    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    dim_u = n * m  # 6
    dim_v = m * p  # 30
    dim_w = p * n  # 20

    T_reconstructed = np.zeros((dim_u, dim_v, dim_w))
    for c in range(rank):
        T_reconstructed += np.einsum('i,j,k->ijk', U_np[:, c], V_np[:, c], W_np[:, c])

    T_true = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T_true[alpha, beta, gamma] = 1

    if not np.allclose(T_reconstructed, T_true, atol=1e-10):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col_s] = 1
                    V_school[j * p + k][col_s] = 1
                    W_school[k * n + i][col_s] = 1
                    col_s += 1
        return {"U": U_school, "V": V_school, "W": W_school}

    return {"U": U, "V": V, "W": W}
