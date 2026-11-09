"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin

    def _build_schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.int32)
        V = np.zeros((m * p, rank), dtype=np.int32)
        W = np.zeros((p * n, rank), dtype=np.int32)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
        """Verify that U, V, W form a valid tensor decomposition using vectorized ops."""
        rank = U.shape[1]
        rng = np.random.RandomState(42)
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B

        # Vectorized verification
        # For each rank-r term: alpha_r = U_r^T A_flat, beta_r = V_r^T B_flat, then C += W_r * outer(alpha_r, beta_r)
        # U_flat: (n*m, rank), A_flat: (n*m,)
        # alpha: (rank,) = U.T @ A_flat
        A_flat = A.flatten()  # row-major: [A[0,0], A[0,1], ..., A[0,m-1], A[1,0], ...]
        B_flat = B.flatten()  # row-major: [B[0,0], B[0,1], ..., B[0,p-1], B[1,0], ...]
        
        alpha = U.T @ A_flat  # (rank,)
        beta = V.T @ B_flat   # (rank,)
        
        # C_computed[i,k] = sum_r W[k*n+i, r] * alpha[r] * beta[r]
        # W is (p*n, rank), reshape to (p, n, rank)
        W_3d = W.reshape(p, n, rank)
        # C_computed[i,k] = sum_r W_3d[k, i, r] * alpha[r] * beta[r]
        # = (W_3d * (alpha * beta)[None, None, :]).sum(axis=2)
        weighted = alpha * beta  # (rank,)
        C_computed = np.einsum('knr,r->nk', W_3d, weighted)
        
        return np.allclose(C_computed, C_expected, atol=tol)

    def _build_strassen_2x2x7():
        """Build rank-7 Strassen decomposition for 2x2 @ 2x7.
        U: (4, 7), V: (14, 7), W: (14, 7)
        Indexing: U[i*2+j, r], V[j*7+k, r], W[k*2+i, r]
        """
        rank = 7
        U = np.zeros((4, rank), dtype=np.int32)
        V = np.zeros((14, rank), dtype=np.int32)
        W = np.zeros((14, rank), dtype=np.int32)

        # M1 = (a00 + a11)(b00 + b11)
        # U: a00 and a11 -> indices 0, 3
        U[0, 0] = 1; U[3, 0] = 1
        # V: b00 and b11 -> b00 is row 0 (indices 0-6), b11 is row 1 (indices 7-13)
        # b00: V[0*7+0, 0] = 1, b11: V[1*7+0, 0] = 1
        V[0, 0] = 1; V[7, 0] = 1
        # W: contributes to c00 and c11
        # c00: W[0*2+0, 0] = 1, c11: W[1*2+1, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1

        # M2 = (a10 + a11)*b00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1  # b00
        W[2, 1] = 1  # c10
        W[3, 1] = -1 # c11

        # M3 = a00*(b01 - b11)
        U[0, 2] = 1
        V[1, 2] = 1   # b01
        V[8, 2] = -1  # b11
        W[1, 2] = 1   # c01
        W[3, 2] = 1   # c11

        # M4 = a11*(b10 - b00)
        U[3, 3] = 1
        V[7, 3] = 1   # b10
        V[0, 3] = -1  # b00
        W[2, 3] = 1   # c10
        W[0, 3] = 1   # c00

        # M5 = (a00 + a01)*b11
        U[0, 4] = 1; U[1, 4] = 1
        V[7, 4] = 1   # b11
        W[1, 4] = 1   # c01
        W[0, 4] = -1  # c00

        # M6 = (a10 - a00)*(b00 + b01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1   # b00
        V[1, 5] = 1   # b01
        W[3, 5] = 1   # c11

        # M7 = (a01 - a11)*(b10 + b11)
        U[1, 6] = 1; U[3, 6] = -1
        V[7, 6] = 1   # b10
        V[8, 6] = 1   # b11
        W[0, 6] = 1   # c00

        return U, V, W

    def _build_2x4x7_decomposition():
        """Build rank-14 decomposition for 2x4 @ 4x7 using two 2x2x7 Strassen blocks.
        A = [A1 | A2] where A1, A2 are 2x2
        B = [B1; B2] where B1, B2 are 2x7
        C = A1*B1 + A2*B2
        """
        n, m, p = 2, 4, 7
        total_rank = 14
        
        U = np.zeros((n * m, total_rank), dtype=np.int32)  # (8, 14)
        V = np.zeros((m * p, total_rank), dtype=np.int32)  # (28, 14)
        W = np.zeros((p * n, total_rank), dtype=np.int32)  # (14, 14)
        
        # Block 1: A1 (cols 0-1 of A) * B1 (rows 0-1 of B)
        U1, V1, W1 = _build_strassen_2x2x7()
        # A1 is at A[0:2, 0:2], so U indices: i*4+j for i in 0-1, j in 0-1
        for i in range(2):
            for j in range(2):
                U[i*4 + j, 0:7] = U1[i*2 + j, :]
        # B1 is at B[0:2, :], so V indices: j*7+k for j in 0-1, k in 0-6
        for j in range(2):
            for k in range(7):
                V[j*7 + k, 0:7] = V1[j*7 + k, :]
        W[:, 0:7] = W1
        
        # Block 2: A2 (cols 2-3 of A) * B2 (rows 2-3 of B)
        U2, V2, W2 = _build_strassen_2x2x7()
        # A2 is at A[0:2, 2:4], so U indices: i*4+(2+j) for i in 0-1, j in 0-1
        for i in range(2):
            for j in range(2):
                U[i*4 + 2 + j, 7:14] = U2[i*2 + j, :]
        # B2 is at B[2:4, :], so V indices: (2+j)*7+k for j in 0-1, k in 0-6
        for j in range(2):
            for k in range(7):
                V[(2+j)*7 + k, 7:14] = V2[j*7 + k, :]
        W[:, 7:14] = W2
        
        return U, V, W

    # Build the rank-14 decomposition
    U, V, W = _build_2x4x7_decomposition()
    rank = U.shape[1]

    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # If verification fails, fall back to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
