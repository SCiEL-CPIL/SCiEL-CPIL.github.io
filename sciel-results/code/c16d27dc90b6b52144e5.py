import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 125.0

    # Build the 189-rank decomposition: Strassen on 2x2 blocks within 3x3 grid
    # Total rank = 3^3 * 7 = 189
    def build_block_strassen():
        """3x3 grid of 2x2 Strassen blocks, rank = 27*7 = 189"""
        rank = 189
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)

        idx = 0
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    for s in range(7):
                        a00 = (2*i) * 6 + 2*k
                        a01 = (2*i) * 6 + 2*k + 1
                        a10 = (2*i+1) * 6 + 2*k
                        a11 = (2*i+1) * 6 + 2*k + 1

                        b00 = (2*k) * 6 + 2*j
                        b01 = (2*k) * 6 + 2*j + 1
                        b10 = (2*k+1) * 6 + 2*j
                        b11 = (2*k+1) * 6 + 2*j + 1

                        c00 = (2*j) * 6 + 2*i
                        c01 = (2*j+1) * 6 + 2*i
                        c10 = (2*j) * 6 + (2*i+1)
                        c11 = (2*j+1) * 6 + (2*i+1)

                        if s == 0:
                            U[a00, idx] = 1; U[a11, idx] = 1
                            V[b00, idx] = 1; V[b11, idx] = 1
                            W[c00, idx] = 1; W[c11, idx] = 1
                        elif s == 1:
                            U[a10, idx] = 1; U[a11, idx] = 1
                            V[b00, idx] = 1
                            W[c10, idx] = 1; W[c11, idx] = -1
                        elif s == 2:
                            U[a00, idx] = 1
                            V[b01, idx] = 1; V[b11, idx] = -1
                            W[c01, idx] = 1; W[c11, idx] = 1
                        elif s == 3:
                            U[a11, idx] = 1
                            V[b10, idx] = 1; V[b00, idx] = -1
                            W[c00, idx] = 1; W[c10, idx] = 1
                        elif s == 4:
                            U[a00, idx] = 1; U[a01, idx] = 1
                            V[b11, idx] = 1
                            W[c00, idx] = -1; W[c01, idx] = 1
                        elif s == 5:
                            U[a10, idx] = 1; U[a00, idx] = -1
                            V[b00, idx] = 1; V[b01, idx] = 1
                            W[c11, idx] = 1
                        elif s == 6:
                            U[a01, idx] = 1; U[a11, idx] = -1
                            V[b10, idx] = 1; V[b11, idx] = 1
                            W[c00, idx] = 1
                        idx += 1
        return U, V, W, rank

    U, V, W, rank = build_block_strassen()

    # Fast verification using matrix operations
    if time.time() < deadline - 3:
        U_f = U.astype(float)
        V_f = V.astype(float)
        W_f = W.astype(float)

        np.random.seed(42)
        all_valid = True
        for trial in range(3):
            A_test = np.random.randint(-10, 10, (n, m)).astype(float)
            B_test = np.random.randint(-10, 10, (m, p)).astype(float)
            C_expected = A_test @ B_test

            # Reshape and use matrix multiplication for fast check
            # C[i,j] = sum_t W[j*n+i, t] * (sum_k U[i*m+k,t] * V[k*p+j,t])
            # = sum_t W[j*n+i,t] * (A_row_i @ U_col_t) * (V_col_t @ B_col_j)
            # More efficiently: compute M = A @ U (n x rank), N = V^T @ B (rank x p)
            # Then C[i,j] = sum_t M[i,t] * W[j*n+i,t] * N[t,j]
            M = A_test @ U_f  # (n, rank)
            N = V_f.T @ B_test  # (rank, p)

            C_computed = np.zeros((n, p))
            for i in range(n):
                for j in range(p):
                    C_computed[i, j] = np.dot(M[i, :], W_f[j*n+i, :] * N[:, j])

            if not np.allclose(C_computed, C_expected, atol=1e-8):
                all_valid = False
                break

        if not all_valid:
            rank_school = n * m * p
            U_s = np.zeros((n * m, rank_school), dtype=int)
            V_s = np.zeros((m * p, rank_school), dtype=int)
            W_s = np.zeros((p * n, rank_school), dtype=int)
            idx_s = 0
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        U_s[i * m + k, idx_s] = 1
                        V_s[k * p + j, idx_s] = 1
                        W_s[j * n + i, idx_s] = 1
                        idx_s += 1
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    if time.time() > deadline - 2:
        # Emergency: return what we have
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
