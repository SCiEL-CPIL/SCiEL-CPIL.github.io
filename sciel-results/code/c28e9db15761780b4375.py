"""Rewritten program: rank-55 exact decomposition for 2x3 @ 3x10 matrix multiplication.
Uses Strassen (rank 7) on five 2x2 blocks of the first two rows/cols, plus schoolbook
for the remaining row/col. Includes a time guard and a final sanity check.
"""
import numpy as np
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    DEADLINE = 120.0  # must finish well before 150s kill

    n, m, p = 2, 3, 10
    rank = 55

    U = np.zeros((n * m, rank), dtype=int)  # 6 x 55
    V = np.zeros((m * p, rank), dtype=int)  # 30 x 55
    W = np.zeros((p * n, rank), dtype=int)  # 20 x 55

    idx = 0

    # Strassen decomposition for each 2x2 block of columns (k0, k1)
    # A is 2x3, B is 3x10, C is 2x10
    # For columns k0, k1 of B and rows 0,1 of A:
    #   a00 = A[0,0] = U index 0
    #   a01 = A[0,1] = U index 1
    #   a10 = A[1,0] = U index 3
    #   a11 = A[1,1] = U index 4
    #   b0k0 = B[0,k0] = V index k0
    #   b0k1 = B[0,k1] = V index k1
    #   b1k0 = B[1,k0] = V index 10+k0
    #   b1k1 = B[1,k1] = V index 10+k1
    #   c0k0 = C[0,k0] = W index 2*k0
    #   c0k1 = C[0,k1] = W index 2*k1
    #   c1k0 = C[1,k0] = W index 2*k0+1
    #   c1k1 = C[1,k1] = W index 2*k1+1

    for l in range(5):
        k0 = 2 * l
        k1 = 2 * l + 1

        a00 = 0
        a01 = 1
        a10 = 3
        a11 = 4

        b0k0 = k0
        b0k1 = k1
        b1k0 = 10 + k0
        b1k1 = 10 + k1

        c0k0 = 2 * k0
        c0k1 = 2 * k1
        c1k0 = 2 * k0 + 1
        c1k1 = 2 * k1 + 1

        # M1 = (a00 + a11) * (b0k0 + b1k1)
        U[a00, idx] += 1
        U[a11, idx] += 1
        V[b0k0, idx] += 1
        V[b1k1, idx] += 1
        W[c0k0, idx] += 1
        W[c1k1, idx] += 1
        idx += 1

        # M2 = (a10 + a11) * b0k0
        U[a10, idx] += 1
        U[a11, idx] += 1
        V[b0k0, idx] += 1
        W[c1k0, idx] += 1
        W[c1k1, idx] -= 1
        idx += 1

        # M3 = a00 * (b0k1 - b1k1)
        U[a00, idx] += 1
        V[b0k1, idx] += 1
        V[b1k1, idx] -= 1
        W[c0k1, idx] += 1
        W[c1k1, idx] += 1
        idx += 1

        # M4 = a11 * (b1k0 - b0k0)
        U[a11, idx] += 1
        V[b1k0, idx] += 1
        V[b0k0, idx] -= 1
        W[c0k0, idx] += 1
        W[c1k0, idx] += 1
        idx += 1

        # M5 = (a00 + a01) * b1k1
        U[a00, idx] += 1
        U[a01, idx] += 1
        V[b1k1, idx] += 1
        W[c0k0, idx] -= 1
        W[c0k1, idx] += 1
        idx += 1

        # M6 = (a10 - a00) * (b0k0 + b0k1)
        U[a10, idx] += 1
        U[a00, idx] -= 1
        V[b0k0, idx] += 1
        V[b0k1, idx] += 1
        W[c1k1, idx] += 1
        idx += 1

        # M7 = (a01 - a11) * (b1k0 + b1k1)
        U[a01, idx] += 1
        U[a11, idx] -= 1
        V[b1k0, idx] += 1
        V[b1k1, idx] += 1
        W[c0k0, idx] += 1
        idx += 1

        if time.time() - t_start > DEADLINE - 30:
            break

    # Schoolbook for the third row of B (j=2) and third column of A (j=2)
    # C[i,k] += A[i,2] * B[2,k]
    # A[i,2] is at U index i*3+2, B[2,k] is at V index 20+k, C[k,i] is at W index 2*k+i
    for i in range(2):
        for k in range(10):
            u_idx = i * 3 + 2
            v_idx = 20 + k
            w_idx = 2 * k + i

            U[u_idx, idx] += 1
            V[v_idx, idx] += 1
            W[w_idx, idx] += 1
            idx += 1

            if time.time() - t_start > DEADLINE - 10:
                break
        if time.time() - t_start > DEADLINE - 10:
            break

    # Sanity check: verify the decomposition is correct
    # The tensor T[i,j,k] = 1 if C[i,k] = sum_j A[i,j]*B[j,k]
    # Our decomposition: sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] should equal T[i,j,k]
    # Actually the convention is: C[i,k] = sum_t (sum_j U[i,j,t]*A[i,j]) * (sum_j V[j,k,t]*B[j,k]) * W[k,i,t]
    # Wait, the standard tensor decomposition for matrix multiplication:
    # C[i,k] = sum_t (sum_j U[i,j,t] * A[i,j]) * (sum_j V[j,k,t] * B[j,k]) * W[k,i,t]
    # This is NOT right. The correct form is:
    # C[i,k] = sum_t W[k,i,t] * (sum_j U[i,j,t] * A[i,j]) * (sum_j V[j,k,t] * B[j,k])
    # No, that's not standard either. Let me think...
    #
    # The standard formulation: M_{n,m,p} tensor has entries T_{i,j,k} = delta_{ik} (for the multiplication)
    # Actually T_{i,j,k} = 1 if the product C[i,k] involves A[i,j]*B[j,k], so T_{i,j,k} = delta_{(i,k) from (i,j) and (j,k)}
    # More precisely, for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if this term appears, i.e., always 1 for all valid (i,j,k)
    #
    # A rank-r decomposition: T[i,j,k] = sum_t U[i,t] * V[j,t] * W[k,t]
    # Wait, but the interface says U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # So U is indexed by (i,j) flattened to i*m+j, V by (j,k) flattened to j*p+k, W by (k,i) flattened to k*n+i
    #
    # The decomposition is: T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # And T[i,j,k] = 1 for all i,j,k (since every combination contributes)
    #
    # Let me verify with a small example.

    # Quick verification on random inputs
    if time.time() - t_start < DEADLINE - 5:
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, size=(n, m))
        B_test = np.random.randint(-5, 5, size=(m, p))
        C_expected = A_test @ B_test

        # Compute via decomposition
        C_computed = np.zeros((n, p), dtype=int)
        for t in range(rank):
            u_vec = U[:, t]  # length n*m
            v_vec = V[:, t]  # length m*p
            w_vec = W[:, t]  # length p*n

            # u_vec[i*m + j] * A[i,j]
            u_sum = 0
            for i in range(n):
                for j in range(m):
                    u_sum += u_vec[i * m + j] * A_test[i, j]

            # v_vec[j*p + k] * B[j,k]
            v_sum = 0
            for j in range(m):
                for k in range(p):
                    v_sum += v_vec[j * p + k] * B_test[j, k]

            # w_vec[k*n + i]
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += w_vec[k * n + i] * u_sum * v_sum

        if not np.allclose(C_computed, C_expected):
            # If verification fails, fall back to pure schoolbook (rank 60)
            rank = n * m * p  # 60
            U = np.zeros((n * m, rank), dtype=int)
            V = np.zeros((m * p, rank), dtype=int)
            W = np.zeros((p * n, rank), dtype=int)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, idx] = 1
                        V[j * p + k, idx] = 1
                        W[k * n + i, idx] = 1
                        idx += 1
        else:
            # Verification passed
            pass

    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
# EVOLVE-BLOCK-END
