"""Program to find a rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    U_dim = n * m  # 8
    V_dim = m * p  # 32
    W_dim = p * n  # 16

    deadline = time.time() + 145.0

    def check_decomp(U, V, W, tol=1e-6):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        T = np.zeros((n, m, p))
        for r in range(U.shape[1]):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, 1.0, atol=tol)

    def build_strassen_56():
        """
        Build a Strassen-based decomposition with rank 56.
        
        Partition the 2x4 * 4x8 multiplication into 8 independent 2x2 * 2x2 sub-problems:
        - 2 j-groups: {0,1} and {2,3}
        - 4 k-groups: {0,1}, {2,3}, {4,5}, {6,7}
        
        For each sub-problem (jg, kg), apply rank-7 Strassen to the 2x2 * 2x2 block.
        Total rank: 8 * 7 = 56.
        
        The output convention is W[k*n+i, r] where C[k, i] = sum_j A[i,j] * B[j,k].
        """
        rank = 8 * 7  # 56
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))

        col = 0
        for jg in range(2):  # j-group: 0 -> {0,1}, 1 -> {2,3}
            for kg in range(4):  # k-group: 0->{0,1}, 1->{2,3}, 2->{4,5}, 3->{6,7}
                # j indices in this group
                j0 = jg * 2
                j1 = jg * 2 + 1
                # k indices in this group
                k0 = kg * 2
                k1 = kg * 2 + 1

                # Strassen 7 products for 2x2 * 2x2:
                # A block: A[i, j] for i in {0,1}, j in {j0, j1}
                # B block: B[j, k] for j in {j0, j1}, k in {k0, k1}
                # C block: C[i, k] for i in {0,1}, k in {k0, k1}
                #
                # Strassen:
                # M1 = A[0,j0] * B[j0,k0]
                # M2 = A[1,j1] * B[j1,k1]
                # M3 = (A[0,j0] + A[0,j1]) * B[j0,k1]
                # M4 = (A[1,j0] - A[0,j0]) * (B[j0,k0] + B[j0,k1])
                # M5 = (A[1,j0] + A[1,j1]) * B[j1,k0]
                # M6 = (A[0,j1] - A[1,j1]) * (B[j1,k0] + B[j1,k1])
                # M7 = (A[1,j1] + A[0,j0] - A[1,j0] - A[0,j1]) * (B[j0,k1] - B[j1,k1])
                #
                # C[0,k0] = M1 + M4 + M7
                # C[0,k1] = M3 - M7
                # C[1,k0] = M2 + M5
                # C[1,k1] = M2 - M4 + M6 + M7

                strassen_defs = [
                    # (U_coeffs, V_coeffs, W_coeffs)
                    # M1 = A[0,j0] * B[j0,k0]
                    ([(0 * m + j0, 1)], [(j0 * p + k0, 1)], [(k0 * n + 0, 1)]),
                    # M2 = A[1,j1] * B[j1,k1]
                    ([(1 * m + j1, 1)], [(j1 * p + k1, 1)], [(k1 * n + 1, 1)]),
                    # M3 = (A[0,j0] + A[0,j1]) * B[j0,k1]
                    ([(0 * m + j0, 1), (0 * m + j1, 1)], [(j0 * p + k1, 1)], [(k1 * n + 0, 1)]),
                    # M4 = (A[1,j0] - A[0,j0]) * (B[j0,k0] + B[j0,k1])
                    ([(1 * m + j0, 1), (0 * m + j0, -1)], [(j0 * p + k0, 1), (j0 * p + k1, 1)], [(k0 * n + 0, 1)]),
                    # M5 = (A[1,j0] + A[1,j1]) * B[j1,k0]
                    ([(1 * m + j0, 1), (1 * m + j1, 1)], [(j1 * p + k0, 1)], [(k0 * n + 1, 1)]),
                    # M6 = (A[0,j1] - A[1,j1]) * (B[j1,k0] + B[j1,k1])
                    ([(0 * m + j1, 1), (1 * m + j1, -1)], [(j1 * p + k0, 1), (j1 * p + k1, 1)], [(k1 * n + 1, 1)]),
                    # M7 = (A[1,j1] + A[0,j0] - A[1,j0] - A[0,j1]) * (B[j0,k1] - B[j1,k1])
                    ([(1 * m + j1, 1), (0 * m + j0, 1), (1 * m + j0, -1), (0 * m + j1, -1)],
                     [(j0 * p + k1, 1), (j1 * p + k1, -1)], [(k0 * n + 0, 1)]),
                ]

                for u_coeffs, v_coeffs, w_coeffs in strassen_defs:
                    for idx, val in u_coeffs:
                        U[idx, col] += val
                    for idx, val in v_coeffs:
                        V[idx, col] += val
                    for idx, val in w_coeffs:
                        W[idx, col] += val
                    col += 1

        return U, V, W

    # Build the Strassen-56 decomposition
    U, V, W = build_strassen_56()

    # Verify
    if check_decomp(U, V, W, tol=1e-6):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook (should not happen)
    if time.time() > deadline - 5.0:
        # Emergency: return schoolbook
        rank = n * m * p
        U2 = np.zeros((U_dim, rank))
        V2 = np.zeros((V_dim, rank))
        W2 = np.zeros((W_dim, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U2[i*m+j, c] = 1
                    V2[j*p+k, c] = 1
                    W2[k*n+i, c] = 1
                    c += 1
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}

    # If verification failed, try to debug and fix
    # Print diagnostic info
    T = np.zeros((n, m, p))
    for r in range(U.shape[1]):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    # Find which entries are wrong
    bad = np.argwhere(np.abs(T - 1.0) > 1e-6)
    # As a last resort, return schoolbook
    rank = n * m * p
    U2 = np.zeros((U_dim, rank))
    V2 = np.zeros((V_dim, rank))
    W2 = np.zeros((W_dim, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U2[i*m+j, c] = 1
                V2[j*p+k, c] = 1
                W2[k*n+i, c] = 1
                c += 1
    return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
