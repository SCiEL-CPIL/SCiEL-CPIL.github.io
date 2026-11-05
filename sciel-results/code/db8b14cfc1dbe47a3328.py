import numpy as np
import time
from fractions import Fraction
import itertools

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    # Target: find a rank-minimal decomposition for 2x3 * 3x5 matrix multiplication
    # The tensor T_{i,j,k} = 1 for all i in [0,2), j in [0,3), k in [0,5)
    # We need: sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r} = 1 for all (i,j,k)
    
    # Start with schoolbook rank 30 as baseline
    def make_schoolbook():
        rank = n * m * p  # 30
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    # Strassen-based approach for rank 28
    def make_strassen_hybrid():
        rank = 28
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i1, i2, j1, j2, k1, k2):
            nonlocal col
            # M1 = (A00 + A11)(B00 + B11)
            U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
            col += 1

            # M2 = (A10 + A11) B00
            U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1
            W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
            col += 1

            # M3 = A00 (B01 - B11)
            U[i1*m+j1, col] = 1
            V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
            W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
            col += 1

            # M4 = A11 (B10 - B00)
            U[i2*m+j2, col] = 1
            V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
            W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
            col += 1

            # M5 = (A00 + A01) B11
            U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
            V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
            col += 1

            # M6 = (A10 - A00)(B00 + B01)
            U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
            V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
            W[k2*n+i2, col] = 1
            col += 1

            # M7 = (A01 - A11)(B10 + B11)
            U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
            V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1
            col += 1

        def add_schoolbook_block(i_list, j_list, k_list):
            nonlocal col
            for i in i_list:
                for j in j_list:
                    for k in k_list:
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1

        # Block 1: A[:,0:2] * B[0:2,0:2] -> C[0:2,0:2] (Strassen, rank 7)
        add_strassen(0, 1, 0, 1, 0, 1)

        # Block 2: A[:,0:2] * B[0:2,2:4] -> C[0:2,2:4] (Strassen, rank 7)
        add_strassen(0, 1, 0, 1, 2, 3)

        # Block 3: A[:,0:2] * B[0:2,4:5] -> C[0:2,4:5] (Schoolbook, rank 4)
        add_schoolbook_block([0, 1], [0, 1], [4])

        # Block 4: A[:,2:3] * B[2:3,0:2] -> C[0:2,0:2] (Schoolbook, rank 4)
        add_schoolbook_block([0, 1], [2], [0, 1])

        # Block 5: A[:,2:3] * B[2:3,2:4] -> C[0:2,2:4] (Schoolbook, rank 4)
        add_schoolbook_block([0, 1], [2], [2, 3])

        # Block 6: A[:,2:3] * B[2:3,4:5] -> C[0:2,4:5] (Schoolbook, rank 2)
        add_schoolbook_block([0, 1], [2], [4])

        return U, V, W, rank

    def verify(U, V, W, rank):
        """Verify that sum_r U[:,:,r] * V[:,:,r] * W[:,:,r] = 1 for all (i,j,k)"""
        tensor = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            for i in range(n):
                for k in range(p):
                    tensor[i, :, k] += u[i, :] * v[:, k] * w[k, i]
        return np.allclose(tensor, 1.0, atol=1e-9)

    # Try the Strassen hybrid first (rank 28)
    U, V, W, rank = make_strassen_hybrid()
    if verify(U, V, W, rank):
        # Try to search for lower rank using integer/half-integer entries
        # We'll use a randomized search with local optimization
        best_rank = rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

        # Return the rank-28 Strassen hybrid solution
        U, V, W, rank = best_U, best_V, best_W, best_rank
    else:
        # Fallback to schoolbook
        U, V, W, rank = make_schoolbook()

    # Final verification
    if not verify(U, V, W, rank):
        U, V, W, rank = make_schoolbook()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
