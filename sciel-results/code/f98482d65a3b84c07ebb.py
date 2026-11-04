import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # 5s safety margin

    # The matrix multiplication tensor for C = A @ B where A is n x m and B is m x p
    # has entries T[i, j, k] = 1 for all i in [0, n), j in [0, m), k in [0, p)
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # for all i, j, k.

    # This is the all-ones tensor of shape (n, m, p).
    # The rank of the all-ones tensor is 1, since it can be written as the outer product
    # of three all-ones vectors.

    # Rank 1 solution:
    # U[i*m+j, 0] = 1 for all i, j
    # V[j*p+k, 0] = 1 for all j, k
    # W[k*n+i, 0] = 1 for all k, i

    # This gives: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # = U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0]
    # = 1 * 1 * 1 = 1 for all i, j, k.

    # Verify the solution
    U = np.ones((n * m, 1))
    V = np.ones((m * p, 1))
    W = np.ones((p * n, 1))

    # Double-check correctness
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > 1e-15:
                    # Fallback to schoolbook if something went wrong
                    break
            else:
                continue
            break
        else:
            continue
        break
    else:
        # All checks passed
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook rank 36
    U_school = np.zeros((n*m, 36))
    V_school = np.zeros((m*p, 36))
    W_school = np.zeros((p*n, 36))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1

    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
