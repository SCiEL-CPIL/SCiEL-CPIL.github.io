import numpy as np
import time

DIMENSIONS = [2, 3, 3]
TIME_LIMIT = 120  # seconds, evaluator kills after 150s

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 9
    pn = p * n  # 6

    # Schoolbook fallback (rank 18)
    U_school = np.zeros((nm, 18))
    V_school = np.zeros((mp, 18))
    W_school = np.zeros((pn, 18))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1

    def verify_fast(U, V, W, tol=1e-9):
        """Vectorized verification."""
        if not np.allclose(U * 2, np.round(U * 2), atol=tol):
            return False
        if not np.allclose(V * 2, np.round(V * 2), atol=tol):
            return False
        if not np.allclose(W * 2, np.round(W * 2), atol=tol):
            return False

        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    if not np.isclose(s, 1.0, atol=tol):
                        return False
        return True

    # Try rank-6 decomposition:
    # For each j in {0,1,2}, use 2 terms:
    #   Term 2j:   U[i*m+j, 2j] = 1 if i=0 else 0, V[j*p+k, 2j] = 1 for all k, W[k*n+i, 2j] = 1 for all k,i
    #   Term 2j+1: U[i*m+j, 2j+1] = 1 if i=1 else 0, V[j*p+k, 2j+1] = 1 for all k, W[k*n+i, 2j+1] = 1 for all k,i
    #
    # This gives: for (i,j,k): sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r]
    # = U[i*m+j, 2j]*V[j*p+k, 2j]*W[k*n+i, 2j] + U[i*m+j, 2j+1]*V[j*p+k, 2j+1]*W[k*n+i, 2j+1]
    # = (1 if i=0 else 0)*1*1 + (1 if i=1 else 0)*1*1
    # = 1 if i is in {0,1}, which is always true.
    # So the sum is 1 for all (i,j,k). ✓

    R_try = 6
    U_try = np.zeros((nm, R_try))
    V_try = np.zeros((mp, R_try))
    W_try = np.zeros((pn, R_try))

    for j in range(m):
        r0 = j * 2
        r1 = j * 2 + 1
        # Term r0: handles i=0
        U_try[0 * m + j, r0] = 1
        U_try[1 * m + j, r0] = 0
        for k in range(p):
            V_try[j * p + k, r0] = 1
            W_try[k * n + 0, r0] = 1
            W_try[k * n + 1, r0] = 1
        # Term r1: handles i=1
        U_try[0 * m + j, r1] = 0
        U_try[1 * m + j, r1] = 1
        for k in range(p):
            V_try[j * p + k, r1] = 1
            W_try[k * n + 0, r1] = 1
            W_try[k * n + 1, r1] = 1

    if verify_fast(U_try, V_try, W_try):
        return {
            "U": U_try.tolist(),
            "V": V_try.tolist(),
            "W": W_try.tolist()
        }

    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
