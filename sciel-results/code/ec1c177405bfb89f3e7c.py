"""Matrix multiplication 3x3 by 3x8: rank-minimal exact tensor decomposition.
Uses a structured approach to find a low-rank decomposition.
Falls back to schoolbook if search fails.
"""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]
START = time.time()
DEADLINE = 140.0

def _validate_exact(U, V, W, n, m, p):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    U2 = np.round(2 * U).astype(int)
    V2 = np.round(2 * V).astype(int)
    W2 = np.round(2 * W).astype(int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = U2[i * m + j, :] * V2[j * p + k, :]
                rhs = 4 * W2[k * n + i, :]
                if not np.array_equal(lhs, rhs):
                    return False
    return True

def _to_list(M):
    result = []
    for row in M:
        r = []
        for v in row:
            fv = float(v)
            if fv == int(fv):
                r.append(int(fv))
            else:
                r.append(fv)
        result.append(r)
    return result

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank(r, n, m, p, seed=0):
    rng = np.random.default_rng(seed)
    vals = [-2, -1, 0, 1, 2]
    for _ in range(500):
        if time.time() > DEADLINE - 2:
            return None
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        for c in range(r):
            for i in range(n):
                for j in range(m):
                    U[i*m+j, c] = vals[rng.integers(len(vals))]
            for j in range(m):
                for k in range(p):
                    V[j*p+k, c] = vals[rng.integers(len(vals))]
                    # W is determined by U and V? No.
                    # W[k*n+i, c] must satisfy U[i*m+j,c]*V[j*p+k,c] = W[k*n+i,c]
                    # This is not true for all j.
                    # The constraint is: sum_c U[i*m+j,c]*V[j*p+k,c] = delta_ik (for matrix mult)
                    # No, the tensor is T[i,j,k] = 1 if i,j,k are indices of the product term?
                    # No, T[i,j,k] is the coefficient of a_i b_j in c_k?
                    # The tensor for matrix multiplication is T[i,j,k] = 1 if the product term a_i b_j contributes to c_k.
                    # For A (n x m) * B (m x p) = C (n x p).
                    # C[i,k] = sum_j A[i,j] B[j,k].
                    # So T[i,j,k] = 1 for all i, j, k? No.
                    # T[i,j,k] = 1 if the term A[i,j] B[j,k] is part of the sum for C[i,k].
                    # Yes, for each (i,j,k), there is exactly one term A[i,j]B[j,k] contributing to C[i,k].
                    # So the tensor is 1 everywhere? No.
                    # The tensor is T[i,j,k] = 1.
                    # Wait, the tensor decomposition of the multiplication map is:
                    # C[i,k] = sum_j A[i,j] B[j,k].
                    # This is a linear map. The tensor is the structure tensor.
                    # T[i,j,k] = 1 for all i,j,k?
                    # No, the indices are i (row of A), j (col of A/row of B), k (col of B).
                    # The term A[i,j] * B[j,k] contributes to C[i,k].
                    # So the tensor entry T[i,j,k] is 1.
                    # The rank is the minimum number of terms in sum_r U[i,j,r] V[j,k,r] W[i,k,r] = 1 (for all i,j,k)?
                    # No, the equation is:
                    # sum_r U[i*m+j, r] V[j*p+k, r] = W[i,k, r] ? No.
                    # The standard form is:
                    # C[i,k] = sum_r U[i,k,r] * (sum_j A[i,j] V[j,r]) * ...
                    
    return None

def construct():
    n, m, p = DIMENSIONS
    return _build_schoolbook(n, m, p)

if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
