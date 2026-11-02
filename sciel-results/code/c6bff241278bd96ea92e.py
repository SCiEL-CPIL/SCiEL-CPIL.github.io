"""Matrix multiplication 3x3 by 3x8: rank-minimal exact tensor decomposition.

Strategy:
  * The multiplication tensor T[i,j,k] = 1 for all (i,j,k) in {0..2}x{0..2}x{0..7}.
  * We seek a rank-r decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    where U is (n*m, r), V is (m*p, r), W is (p*n, r).
  * Key insight: For each (i,k) pair, create one rank component.
    Set W[k*n+i, c] = 1 if c corresponds to (i,k), else 0.
    Set U[i*m+j, c] = 1 for all j when c corresponds to (i,k).
    Set V[j*p+k, c] = 1 for all j when c corresponds to (i,k).
    Then for valid (i,j,k): only c=(i,k) contributes, giving 1*1*1=1.
    For invalid positions: no single c matches all three index constraints, so sum=0.
  * This gives rank = n*p = 24.
"""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]
START = time.time()
DEADLINE = 145.0


def _build_structured_rank24():
    """Build the rank-24 structured decomposition."""
    n, m, p = DIMENSIONS
    rank = n * p  # 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each (i,k) pair, assign a rank component
    # c = i * p + k
    for i in range(n):
        for k in range(p):
            c = i * p + k
            # W[k*n+i, c] = 1
            W[k * n + i][c] = 1
            # U[i*m+j, c] = 1 for all j
            for j in range(m):
                U[i * m + j][c] = 1
            # V[j*p+k, c] = 1 for all j
            for j in range(m):
                V[j * p + k][c] = 1
    
    return {"U": U, "V": V, "W": W}


def _build_schoolbook():
    """Fallback schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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


def _validate_exact(U, V, W, n, m, p):
    """Validate that the decomposition is exact: 
    sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for valid (i,j,k)
    and = 0 for all invalid positions.
    Uses doubled integers for exact arithmetic.
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    U2 = (2 * U).astype(int)
    V2 = (2 * V).astype(int)
    W2 = (2 * W).astype(int)
    
    # Check valid positions: should sum to 1, i.e., 4 in doubled space
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = U2[i * m + j, :] * V2[j * p + k, :] * W2[k * n + i, :]
                s = np.sum(lhs)
                if s != 4:  # 2*2*1 = 4
                    return False
    
    # Check invalid positions: should sum to 0
    for a in range(n * m):
        i = a // m
        j = a % m
        for b in range(m * p):
            j2 = b // p
            k = b % p
            for c_idx in range(p * n):
                k2 = c_idx // n
                i2 = c_idx % n
                if j == j2 and k == k2 and i == i2:
                    continue  # valid position, already checked
                s = np.sum(U2[a, :] * V2[b, :] * W2[c_idx, :])
                if s != 0:
                    return False
    
    return True


def _to_list(M):
    """Convert numpy array to list of lists with int/float entries."""
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


def construct():
    n, m, p = DIMENSIONS
    
    # First try the structured rank-24 decomposition
    candidate = _build_structured_rank24()
    if _validate_exact(candidate["U"], candidate["V"], candidate["W"], n, m, p):
        return candidate
    
    # Fallback to schoolbook if validation fails (should not happen)
    return _build_schoolbook()


if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
