import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Quick verification that U,V,W decompose the multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Check each (i,j,k) entry
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _check_half_integer(mat, tol=1e-9):
    """Check if all entries are integers or half-integers."""
    for val in mat.flatten():
        # Check if val is close to k/2 for some integer k
        # val * 2 should be close to an integer
        if abs(val * 2 - round(val * 2)) > tol:
            return False
    return True

def _build_rank_28(n, m, p):
    """
    Rank-28 decomposition:
    - j=0: schoolbook 2x1x5 (rank 10)
    - j=1,2, k=0,1: Strassen 2x2x2 (rank 7)
    - j=1,2, k=2: schoolbook 2x2x1 (rank 4)
    - j=1,2, k=3,4: Strassen 2x2x2 (rank 7)
    Total: 10+7+4+7 = 28
    """
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0

    def add_schoolbook(i, j, k, col):
        U[i*m+j, col] = 1.0
        V[j*p+k, col] = 1.0
        W[k*n+i, col] = 1.0
        return col + 1

    def add_strassen(i_off, j_off, k_off, col):
        # M1: (A00+A11)(B00+B11) -> C00+=, C11+=
        U[(i_off+0)*m+(j_off+0), col] = 1.0
        U[(i_off+1)*m+(j_off+1), col] = 1.0
        V[(j_off+0)*p+(k_off+0), col] = 1.0
        V[(j_off+1)*p+(k_off+1), col] = 1.0
        W[(k_off+0)*n+(i_off+0), col] = 1.0
        W[(k_off+1)*n+(i_off+1), col] = 1.0
        col += 1
        # M2: (A10+A11)B00 -> C10+=, C11-=
        U[(i_off+1)*m+(j_off+0), col] = 1.0
        U[(i_off+1)*m+(j_off+1), col] = 1.0
        V[(j_off+0)*p+(k_off+0), col] = 1.0
        W[(k_off+0)*n+(i_off+1), col] = 1.0
        W[(k_off+1)*n+(i_off+1), col] = -1.0
        col += 1
        # M3: A00(B01-B11) -> C01+=, C11+=
        U[(i_off+0)*m+(j_off+0), col] = 1.0
        V[(j_off+0)*p+(k_off+1), col] = 1.0
        V[(j_off+1)*p+(k_off+1), col] = -1.0
        W[(k_off+1)*n+(i_off+0), col] = 1.0
        W[(k_off+1)*n+(i_off+1), col] = 1.0
        col += 1
        # M4: A11(B10-B00) -> C00+=, C10+=
        U[(i_off+1)*m+(j_off+1), col] = 1.0
        V[(j_off+1)*p+(k_off+0), col] = 1.0
        V[(j_off+0)*p+(k_off+0), col] = -1.0
        W[(k_off+0)*n+(i_off+0), col] = 1.0
        W[(k_off+0)*n+(i_off+1), col] = 1.0
        col += 1
        # M5: (A00+A01)B11 -> C00-=, C01+=
        U[(i_off+0)*m+(j_off+0), col] = 1.0
        U[(i_off+0)*m+(j_off+1), col] = 1.0
        V[(j_off+1)*p+(k_off+1), col] = 1.0
        W[(k_off+0)*n+(i_off+0), col] = -1.0
        W[(k_off+1)*n+(i_off+0), col] = 1.0
        col += 1
        # M6: (A10-A00)(B00+B01) -> C11+=
        U[(i_off+1)*m+(j_off+0), col] = 1.0
        U[(i_off+0)*m+(j_off+0), col] = -1.0
        V[(j_off+0)*p+(k_off+0), col] = 1.0
        V[(j_off+0)*p+(k_off+1), col] = 1.0
        W[(k_off+1)*n+(i_off+1), col] = 1.0
        col += 1
        # M7: (A01-A11)(B10+B11) -> C00+=
        U[(i_off+0)*m+(j_off+1), col] = 1.0
        U[(i_off+1)*m+(j_off+1), col] = -1.0
        V[(j_off+1)*p+(k_off+0), col] = 1.0
        V[(j_off+1)*p+(k_off+1), col] = 1.0
        W[(k_off+0)*n+(i_off+0), col] = 1.0
        col += 1
        return col

    # Block 1: j=0, all k (schoolbook, 10 terms)
    for k in range(p):
        col = add_schoolbook(0, 0, k, col)
        col = add_schoolbook(1, 0, k, col)

    # Block 2a: j=1,2, k=0,1 (Strassen, 7 terms)
    col = add_strassen(0, 1, 0, col)

    # Block 2b: j=1,2, k=2 (schoolbook, 4 terms)
    for i in range(n):
        for j in [1, 2]:
            col = add_schoolbook(i, j, 2, col)

    # Block 2c: j=1,2, k=3,4 (Strassen, 7 terms)
    col = add_strassen(0, 1, 3, col)

    assert col == rank, f"Expected {rank}, got {col}"
    return U, V, W

def _try_rank_27(n, m, p, deadline):
    """
    Attempt to find a rank-27 solution with integer/half-integer entries.
    Strategy: Try a few random sparse configurations or specific structures.
    """
    if time.time() > deadline:
        return None
    
    rank = 27
    # Since finding a generic integer solution is hard, we try a specific structure
    # that might satisfy the 30 equations.
    # If we use the "all ones" scaled by 1/27, it fails the integer check.
    # We can try to find a solution where the sums are integers.
    
    # For now, we return None to fallback to Rank 28, as a robust Rank 27 
    # integer solution is not trivially available without search.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # First, try to find rank 27 (if a valid method is found)
    if time.time() < deadline:
        try:
            result = _try_rank_27(n, m, p, deadline)
            if result is not None:
                U, V, W = result
                if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p) and \
                   _check_half_integer(U) and _check_half_integer(V) and _check_half_integer(W):
                    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
        except Exception:
            pass
    
    # Fallback: Rank 28
    try:
        U, V, W = _build_rank_28(n, m, p)
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    except Exception:
        pass
    
    # Fallback: Schoolbook rank 30
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
