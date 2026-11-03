"""Matrix multiplication 2x3x10 tensor decomposition with rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]
DEADLINE = 120.0
SAFETY_MARGIN = 5.0

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Reshape
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    except Exception:
        return False

    return np.allclose(T, 1.0, atol=1e-9)

def _build_strassen55(n, m, p):
    """
    Build the rank-55 decomposition.
    A is 2x3, B is 3x10, C is 2x10.
    Split B into 5 blocks of 2 columns.
    For each block:
      Part 1: A[:, 0] * B[0, block] -> Rank 4 (2x2 block, but actually 2x1 * 1x2 = 2x2, rank 4? No, 2*1*2 = 4 terms? No, 2*2=4 terms if schoolbook. But we can do better?)
      Actually, A[:, 0] is 2x1. B[0, block] is 1x2.
      Product is 2x2.
      Rank of 2x1x2 is 2?
      C_{i,k} = A_{i,0} B_{0,k}.
      For a 2x2 block (k0, k1), we have 4 outputs.
      C_{0,k0} = A_{0,0} B_{0,k0}
      C_{0,k1} = A_{0,0} B_{0,k1}
      C_{1,k0} = A_{1,0} B_{0,k0}
      C_{1,k1} = A_{1,0} B_{0,k1}
      This requires 4 multiplications?
      Or can we do it in 2?
      M1 = A_{0,0} (B_{0,k0} + B_{0,k1}) -> C_{0,k0} + C_{0,k1}
      M2 = A_{1,0} (B_{0,k0} + B_{0,k1}) -> C_{1,k0} + C_{1,k1}
      M3 = A_{0,0} (B_{0,k0} - B_{0,k1}) -> C_{0,k0} - C_{0,k1}
      M4 = A_{1,0} (B_{0,k0} - B_{0,k1}) -> C_{1,k0} - C_{1,k1}
      This gives sums and differences.
      To get individual values, we need to add/subtract.
      But the tensor decomposition is linear.
      We need to form $C_{0,k0}$ exactly.
      If we use 4 terms, it's schoolbook.
      Can we use 2 terms?
      If we use 2 terms, we can only form 2 independent linear combinations.
      We have 4 outputs.
      So rank 4 is likely minimal for 2x1x2.
      
      Wait, the current code uses 20 terms for the first part (5 blocks * 4 terms/block? No, 20 terms total).
      In the current code:
      `for i in range(n): for k in range(p):` -> 2 * 10 = 20 terms.
      This is the schoolbook for the 2x1x10 part.
      
      Part 2: 5 blocks of 2x2x2.
      Each uses 7 terms (Strassen).
      Total 35.
      
      Total 55.
    """
    rank = 55
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    col = 0

    # Part A: j=0 (first column of A, first row of B)
    # A[i, 0] * B[0, k] -> C[i, k]
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            
    # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
    # A2 is 2x2, B2 is 2x10.
    # Split B2 into 5 blocks of 2x2.
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        # Strassen indices for U (A2 entries: A[0,1], A[0,2], A[1,1], A[1,2])
        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2

        # Strassen indices for V (B2 block entries: B[1,k0], B[1,k1], B[2,k0], B[2,k1])
        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1

        # W indices for outputs (C[0,k0], C[0,k1], C[1,k0], C[1,k1])
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        # M1 = (a11 + a22)(b11 + b22)
        U[u_11, col] += 1
        U[u_22, col] += 1
        V[v_11, col] += 1
        V[v_22, col] += 1
        W[w_00, col] += 1
        W[w_11, col] += 1
        col += 1

        # M2 = (a21 + a22)b11
        U[u_21, col] += 1
        U[u_22, col] += 1
        V[v_11, col] += 1
        W[w_10, col] += 1
        W[w_11, col] += -1
        col += 1

        # M3 = a11(b12 - b22)
        U[u_11, col] += 1
        V[v_12, col] += 1
        V[v_22, col] += -1
        W[w_01, col] += 1
        W[w_11, col] += 1
        col += 1

        # M4 = a22(b21 - b11)
        U[u_22, col] += 1
        V[v_21, col] += 1
        V[v_11, col] += -1
        W[w_00, col] += 1
        W[w_10, col] += 1
        col += 1

        # M5 = (a11 + a12)b22
        U[u_11, col] += 1
        U[u_12, col] += 1
        V[v_22, col] += 1
        W[w_00, col] += -1
        W[w_01, col] += 1
        col += 1

        # M6 = (a21 - a11)(b11 + b12)
        U[u_21, col] += 1
        U[u_11, col] += -1
        V[v_11, col] += 1
        V[v_12, col] += 1
        W[w_11, col] += 1
        col += 1

        # M7 = (a12 - a22)(b21 + b22)
        U[u_12, col] += 1
        U[u_22, col] += -1
        V[v_21, col] += 1
        V[v_22, col] += 1
        W[w_00, col] += 1
        col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build the rank-55 solution
    U, V, W = _build_strassen55(n, m, p)
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
