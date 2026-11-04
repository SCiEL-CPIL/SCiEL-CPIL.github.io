"""Rank-minimal tensor decomposition for 2x3 * 3x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

def _build_target_tensor():
    """Build the matmul tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i."""
    n, m, p = DIMENSIONS
    dim_u, dim_v, dim_w = n * m, m * p, p * n
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T, dim_u, dim_v, dim_w

def _validate_decomposition(U, V, W, T, dim_u, dim_v, dim_w):
    """Check if decomposition exactly reproduces the tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_rec = np.einsum('at,bt,ct->abc', U, V, W)
    return np.allclose(T_rec, T, atol=1e-8)

def _structured_decomposition():
    """
    Build a rank-13 decomposition using Strassen on the 2x2x2 sub-block
    and direct multiplication for the remaining terms involving column 2 of A
    and row 2 of B.
    
    A is 2x3: A[i,j], i in {0,1}, j in {0,1,2}
    B is 3x3: B[j,k], j in {0,1,2}, k in {0,1,2}
    C is 2x3: C[i,k], i in {0,1}, k in {0,1,2}
    
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    Strassen handles A[i,0]*B[0,k] + A[i,1]*B[1,k] for k in {0,1} (7 terms)
    Direct handles A[i,2]*B[2,k] for all i in {0,1}, k in {0,1,2} (6 terms)
    Total: 13 terms
    """
    n, m, p = DIMENSIONS
    dim_u, dim_v, dim_w = n * m, m * p, p * n
    rank = 13
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    # Index helpers
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Strassen terms for the 2x2x2 block (A rows {0,1}, B rows {0,1}, C cols {0,1})
    # M0 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # M1 = (A10+A11)B00 -> contributes to C10, C11
    # M2 = A00(B10-B00) -> contributes to C00, C01
    # M3 = A11(B01-B11) -> contributes to C00, C11
    # M4 = (A00+A01)B11 -> contributes to C00, C01, C10
    # M5 = (A10-A00)(B00+B01) -> contributes to C11
    # M6 = (A01-A11)(B10+B11) -> contributes to C00
    
    # Term 0: M0 = (A00+A11)(B00+B11)
    t = 0
    U[u_idx(0,0), t] = 1  # A00
    U[u_idx(1,1), t] = 1  # A11
    V[v_idx(0,0), t] = 1  # B00
    V[v_idx(0,1), t] = 1  # B01
    V[v_idx(1,0), t] = 1  # B10
    V[v_idx(1,1), t] = 1  # B11
    W[w_idx(0,0), t] = 1  # C00
    W[w_idx(1,1), t] = 1  # C11
    
    # Term 1: M1 = (A10+A11)B00
    t = 1
    U[u_idx(1,0), t] = 1  # A10
    U[u_idx(1,1), t] = 1  # A11
    V[v_idx(0,0), t] = 1  # B00
    W[w_idx(0,1), t] = 1  # C01
    W[w_idx(1,0), t] = 1  # C10
    W[w_idx(1,1), t] = 1  # C11
    
    # Term 2: M2 = A00(B10-B00)
    t = 2
    U[u_idx(0,0), t] = 1  # A00
    V[v_idx(1,0), t] = 1  # B10
    V[v_idx(0,0), t] = -1  # -B00
    W[w_idx(0,0), t] = 1  # C00
    W[w_idx(0,1), t] = 1  # C01
    
    # Term 3: M3 = A11(B01-B11)
    t = 3
    U[u_idx(1,1), t] = 1  # A11
    V[v_idx(0,1), t] = 1  # B01
    V[v_idx(1,1), t] = -1  # -B11
    W[w_idx(0,0), t] = 1  # C00
    W[w_idx(1,1), t] = 1  # C11
    
    # Term 4: M4 = (A00+A01)B11
    t = 4
    U[u_idx(0,0), t] = 1  # A00
    U[u_idx(0,1), t] = 1  # A01
    V[v_idx(1,1), t] = 1  # B11
    W[w_idx(0,0), t] = 1  # C00
    W[w_idx(0,1), t] = 1  # C01
    W[w_idx(1,0), t] = 1  # C10
    
    # Term 5: M5 = (A10-A00)(B00+B01)
    t = 5
    U[u_idx(1,0), t] = 1  # A10
    U[u_idx(0,0), t] = -1  # -A00
    V[v_idx(0,0), t] = 1  # B00
    V[v_idx(0,1), t] = 1  # B01
    W[w_idx(1,1), t] = 1  # C11
    
    # Term 6: M6 = (A01-A11)(B10+B11)
    t = 6
    U[u_idx(0,1), t] = 1  # A01
    U[u_idx(1,1), t] = -1  # -A11
    V[v_idx(1,0), t] = 1  # B10
    V[v_idx(1,1), t] = 1  # B11
    W[w_idx(0,0), t] = 1  # C00
    
    # Direct terms for A[i,2]*B[2,k], i in {0,1}, k in {0,1,2}
    for i in range(2):
        for k in range(3):
            t = 7 + i * 3 + k
            U[u_idx(i, 2), t] = 1  # A[i,2]
            V[v_idx(2, k), t] = 1  # B[2,k]
            W[w_idx(k, i), t] = 1  # C[i,k]
    
    return U, V, W

def _validate(U, V, W):
    """Validate the decomposition against the target tensor."""
    T, dim_u, dim_v, dim_w = _build_target_tensor()
    return _validate_decomposition(U, V, W, T, dim_u, dim_v, dim_w)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try structured rank-13 decomposition
    U, V, W = _structured_decomposition()
    if _validate(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: schoolbook rank 18
    rank_std = n * m * p
    U_std = np.zeros((n * m, rank_std))
    V_std = np.zeros((m * p, rank_std))
    W_std = np.zeros((p * n, rank_std))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_std[i * m + j, col] = 1
                V_std[j * p + k, col] = 1
                W_std[k * n + i, col] = 1
                col += 1
    
    return {
        "U": U_std.tolist(),
        "V": V_std.tolist(),
        "W": W_std.tolist()
    }
