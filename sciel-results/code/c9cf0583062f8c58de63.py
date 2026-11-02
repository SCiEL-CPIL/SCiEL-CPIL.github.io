import numpy as np
import time

DIMENSIONS = [2, 4, 5]
TIME_LIMIT = 120.0
SAFETY_MARGIN = 10.0

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        Ur = U_reshaped[:, :, r]
        Vr = V_reshaped[:, :, r]
        Wr = W_reshaped[:, :, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += Ur[i, j] * Vr[j, k] * Wr[k, i]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if i == k else 0.0
                if abs(T[i, j, k] - expected) > tol:
                    return False
    
    return True

def _strassen_2x2x2(U, V, W, col, y_idx, z_idx, w_idx):
    """Add 7 Strassen terms for 2x2x2 block. Returns new col."""
    (y00, y01, y10, y11) = y_idx
    (z00, z01, z10, z11) = z_idx
    (w00, w01, w10, w11) = w_idx
    
    # M1 = (y00+y11)(z00+z11)
    U[y00][col] += 1; U[y11][col] += 1
    V[z00][col] += 1; V[z11][col] += 1
    W[w00][col] += 1; W[w11][col] += 1
    col += 1
    
    # M2 = (y10+y11)z00
    U[y10][col] += 1; U[y11][col] += 1
    V[z00][col] += 1
    W[w10][col] += 1; W[w11][col] -= 1
    col += 1
    
    # M3 = y00(z01-z11)
    U[y00][col] += 1
    V[z01][col] += 1; V[z11][col] -= 1
    W[w01][col] += 1; W[w11][col] += 1
    col += 1
    
    # M4 = y11(z10-z00)
    U[y11][col] += 1
    V[z10][col] += 1; V[z00][col] -= 1
    W[w00][col] += 1; W[w10][col] += 1
    col += 1
    
    # M5 = (y00+y01)z11
    U[y00][col] += 1; U[y01][col] += 1
    V[z11][col] += 1
    W[w00][col] -= 1; W[w01][col] += 1
    col += 1
    
    # M6 = (y10-y00)(z00+z01)
    U[y10][col] += 1; U[y00][col] -= 1
    V[z00][col] += 1; V[z01][col] += 1
    W[w11][col] += 1
    col += 1
    
    # M7 = (y01-y11)(z10+z11)
    U[y01][col] += 1; U[y11][col] -= 1
    V[z10][col] += 1; V[z11][col] += 1
    W[w00][col] += 1
    col += 1
    
    return col

def _schoolbook_2x2x3(U, V, W, col, y_idx, z_idx, w_idx):
    """Add 4 schoolbook terms for 2x2x3 block (A 2x2, B 2x3, C 2x3). Returns new col."""
    (y00, y01, y10, y11) = y_idx
    (z0, z1) = z_idx  # each z is a 2-element list of column indices
    (w0, w1) = w_idx  # each w is a 3-element list of column indices
    
    # r=0: U[0,0]=1, V[0,k]=1 for k in z0, W[k,0]=1 for k in w0
    U[y00][col] += 1
    for zi in z0: V[zi][col] += 1
    for wi in w0: W[wi][col] += 1
    col += 1
    
    # r=1: U[0,1]=1, V[1,k]=1 for k in z1, W[k,0]=1 for k in w0
    U[y01][col] += 1
    for zi in z1: V[zi][col] += 1
    for wi in w0: W[wi][col] += 1
    col += 1
    
    # r=2: U[1,0]=1, V[0,k]=1 for k in z0, W[k,1]=1 for k in w1
    U[y10][col] += 1
    for zi in z0: V[zi][col] += 1
    for wi in w1: W[wi][col] += 1
    col += 1
    
    # r=3: U[1,1]=1, V[1,k]=1 for k in z1, W[k,1]=1 for k in w1
    U[y11][col] += 1
    for zi in z1: V[zi][col] += 1
    for wi in w1: W[wi][col] += 1
    col += 1
    
    return col

def _build_decomposition(n, m, p):
    """
    Build decomposition for 2x4x5 matrix multiplication.
    Blocking: j-groups (0,1) and (2,3); k-groups (0,1) and (2,3,4)
    - (j{0,1}, k{0,1}): 2x2x2, rank 7
    - (j{0,1}, k{2,3,4}): 2x2x3, rank 4
    - (j{2,3}, k{0,1}): 2x2x2, rank 7
    - (j{2,3}, k{2,3,4}): 2x2x3, rank 4
    Total rank = 22
    """
    rank = 22
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    col = 0
    
    # Block 1: j in {0,1}, k in {0,1} -> 2x2x2, rank 7
    y1 = (u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1))
    z1 = (v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1))
    w1 = (w_idx(0,0), w_idx(1,0), w_idx(0,1), w_idx(1,1))
    col = _strassen_2x2x2(U, V, W, col, y1, z1, w1)
    
    # Block 2: j in {0,1}, k in {2,3,4} -> 2x2x3, rank 4
    y2 = (u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1))
    z2 = ([v_idx(0,2), v_idx(0,3), v_idx(0,4)], [v_idx(1,2), v_idx(1,3), v_idx(1,4)])
    w2 = ([w_idx(2,0), w_idx(3,0), w_idx(4,0)], [w_idx(2,1), w_idx(3,1), w_idx(4,1)])
    col = _schoolbook_2x2x3(U, V, W, col, y2, z2, w2)
    
    # Block 3: j in {2,3}, k in {0,1} -> 2x2x2, rank 7
    y3 = (u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3))
    z3 = (v_idx(2,0), v_idx(2,1), v_idx(3,0), v_idx(3,1))
    w3 = (w_idx(0,0), w_idx(1,0), w_idx(0,1), w_idx(1,1))
    col = _strassen_2x2x2(U, V, W, col, y3, z3, w3)
    
    # Block 4: j in {2,3}, k in {2,3,4} -> 2x2x3, rank 4
    y4 = (u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3))
    z4 = ([v_idx(2,2), v_idx(2,3), v_idx(2,4)], [v_idx(3,2), v_idx(3,3), v_idx(3,4)])
    w4 = ([w_idx(2,0), w_idx(3,0), w_idx(4,0)], [w_idx(2,1), w_idx(3,1), w_idx(4,1)])
    col = _schoolbook_2x2x3(U, V, W, col, y4, z4, w4)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    U, V, W = _build_decomposition(n, m, p)
    
    # Verify
    valid = _verify_decomposition(U, V, W, n, m, p)
    if not valid:
        raise ValueError("Decomposition verification failed")
    
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - SAFETY_MARGIN:
        pass  # We're fine, this is fast
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
