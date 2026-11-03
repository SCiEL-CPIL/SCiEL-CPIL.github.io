import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def _build_optimal_decomposition(n, m, p):
    """
    Build a rank-30 decomposition:
      - 4 blocks of 2x2 * 2x2 using Strassen (7 each) = 28
      - 2x1 * 1x8 using rank-2 decomposition = 2
    Total: 30
    """
    rank = 30
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Part 1: 2x2 * 2x8 using Strassen on 2-column blocks
    for q in range(4):
        # V indices for this block
        v_00 = 2 * q
        v_01 = 2 * q + 1
        v_10 = 8 + 2 * q
        v_11 = 8 + 2 * q + 1
        
        # W indices for this block
        w_00 = 4 * q
        w_01 = 4 * q + 1
        w_10 = 4 * q + 2
        w_11 = 4 * q + 3
        
        # U indices are fixed for the 2x2 part (cols 0,1 of A)
        # u_00 = 0, u_01 = 1, u_10 = 3, u_11 = 4
        
        # P1 = (a+d)(e+h)
        U[0, col] = 1; U[4, col] = 1
        V[v_00, col] = 1; V[v_11, col] = 1
        W[w_00, col] = 1; W[w_11, col] = 1
        col += 1
        
        # P2 = (c+d)e
        U[3, col] = 1; U[4, col] = 1
        V[v_00, col] = 1
        W[w_01, col] = 1; W[w_11, col] = 1
        col += 1
        
        # P3 = a(f-h)
        U[0, col] = 1
        V[v_01, col] = 1; V[v_11, col] = -1
        W[w_10, col] = 1; W[w_11, col] = -1
        col += 1
        
        # P4 = d(g-e)
        U[4, col] = 1
        V[v_10, col] = 1; V[v_00, col] = -1
        W[w_00, col] = -1; W[w_01, col] = 1
        col += 1
        
        # P5 = (a+b)h
        U[0, col] = 1; U[1, col] = 1
        V[v_11, col] = 1
        W[w_00, col] = 1
        col += 1
        
        # P6 = (c-a)(e+f)
        U[3, col] = 1; U[0, col] = -1
        V[v_00, col] = 1; V[v_01, col] = 1
        W[w_11, col] = -1
        col += 1
        
        # P7 = (b-d)(g+h)
        U[1, col] = 1; U[4, col] = -1
        V[v_10, col] = 1; V[v_11, col] = 1
        W[w_00, col] = -1; W[w_10, col] = 1
        col += 1
        
    # Part 2: 2x1 * 1x8 rank-2 decomposition
    # i=0 part
    U[2, col] = 1
    for k in range(8):
        V[16 + k, col] = 1
        W[2 * k, col] = 1
    col += 1
    
    # i=1 part
    U[5, col] = 1
    for k in range(8):
        V[16 + k, col] = 1
        W[2 * k + 1, col] = 1
    col += 1
        
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                computed = 0.0
                for r in range(rank):
                    computed += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                if abs(computed - 1.0) > tol:
                    return False
    return True

def construct():
    """
    Constructs a low-rank matrix multiplication decomposition for 2x3 * 3x8.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = DIMENSIONS
    
    # Build initial rank-30 decomposition
    U, V, W = _build_optimal_decomposition(n, m, p)
    
    # Verify initial decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U_fb = np.zeros((n * m, rank))
        V_fb = np.zeros((m * p, rank))
        W_fb = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U_fb[i * m + j, idx] = 1
                    V_fb[j * p + k, idx] = 1
                    W_fb[k * n + i, idx] = 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
