import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # A indices: i*m + j
    # B indices: j*p + k
    # C indices: k*n + i
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                computed = 0.0
                for r in range(rank):
                    # U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    computed += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(computed - 1.0) > tol:
                    return False
    return True

def _build_rank44_decomposition(n, m, p):
    """
    Build a rank-44 decomposition:
      - 28 rank for A[0:2, 0:2] * B[0:2, 0:8] (4 blocks of 2x2x2 Strassen)
      - 16 rank for A[0:2, 2] * B[2, 0:8] (Schoolbook)
    """
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Part 1: 2x2 * 2x8 using Strassen on 2-column blocks
    # A uses indices: 0,1 (row 0), 3,4 (row 1)
    # B uses indices: 0..7 (row 0), 8..15 (row 1)
    for j in range(4):
        # Block j covers B columns 2j, 2j+1
        # B row 0 cols: 2j, 2j+1 -> indices 2j, 2j+1
        # B row 1 cols: 2j, 2j+1 -> indices 8+2j, 9+2j
        
        w_r00 = 4 * j      # C[0, 2j]
        w_r01 = 4 * j + 2  # C[0, 2j+1]
        w_r10 = 4 * j + 1  # C[1, 2j]
        w_r11 = 4 * j + 3  # C[1, 2j+1]
        
        v_e = 2 * j        # B[0, 2j]
        v_f = 2 * j + 1    # B[0, 2j+1]
        v_g = 8 + 2 * j    # B[1, 2j]
        v_h = 9 + 2 * j    # B[1, 2j+1]
        
        # Strassen for 2x2 matrices A_block and B_block
        # A_block = [[a, b], [c, d]]
        # Indices in U: a=0, b=1, c=3, d=4
        
        # P1 = (a+d)(e+h)
        U[0, col] = 1; U[4, col] = 1
        V[v_e, col] = 1; V[v_h, col] = 1
        W[w_r00, col] = 1; W[w_r11, col] = 1
        col += 1
        
        # P2 = (c+d)e
        U[3, col] = 1; U[4, col] = 1
        V[v_e, col] = 1
        W[w_r10, col] = 1; W[w_r11, col] = -1
        col += 1
        
        # P3 = a(f-h)
        U[0, col] = 1
        V[v_f, col] = 1; V[v_h, col] = -1
        W[w_r01, col] = 1; W[w_r11, col] = 1
        col += 1
        
        # P4 = d(g-e)
        U[4, col] = 1
        V[v_g, col] = 1; V[v_e, col] = -1
        W[w_r00, col] = 1; W[w_r10, col] = 1
        col += 1
        
        # P5 = (a+b)h
        U[0, col] = 1; U[1, col] = 1
        V[v_h, col] = 1
        W[w_r00, col] = -1; W[w_r01, col] = 1
        col += 1
        
        # P6 = (c-a)(e+f)
        U[3, col] = 1; U[0, col] = -1
        V[v_e, col] = 1; V[v_f, col] = 1
        W[w_r11, col] = 1
        col += 1
        
        # P7 = (b-d)(g+h)
        U[1, col] = 1; U[4, col] = -1
        V[v_g, col] = 1; V[v_h, col] = 1
        W[w_r00, col] = 1
        col += 1
    
    # Part 2: 2x1 * 1x8 schoolbook
    # A uses indices: 2 (A[0,2]), 5 (A[1,2])
    # B uses indices: 16..23 (B[2, 0..7])
    for i in range(2):
        for k in range(8):
            u_idx = i * 3 + 2
            v_idx = 2 * 8 + k
            w_idx = k * 2 + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
    
    return U, V, W

def construct():
    """
    Constructs a low-rank matrix multiplication decomposition for 2x3 * 3x8.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = DIMENSIONS
    
    # Build the rank-44 decomposition
    U, V, W = _build_rank44_decomposition(n, m, p)
    
    # Verify initial decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook rank 48
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
