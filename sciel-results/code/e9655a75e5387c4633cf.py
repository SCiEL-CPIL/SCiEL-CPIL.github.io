"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank decomposition for 2x2 * 2x9 matrix multiplication.
    Returns dict with U (4 x rank), V (18 x rank), W (18 x rank) as lists.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 2, 9
    # U: (n*m) x rank = 4 x rank
    # V: (m*p) x rank = 18 x rank
    # W: (p*n) x rank = 18 x rank
    
    # Strategy:
    # Decompose 2x2 * 2x9 into:
    # 4 blocks of 2x2 * 2x2 (Strassen, rank 7 each) for cols 0-7
    # 1 block of 2x2 * 2x1 (Schoolbook, rank 4) for col 8
    # Total rank = 4*7 + 4 = 32
    
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        # Strassen decomposition for 2x2 * 2x2 block at columns k0, k1
        # Indices in B: b0k0=k0, b0k1=k1, b1k0=9+k0, b1k1=9+k1
        # Indices in C: c0k0=2k0, c0k1=2k1, c1k0=2k0+1, c1k1=2k1+1
        
        # M1: (a00+a11)(b0k0+b1k1) -> c0k0 + c1k1
        U[0, col] = 1
        U[3, col] = 1
        V[k0, col] = 1
        V[9 + k1, col] = 1
        W[2 * k0, col] = 1
        W[2 * k1 + 1, col] = 1
        col += 1
        
        # M2: (a10+a11)b0k0 -> c1k0 - c1k1
        U[2, col] = 1
        U[3, col] = 1
        V[k0, col] = 1
        W[2 * k0 + 1, col] = 1
        W[2 * k1 + 1, col] = -1
        col += 1
        
        # M3: a00(b0k1-b1k1) -> c0k1 + c1k1
        U[0, col] = 1
        V[k1, col] = 1
        V[9 + k1, col] = -1
        W[2 * k1, col] = 1
        W[2 * k1 + 1, col] = 1
        col += 1
        
        # M4: a11(b1k0-b0k0) -> c0k0 + c1k0
        U[3, col] = 1
        V[9 + k0, col] = 1
        V[k0, col] = -1
        W[2 * k0, col] = 1
        W[2 * k0 + 1, col] = 1
        col += 1
        
        # M5: (a00+a01)b1k1 -> -c0k0 + c0k1
        U[0, col] = 1
        U[1, col] = 1
        V[9 + k1, col] = 1
        W[2 * k0, col] = -1
        W[2 * k1, col] = 1
        col += 1
        
        # M6: (a10-a00)(b0k0+b0k1) -> c1k1
        U[2, col] = 1
        U[0, col] = -1
        V[k0, col] = 1
        V[k1, col] = 1
        W[2 * k1 + 1, col] = 1
        col += 1
        
        # M7: (a01-a11)(b1k0+b1k1) -> c0k0
        U[1, col] = 1
        U[3, col] = -1
        V[9 + k0, col] = 1
        V[9 + k1, col] = 1
        W[2 * k0, col] = 1
        col += 1

    def add_schoolbook_2x2_1(k):
        nonlocal col
        # Schoolbook for 2x2 * 2x1 block at column k
        # C0k = A00*B0k + A01*B1k
        # C1k = A10*B0k + A11*B1k
        
        # Term 1: A00 * B0k -> C0k
        U[0, col] = 1
        V[k, col] = 1
        W[2 * k, col] = 1
        col += 1
        
        # Term 2: A01 * B1k -> C0k
        U[1, col] = 1
        V[9 + k, col] = 1
        W[2 * k, col] = 1
        col += 1
        
        # Term 3: A10 * B0k -> C1k
        U[2, col] = 1
        V[k, col] = 1
        W[2 * k + 1, col] = 1
        col += 1
        
        # Term 4: A11 * B1k -> C1k
        U[3, col] = 1
        V[9 + k, col] = 1
        W[2 * k + 1, col] = 1
        col += 1

    # Apply blocks
    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_schoolbook_2x2_1(8)
    
    # Verify the decomposition
    # T_decomposed[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    # True tensor T_true[i, j, k] = 1 if i=a*m+b, j=b*p+c, k=c*n+a else 0
    
    try:
        T_decomposed = np.einsum('ir,jr,kr->ijk', U, V, W)
        
        T_true = np.zeros((n * m, m * p, p * n), dtype=np.int8)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    T_true[idx_u, idx_v, idx_w] = 1
        
        if not np.allclose(T_decomposed, T_true, atol=1e-9):
            # Fallback to schoolbook rank 36
            rank_fb = n * m * p
            U_fb = np.zeros((n * m, rank_fb))
            V_fb = np.zeros((m * p, rank_fb))
            W_fb = np.zeros((p * n, rank_fb))
            col_fb = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j, col_fb] = 1
                        V_fb[j * p + k, col_fb] = 1
                        W_fb[k * n + i, col_fb] = 1
                        col_fb += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
            
    except Exception:
        # Fallback on error
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col_fb] = 1
                    V_fb[j * p + k, col_fb] = 1
                    W_fb[k * n + i, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
