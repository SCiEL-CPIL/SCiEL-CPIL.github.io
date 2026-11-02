"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _strassen_6x6x6():
    """
    Strassen 2x2 block decomposition for 6x6x6 matrix multiplication.
    Split into 3x3 blocks. Each 3x3*3x3*3x3 has rank 27 (schoolbook).
    7 Strassen steps -> rank 7*27 = 189.
    
    Returns dict with U, V, W as lists of lists.
    """
    n = m = p = 6
    nh = mh = ph = 3  # half sizes
    block_rank = nh * mh * ph  # 27
    total_rank = 7 * block_rank  # 189
    
    # Strassen 2x2 block formulas:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # For each step s, define which A blocks and B blocks with what signs
    steps = [
        # (a_block_signs, b_block_signs)
        # a_block_signs: dict {block_idx: sign}
        # b_block_signs: dict {block_idx: sign}
        ({0: 1, 3: 1}, {0: 1, 3: 1}),       # M1: (A00+A11)*(B00+B11)
        ({2: 1, 3: 1}, {0: 1}),              # M2: (A10+A11)*B00
        ({0: 1}, {1: 1, 3: -1}),             # M3: A00*(B01-B11)
        ({3: 1}, {2: 1, 0: -1}),             # M4: A11*(B10-B00)
        ({0: 1, 1: 1}, {3: 1}),              # M5: (A00+A01)*B11
        ({2: 1, 0: -1}, {0: 1, 1: 1}),       # M6: (A10-A00)*(B00+B01)
        ({1: 1, 3: -1}, {2: 1, 3: 1}),       # M7: (A01-A11)*(B10+B11)
    ]
    
    # C block combinations: for each C block, which M_s contribute with what sign
    # C00 = M1 - M4 + M6 - M7  -> step indices 0,3,5,6 with signs 1,-1,1,-1
    # C01 = M3 + M5            -> step indices 2,4 with signs 1,1
    # C10 = M2 + M4            -> step indices 1,3 with signs 1,1
    # C11 = M1 - M2 + M3 + M7  -> step indices 0,1,2,6 with signs 1,-1,1,1
    c_combos = [
        {0: 1, 3: -1, 5: 1, 6: -1},   # C00
        {2: 1, 4: 1},                  # C01
        {1: 1, 3: 1},                  # C10
        {0: 1, 1: -1, 2: 1, 6: 1},    # C11
    ]
    
    # Initialize factor matrices
    U = [[0] * total_rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * total_rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * total_rank for _ in range(p * n)]  # 36 x 189
    
    for s in range(7):
        a_coeffs, b_coeffs = steps[s]
        for ai, a_sign in a_coeffs.items():
            for bi, b_sign in b_coeffs.items():
                # A block ai: rows [ai//2 * nh, ...), cols [ai%2 * mh, ...)
                # B block bi: rows [bi//2 * mh, ...), cols [bi%2 * ph, ...)
                a_row_start = (ai // 2) * nh
                a_col_start = (ai % 2) * mh
                b_row_start = (bi // 2) * mh
                b_col_start = (bi % 2) * ph
                
                for i in range(nh):
                    for j in range(mh):
                        for k in range(ph):
                            gi = a_row_start + i  # global row of A
                            gj = a_col_start + j  # global col of A / row of B
                            gk = b_col_start + k  # global col of B / col of C
                            
                            # C block index for this (gi, gk)
                            c_block_row = gi // nh
                            c_block_col = gk // ph
                            c_block_idx = c_block_row * 2 + c_block_col
                            
                            c_sign = c_combos[c_block_idx].get(s, 0)
                            if c_sign == 0:
                                continue
                            
                            col = s * block_rank + i * (mh * ph) + j * ph + k
                            
                            U[gi * m + gj][col] = a_sign
                            V[gj * p + gk][col] = b_sign
                            W[gk * n + gi][col] = c_sign
    
    return {"U": U, "V": V, "W": W}


def _verify_decomp(decomp, n, m, p):
    """
    Verify that the decomposition exactly decomposes the matrix multiplication tensor.
    T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1 for all (i,j,k).
    """
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0]) if U else 0
    
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U) or any(len(row) != rank for row in V) or any(len(row) != rank for row in W):
        return False
    
    # Use numpy for efficient verification
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # For each (i,j,k), compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This is equivalent to: for each c, the outer product structure
    # Let's check all (i,j,k) tuples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
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


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Try Strassen-based decomposition (rank 189)
    try:
        decomp = _strassen_6x6x6()
        if _verify_decomp(decomp, n, m, p):
            return decomp
    except Exception:
        pass
    
    # Fall back to schoolbook (rank 216)
    decomp = _schoolbook(n, m, p)
    return decomp

# EVOLVE-BLOCK-END
