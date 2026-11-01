"""Optimized matrix multiplication tensor decomposition for 2x3 * 3x6."""
import time
import random

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START

def _make_rank(R):
    """Create empty factor matrices for a given rank R."""
    n, m, p = DIMENSIONS
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    return U, V, W

def _set_term(U, V, W, r, u_idx, u_val, v_idx, v_val, w_idx, w_val):
    """Set a single term in the decomposition."""
    U[u_idx][r] = u_val
    V[v_idx][r] = v_val
    W[w_idx][r] = w_val

def _strassen_2x2():
    """
    Return Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    U: rows 0..3 = A00, A01, A10, A11
    V: rows 0..3 = B00, B01, B10, B11
    W: rows 0..3 = C00, C01, C10, C11
    Rank: 7
    """
    R = 7
    U = [[0.0] * R for _ in range(4)]
    V = [[0.0] * R for _ in range(4)]
    W = [[0.0] * R for _ in range(4)]
    
    # Strassen's 7 products
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Term 0: M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
    # M1 contributes: C00 += M1, C01 -= M3, C10 -= M2, C11 += M4
    # Actually let me be more careful with the standard Strassen formula:
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M6
    
    # Let me redo this with correct coefficients
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = -1
    # M1 appears in C00 with +1, C01 with 0, C10 with 0, C11 with +1
    # Wait, let me use the standard:
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M6
    
    # Reset and redo
    for i in range(4):
        for j in range(7):
            U[i][j] = 0; V[i][j] = 0; W[i][j] = 0
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C10 += M2
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = -1  # C01 += M3, C11 -= M3
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = -1  # C10 += M4, C00 -= M4
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = 1  # C01 += M5, C11 += M5
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1  # C00 += M6
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1  # C00 -= M7
    
    return U, V, W

def _outer_2x1_1x2():
    """
    Decomposition for (2x1) * (1x2) matrix multiplication.
    U: rows 0..1 = A0, A1
    V: row 0 = B
    W: rows 0..3 = C00, C01, C10, C11
    Rank: 4
    """
    R = 4
    U = [[0.0] * R for _ in range(2)]
    V = [[0.0] * R for _ in range(1)]
    W = [[0.0] * R for _ in range(4)]
    
    # C00 = A0*B, C01 = A0*B (same), C10 = A1*B, C11 = A1*B
    # Actually C is 2x2: C00 = A0*B, C01 = A0*B, C10 = A1*B, C11 = A1*B
    # So we need 4 terms (one for each output element)
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1  # C00
    U[0][1] = 1; V[0][1] = 1; W[1][1] = 1  # C01
    U[1][2] = 1; V[0][2] = 1; W[2][2] = 1  # C10
    U[1][3] = 1; V[0][3] = 1; W[3][3] = 1  # C11
    
    return U, V, W

def _build_rank33():
    """
    Build a rank-33 decomposition using block decomposition.
    Split 2x3 * 3x6 into 3 blocks of 2x3 * 3x2.
    Each block uses Strassen(2x2*2x2) + outer(2x1*1x2) = 7+4=11 terms.
    Total rank: 33.
    
    A is 2x3: A[i,j] for i in 0..1, j in 0..2
    B is 3x6: B[j,k] for j in 0..2, k in 0..5
    C is 2x6: C[i,k] for i in 0..1, k in 0..5
    
    Block 0: A[:,0:2] * B[0:2,:] -> C[:,0:2]
    Block 1: A[:,1:3] * B[1:3,:] -> C[:,2:4]
    Block 2: A[:,2:4]... wait, A is only 2x3.
    
    Let me reconsider. A is 2x3, B is 3x6.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Split into 3 column-pairs of B (each 3x2):
    Block 0: columns 0,1 of B -> A * B[:,0:2] contributes to C[:,0:2]
    Block 1: columns 2,3 of B -> A * B[:,2:4] contributes to C[:,2:4]
    Block 2: columns 4,5 of B -> A * B[:,4:6] contributes to C[:,4:6]
    
    For each block b (b=0,1,2):
    A * B[:, 2b:2b+2] = sum_j A[i,j] * B[j, 2b+t] for t=0,1
    
    Split A into A_L (2x2, columns 0,1) and A_R (2x1, column 2)
    Split B_block into B_T (2x2, rows 0,1) and B_B (1x2, row 2)
    
    C_block = A_L * B_T + A_R * B_B
    
    A_L * B_T: 2x2 * 2x2, rank 7 (Strassen)
    A_R * B_B: 2x1 * 1x2, rank 4
    Total per block: 11
    Total: 33
    """
    n, m, p = DIMENSIONS
    R = 33
    U = [[0.0] * R for _ in range(n * m)]  # 6x33
    V = [[0.0] * R for _ in range(m * p)]  # 18x33
    W = [[0.0] * R for _ in range(p * n)]  # 12x33
    
    U_s, V_s, W_s = _strassen_2x2()
    U_o, V_o, W_o = _outer_2x1_1x2()
    
    for b in range(3):
        base = b * 11
        # Strassen part: A_L (cols 0,1) * B_T (rows 0,1 of block)
        # A_L rows: i*m + j for i in 0,1; j in 0,1 -> indices 0,1,2,3
        # B_T rows: j*p + (2b+t) for j in 0,1; t in 0,1 -> indices 0,1,6,7 for b=0
        # C output: i for i in 0,1; k=2b+t for t in 0,1 -> W index: k*n+i
        
        for s in range(7):
            r = base + s
            for ai in range(4):
                if U_s[ai][s] != 0:
                    # ai maps to A_L row
                    a_i = ai // 2
                    a_j = ai % 2
                    U[a_i * m + a_j][r] += U_s[ai][s]
            for bi in range(4):
                if V_s[bi][s] != 0:
                    # bi maps to B_T row
                    b_j = bi // 2  # row 0 or 1
                    b_k = bi % 2   # col 0 or 1
                    v_idx = b_j * p + (2*b + b_k)
                    V[v_idx][r] += V_s[bi][s]
            for ci in range(4):
                if W_s[ci][s] != 0:
                    # ci maps to C output
                    c_i = ci // 2  # row 0 or 1
                    c_k = ci % 2   # col 0 or 1
                    w_idx = (2*b + c_k) * n + c_i
                    W[w_idx][r] += W_s[ci][s]
        
        # Outer part: A_R (col 2) * B_B (row 2 of block)
        for o in range(4):
            r = base + 7 + o
            for ai in range(2):
                if U_o[ai][o] != 0:
                    # ai maps to A_R row
                    a_i = ai
                    a_j = 2  # column 2
                    U[a_i * m + a_j][r] += U_o[ai][o]
            if V_o[0][o] != 0:
                # B_B is row 2 of the block
                b_j = 2
                b_k = o % 2  # col 0 or 1
                v_idx = b_j * p + (2*b + b_k)
                V[v_idx][r] += V_o[0][o]
            for ci in range(4):
                if W_o[ci][o] != 0:
                    # ci maps to C output
                    c_i = ci // 2
                    c_k = ci % 2
                    w_idx = (2*b + c_k) * n + c_i
                    W[w_idx][r] += W_o[ci][o]
    
    return U, V, W

def construct():
    """
    Return a rank-minimal (or near-minimal) tensor decomposition for 2x3 * 3x6.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    # Try the rank-33 block decomposition
    try:
        U, V, W = _build_rank33()
        return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook rank 36
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
