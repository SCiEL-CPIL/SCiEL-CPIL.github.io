import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # We use a block decomposition:
    # Split m=5 into 2 and 3.
    # Split p=5 into 2 and 3.
    # Blocks:
    # 1. 2x2x2 (Strassen, rank 7)
    # 2. 2x3x2 (Schoolbook, rank 12)
    # 3. 2x2x3 (Schoolbook, rank 12)
    # 4. 2x3x3 (Schoolbook, rank 18)
    # Total rank = 7 + 12 + 12 + 18 = 49.
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k, u_val=1, v_val=1, w_val=1):
        U_rows.append([0] * 49) # Placeholder, we will build lists of tuples
        # Actually, let's just collect (index, value) pairs for each factor
        pass

    # Let's use lists of (index, value) for each factor to build sparse-like rows
    U_data = [] # List of (row_idx, col_idx, val)
    V_data = []
    W_data = []
    
    col_offset = 0
    
    # Block 1: 2x2x2 (i in 0..1, j in 0..1, k in 0..1)
    # Strassen
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)B00
    # m3 = A00(B01-B11)
    # m4 = A11(B10-B00)
    # m5 = (A00+A01)B11
    # m6 = (A10-A00)(B00+B01)
    # m7 = (A01-A11)(B10+B11)
    
    # We need to map local i,j,k to global indices.
    # Global U index: i*m + j
    # Global V index: j*p + k
    # Global W index: k*n + i
    
    def add_term(u_terms, v_terms, w_terms):
        """
        u_terms: list of (global_u_idx, val)
        v_terms: list of (global_v_idx, val)
        w_terms: list of (global_w_idx, val)
        """
        nonlocal col_offset
        for idx, val in u_terms:
            U_data.append((idx, col_offset, val))
        for idx, val in v_terms:
            V_data.append((idx, col_offset, val))
        for idx, val in w_terms:
            W_data.append((idx, col_offset, val))
        col_offset += 1

    # Strassen terms for Block 1
    # m1
    add_term(
        [(0*5+0, 1), (1*5+1, 1)], # A00, A11
        [(0*5+0, 1), (1*5+1, 1)], # B00, B11
        [(0*2+0, 1), (1*2+1, 1)]  # C00, C11
    )
    # m2
    add_term(
        [(1*5+0, 1), (1*5+1, 1)], # A10, A11
        [(0*5+0, 1)],             # B00
        [(0*2+1, 1), (1*2+1, -1)] # C10, C11
    )
    # m3
    add_term(
        [(0*5+0, 1)],             # A00
        [(0*5+1, 1), (1*5+1, -1)],# B01, -B11
        [(1*2+0, 1), (1*2+1, 1)]  # C01, C11
    )
    # m4
    add_term(
        [(1*5+1, 1)],             # A11
        [(1*5+0, 1), (0*5+0, -1)],# B10, -B00
        [(0*2+1, 1), (0*2+0, 1)]  # C10, C00
    )
    # m5
    add_term(
        [(0*5+0, 1), (0*5+1, 1)], # A00, A01
        [(1*5+1, 1)],             # B11
        [(1*2+0, 1), (0*2+0, -1)] # C01, C00
    )
    # m6
    add_term(
        [(1*5+0, 1), (0*5+0, -1)],# A10, -A00
        [(0*5+0, 1), (0*5+1, 1)], # B00, B01
        [(1*2+1, 1)]              # C11
    )
    # m7
    add_term(
        [(0*5+1, 1), (1*5+1, -1)],# A01, -A11
        [(1*5+0, 1), (1*5+1, 1)], # B10, B11
        [(0*2+0, 1)]              # C00
    )
    
    # Block 2: 2x3x2 (i in 0..1, j in 2..4, k in 0..1)
    # Schoolbook
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                add_term(
                    [(i*5+j, 1)],
                    [(j*5+k, 1)],
                    [(k*2+i, 1)]
                )

    # Block 3: 2x2x3 (i in 0..1, j in 0..1, k in 2..4)
    # Schoolbook
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                add_term(
                    [(i*5+j, 1)],
                    [(j*5+k, 1)],
                    [(k*2+i, 1)]
                )

    # Block 4: 2x3x3 (i in 0..1, j in 2..4, k in 2..4)
    # Schoolbook
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                add_term(
                    [(i*5+j, 1)],
                    [(j*5+k, 1)],
                    [(k*2+i, 1)]
                )

    R = col_offset
    
    # Build matrices
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    for idx, col, val in U_data:
        U[idx][col] = val
    for idx, col, val in V_data:
        V[idx][col] = val
    for idx, col, val in W_data:
        W[idx][col] = val

    # Verify
    if time.time() > deadline - 10:
        return {"U": U, "V": V, "W": W}

    # Simple verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(R):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    # Fallback to schoolbook if error
                    U = [[0] * 50 for _ in range(n * m)]
                    V = [[0] * 50 for _ in range(m * p)]
                    W = [[0] * 50 for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii*m+jj][col] = 1
                                V[jj*p+kk][col] = 1
                                W[kk*n+ii][col] = 1
                                col += 1
                    return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
