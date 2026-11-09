import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # New block decomposition:
    # Split m=5 into 4+1, p=5 into 4+1
    # Blocks:
    # 1. 2x4x4: split into four 2x2x2 Strassen blocks (rank 7 each) = 28
    # 2. 2x4x1: schoolbook rank 8
    # 3. 2x1x4: schoolbook rank 8
    # 4. 2x1x1: schoolbook rank 2
    # Total rank = 28 + 8 + 8 + 2 = 46
    
    U_data = [] # List of (row_idx, col_idx, val)
    V_data = []
    W_data = []
    
    col_offset = 0
    
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
    
    def add_strassen_block(i_base, j_base, k_base):
        """Add Strassen decomposition for a 2x2x2 block at given base indices."""
        # i in {i_base, i_base+1}, j in {j_base, j_base+1}, k in {k_base, k_base+1}
        # Local indices: i_l in {0,1}, j_l in {0,1}, k_l in {0,1}
        # Global: i = i_base + i_l, j = j_base + j_l, k = k_base + k_l
        # U index: i*m + j = (i_base+i_l)*m + (j_base+j_l)
        # V index: j*p + k = (j_base+j_l)*p + (k_base+k_l)
        # W index: k*n + i = (k_base+k_l)*n + (i_base+i_l)
        
        def u_idx(i_l, j_l):
            return (i_base + i_l) * m + (j_base + j_l)
        
        def v_idx(j_l, k_l):
            return (j_base + j_l) * p + (k_base + k_l)
        
        def w_idx(k_l, i_l):
            return (k_base + k_l) * n + (i_base + i_l)
        
        # m1 = (A00+A11)(B00+B11)
        add_term(
            [(u_idx(0,0), 1), (u_idx(1,1), 1)],
            [(v_idx(0,0), 1), (v_idx(1,1), 1)],
            [(w_idx(0,0), 1), (w_idx(1,1), 1)]
        )
        # m2 = (A10+A11)B00
        add_term(
            [(u_idx(1,0), 1), (u_idx(1,1), 1)],
            [(v_idx(0,0), 1)],
            [(w_idx(0,1), 1), (w_idx(1,1), -1)]
        )
        # m3 = A00(B01-B11)
        add_term(
            [(u_idx(0,0), 1)],
            [(v_idx(0,1), 1), (v_idx(1,1), -1)],
            [(w_idx(1,0), 1), (w_idx(1,1), 1)]
        )
        # m4 = A11(B10-B00)
        add_term(
            [(u_idx(1,1), 1)],
            [(v_idx(1,0), 1), (v_idx(0,0), -1)],
            [(w_idx(0,1), 1), (w_idx(0,0), 1)]
        )
        # m5 = (A00+A01)B11
        add_term(
            [(u_idx(0,0), 1), (u_idx(0,1), 1)],
            [(v_idx(1,1), 1)],
            [(w_idx(1,0), 1), (w_idx(0,0), -1)]
        )
        # m6 = (A10-A00)(B00+B01)
        add_term(
            [(u_idx(1,0), 1), (u_idx(0,0), -1)],
            [(v_idx(0,0), 1), (v_idx(0,1), 1)],
            [(w_idx(1,1), 1)]
        )
        # m7 = (A01-A11)(B10+B11)
        add_term(
            [(u_idx(0,1), 1), (u_idx(1,1), -1)],
            [(v_idx(1,0), 1), (v_idx(1,1), 1)],
            [(w_idx(0,0), 1)]
        )
    
    # Block 1: 2x4x4 (i in 0..1, j in 0..3, k in 0..3)
    # Split into four 2x2x2 blocks
    add_strassen_block(0, 0, 0)  # j in 0..1, k in 0..1
    add_strassen_block(0, 2, 0)  # j in 2..3, k in 0..1
    add_strassen_block(0, 0, 2)  # j in 0..1, k in 2..3
    add_strassen_block(0, 2, 2)  # j in 2..3, k in 2..3
    
    # Block 2: 2x4x1 (i in 0..1, j in 0..3, k in 4)
    # Schoolbook
    for i in range(2):
        for j in range(4):
            add_term(
                [(i*m+j, 1)],
                [(j*p+4, 1)],
                [(4*n+i, 1)]
            )
    
    # Block 3: 2x1x4 (i in 0..1, j in 4, k in 0..3)
    # Schoolbook
    for i in range(2):
        for k in range(4):
            add_term(
                [(i*m+4, 1)],
                [(4*p+k, 1)],
                [(k*n+i, 1)]
            )
    
    # Block 4: 2x1x1 (i in 0..1, j in 4, k in 4)
    # Schoolbook
    for i in range(2):
        add_term(
            [(i*m+4, 1)],
            [(4*p+4, 1)],
            [(4*n+i, 1)]
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
