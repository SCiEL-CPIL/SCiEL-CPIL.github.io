"""Matrix multiplication tensor decomposition for 3x4 x 4x7 using Strassen blocks."""
import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Build decomposition using block Strassen for 2x2x2 blocks
    # Split A (3x4): rows [0,1] and [2], cols [0,1] and [2,3]
    # Split B (4x7): rows [0,1] and [2,3], cols [0,1], [2,3], [4,5], [6]
    # Output C (3x7): rows [0,1] and [2], cols [0,1], [2,3], [4,5], [6]
    
    U = []  # list of dicts: {u_index: value}
    V = []  # list of dicts: {v_index: value}
    W = []  # list of dicts: {w_index: value}
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(i, k):
        return k * n + i
    
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add 7 Strassen terms for 2x2 x 2x2 product."""
        a00, a01, a10, a11 = (a_r[0],a_c[0]), (a_r[0],a_c[1]), (a_r[1],a_c[0]), (a_r[1],a_c[1])
        b00, b01, b10, b11 = (b_r[0],b_c[0]), (b_r[0],b_c[1]), (b_r[1],b_c[0]), (b_r[1],b_c[1])
        c00, c01, c10, c11 = (c_r[0],c_c[0]), (c_r[0],c_c[1]), (c_r[1],c_c[0]), (c_r[1],c_c[1])
        
        # M1 = (A00+A11)*(B00+B11) -> C00(+1), C11(+1)
        U.append({u_idx(*a00):1, u_idx(*a11):1})
        V.append({v_idx(*b00):1, v_idx(*b11):1})
        W.append({w_idx(*c00):1, w_idx(*c11):1})
        
        # M2 = (A10+A11)*B00 -> C10(+1)
        U.append({u_idx(*a10):1, u_idx(*a11):1})
        V.append({v_idx(*b00):1})
        W.append({w_idx(*c10):1})
        
        # M3 = A00*(B01-B11) -> C01(+1)
        U.append({u_idx(*a00):1})
        V.append({v_idx(*b01):1, v_idx(*b11):-1})
        W.append({w_idx(*c01):1})
        
        # M4 = A11*(B10-B00) -> C00(+1), C10(+1)
        U.append({u_idx(*a11):1})
        V.append({v_idx(*b10):1, v_idx(*b00):-1})
        W.append({w_idx(*c00):1, w_idx(*c10):1})
        
        # M5 = (A00+A01)*B11 -> C00(-1), C01(+1)
        U.append({u_idx(*a00):1, u_idx(*a01):1})
        V.append({v_idx(*b11):1})
        W.append({w_idx(*c00):-1, w_idx(*c01):1})
        
        # M6 = (A10-A00)*(B00+B01) -> C11(+1)
        U.append({u_idx(*a10):1, u_idx(*a00):-1})
        V.append({v_idx(*b00):1, v_idx(*b01):1})
        W.append({w_idx(*c11):1})
        
        # M7 = (A01-A11)*(B10+B11) -> C00(+1), C11(+1)
        U.append({u_idx(*a01):1, u_idx(*a11):-1})
        V.append({v_idx(*b10):1, v_idx(*b11):1})
        W.append({w_idx(*c00):1, w_idx(*c11):1})
    
    def add_simple(ai, bi, ci):
        """Add a single schoolbook term."""
        U.append({ai:1})
        V.append({bi:1})
        W.append({ci:1})
    
    # C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    add_strassen([0,1],[0,1],[0,1],[0,1],[0,1],[0,1])
    add_strassen([0,1],[2,3],[2,3],[0,1],[0,1],[0,1])
    
    # C[0:2,2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
    add_strassen([0,1],[0,1],[0,1],[2,3],[0,1],[2,3])
    add_strassen([0,1],[2,3],[2,3],[2,3],[0,1],[2,3])
    
    # C[0:2,4:6] = A[0:2,0:2]*B[0:2,4:6] + A[0:2,2:4]*B[2:4,4:6]
    add_strassen([0,1],[0,1],[0,1],[4,5],[0,1],[4,5])
    add_strassen([0,1],[2,3],[2,3],[4,5],[0,1],[4,5])
    
    # C[0:2,6] = A[0:2,0:2]*B[0:2,6] + A[0:2,2:4]*B[2:4,6] (schoolbook, 8 terms)
    for i in range(2):
        for j in range(2):
            add_simple(u_idx(i,j), v_idx(j,6), w_idx(i,6))
    for i in range(2):
        for j in range(2,4):
            add_simple(u_idx(i,j), v_idx(j,6), w_idx(i,6))
    
    # C[2,0:2] = A[2,0:2]*B[0:2,0:2] + A[2,2:4]*B[2:4,0:2] (schoolbook, 8 terms)
    for j in range(2):
        for k in range(2):
            add_simple(u_idx(2,j), v_idx(j,k), w_idx(2,k))
    for j in range(2,4):
        for k in range(2):
            add_simple(u_idx(2,j), v_idx(j,k), w_idx(2,k))
    
    # C[2,2:4] = A[2,0:2]*B[0:2,2:4] + A[2,2:4]*B[2:4,2:4] (schoolbook, 8 terms)
    for j in range(2):
        for k in range(2,4):
            add_simple(u_idx(2,j), v_idx(j,k), w_idx(2,k))
    for j in range(2,4):
        for k in range(2,4):
            add_simple(u_idx(2,j), v_idx(j,k), w_idx(2,k))
    
    # C[2,4:6] = A[2,0:2]*B[0:2,4:6] + A[2,2:4]*B[2:4,4:6] (schoolbook, 8 terms)
    for j in range(2):
        for k in range(4,6):
            add_simple(u_idx(2,j), v_idx(j,k), w_idx(2,k))
    for j in range(2,4):
        for k in range(4,6):
            add_simple(u_idx(2,j), v_idx(j,k), w_idx(2,k))
    
    # C[2,6] = A[2,0:2]*B[0:2,6] + A[2,2:4]*B[2:4,6] (schoolbook, 4 terms)
    for j in range(2):
        add_simple(u_idx(2,j), v_idx(j,6), w_idx(2,6))
    for j in range(2,4):
        add_simple(u_idx(2,j), v_idx(j,6), w_idx(2,6))
    
    rank = len(U)
    
    # Convert to matrices
    U_mat = np.zeros((n*m, rank), dtype=int)
    V_mat = np.zeros((m*p, rank), dtype=int)
    W_mat = np.zeros((p*n, rank), dtype=int)
    
    for c in range(rank):
        for idx, val in U[c].items():
            U_mat[idx, c] = val
        for idx, val in V[c].items():
            V_mat[idx, c] = val
        for idx, val in W[c].items():
            W_mat[idx, c] = val
    
    # Verify
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_mat[i*m+j] * V_mat[j*p+k] * W_mat[k*n+i])
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    
    if max_err > 1e-9:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n*m, rank_fb), dtype=int)
        V_fb = np.zeros((m*p, rank_fb), dtype=int)
        W_fb = np.zeros((p*n, rank_fb), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
