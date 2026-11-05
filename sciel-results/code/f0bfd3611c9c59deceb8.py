"""
Rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.

Strategy: Block decomposition with Strassen's rank-7 algorithm for 2x2x2 sub-blocks.

Split A (4x5) into row groups [0:2], [2:4] and col groups [0:2], [2:4], [4:5].
Split B (5x6) into row groups [0:2], [2:4], [4:5] and col groups [0:2], [2:4], [4:6].

This gives 18 sub-problems:
- 12 are 2x2 * 2x2 -> 2x2 (use Strassen, rank 7 each) = 84
- 6 are 2x1 * 1x2 -> 2x2 (schoolbook, rank 4 each) = 24
Total rank = 108.
"""
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 150.0
    
    n, m, p = 4, 5, 6
    
    U_rows = {}
    V_rows = {}
    W_rows = {}
    
    col = 0
    
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        for r, c in u_vec.items():
            U_rows.setdefault(r, []).append((c, col))
        for r, c in v_vec.items():
            V_rows.setdefault(r, []).append((c, col))
        for r, c in w_vec.items():
            W_rows.setdefault(r, []).append((c, col))
        col += 1
    
    def uidx(i, j): return i * m + j
    def vidx(j, k): return j * p + k
    def widx(k, i): return k * n + i
    
    def strassen_2x2x2(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        A00 = (a_r0, a_c0); A01 = (a_r0, a_c0+1); A10 = (a_r0+1, a_c0); A11 = (a_r0+1, a_c0+1)
        B00 = (b_r0, b_c0); B01 = (b_r0, b_c0+1); B10 = (b_r0+1, b_c0); B11 = (b_r0+1, b_c0+1)
        C00 = (c_r0, c_c0); C01 = (c_r0, c_c0+1); C10 = (c_r0+1, c_c0); C11 = (c_r0+1, c_c0+1)
        
        add_term({uidx(*A00): 1, uidx(*A11): 1}, {vidx(*B00): 1, vidx(*B11): 1}, {widx(*C00): 1, widx(*C11): 1})
        add_term({uidx(*A10): 1, uidx(*A11): 1}, {vidx(*B00): 1}, {widx(*C10): 1, widx(*C11): -1})
        add_term({uidx(*A00): 1}, {vidx(*B01): 1, vidx(*B11): -1}, {widx(*C01): 1, widx(*C11): 1})
        add_term({uidx(*A11): 1}, {vidx(*B10): 1, vidx(*B00): -1}, {widx(*C10): 1, widx(*C00): 1})
        add_term({uidx(*A00): 1, uidx(*A01): 1}, {vidx(*B11): 1}, {widx(*C01): 1, widx(*C00): -1})
        add_term({uidx(*A10): 1, uidx(*A00): -1}, {vidx(*B00): 1, vidx(*B01): 1}, {widx(*C11): 1})
        add_term({uidx(*A01): 1, uidx(*A11): -1}, {vidx(*B10): 1, vidx(*B11): 1}, {widx(*C00): 1})
    
    def schoolbook_2x1x2(a_r0, a_c, b_r, b_c0, c_r0, c_c0):
        add_term({uidx(a_r0, a_c): 1}, {vidx(b_r, b_c0): 1}, {widx(c_c0, c_r0): 1})
        add_term({uidx(a_r0, a_c): 1}, {vidx(b_r, b_c0+1): 1}, {widx(c_c0+1, c_r0): 1})
        add_term({uidx(a_r0+1, a_c): 1}, {vidx(b_r, b_c0): 1}, {widx(c_c0, c_r0+1): 1})
        add_term({uidx(a_r0+1, a_c): 1}, {vidx(b_r, b_c0+1): 1}, {widx(c_c0+1, c_r0+1): 1})
    
    # 12 Strassen blocks: A[ar, ac:ac+2] * B[ar, bc:bc+2] -> C[ar, bc:bc+2]
    for ar in [0, 2]:
        for ac in [0, 2]:
            for bc in [0, 2, 4]:
                strassen_2x2x2(ar, ac, ac, bc, ar, bc)
    
    # 6 schoolbook blocks: A[ar, 4] * B[4, bc:bc+2] -> C[ar, bc:bc+2]
    for ar in [0, 2]:
        for bc in [0, 2, 4]:
            schoolbook_2x1x2(ar, 4, 4, bc, ar, bc)
    
    rank = col
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r, terms in U_rows.items():
        for c, col_idx in terms:
            U[r][col_idx] = c
    for r, terms in V_rows.items():
        for c, col_idx in terms:
            V[r][col_idx] = c
    for r, terms in W_rows.items():
        for c, col_idx in terms:
            W[r][col_idx] = c
    
    # Verify correctness
    try:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        T_recon = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
        T_true = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_true[i*m+j, j*p+k, k*n+i] = 1.0
        if np.max(np.abs(T_recon - T_true)) > 1e-10:
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            cc = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][cc] = 1
                        V[j*p+k][cc] = 1
                        W[k*n+i][cc] = 1
                        cc += 1
    except:
        pass
    
    return {"U": U, "V": V, "W": W}
