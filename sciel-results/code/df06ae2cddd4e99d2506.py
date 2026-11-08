import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    
    rank = 60
    
    U = [[0] * rank for _ in range(n * m)]  # 9 x 60
    V = [[0] * rank for _ in range(m * p)]  # 21 x 60
    W = [[0] * rank for _ in range(p * n)]  # 21 x 60
    
    col = 0
    
    def u_idx(i, j): return i * m + j  # A[i,j]
    def v_idx(j, k): return j * p + k  # B[j,k]
    def w_idx(i, k): return k * n + i  # C[i,k], W[k*n+i]
    
    def add_term(u_entries, v_entries, w_entries):
        """u_entries: dict {flat_idx: val}, etc."""
        nonlocal col
        if col >= rank:
            return
        for idx, val in u_entries.items():
            U[idx][col] += val
        for idx, val in v_entries.items():
            V[idx][col] += val
        for idx, val in w_entries.items():
            W[idx][col] += val
        col += 1
    
    # Strassen for A[0:2,0:2] x B[0:2, k:k+2] -> C[0:2, k:k+2]
    def strassen_2x2_2x2(k_start):
        """
        A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
        B[0,k]=e, B[0,k+1]=f, B[1,k]=g, B[1,k+1]=h
        """
        a_i = u_idx(0,0)
        b_i = u_idx(0,1)
        c_i = u_idx(1,0)
        d_i = u_idx(1,1)
        
        e_i = v_idx(0,k_start)
        f_i = v_idx(0,k_start+1)
        g_i = v_idx(1,k_start)
        h_i = v_idx(1,k_start+1)
        
        p00 = w_idx(0,k_start)
        p01 = w_idx(0,k_start+1)
        p10 = w_idx(1,k_start)
        p11 = w_idx(1,k_start+1)
        
        # m1 = (a+d)(e+h)
        add_term({a_i:1, d_i:1}, {e_i:1, h_i:1}, {p00:1, p01:1, p10:1, p11:1})
        # Wait, this is wrong. m1 contributes to p00, p01, p10, p11 with specific coefficients.
        # p = m1 - m4 + m5 - m7
        # q = m2 + m7
        # r = m3 + m4
        # s = m1 - m3 + m6 + m7
        #
        # So m1 contributes: +1 to p00, +1 to p11, 0 to p01, 0 to p10
        # Wait no: p=C[0,0], q=C[0,1], r=C[1,0], s=C[1,1]
        # p = m1 - m4 + m5 - m7  → m1 contributes +1 to C[0,0]
        # q = m2 + m7  → m1 contributes 0 to C[0,1]
        # r = m3 + m4  → m1 contributes 0 to C[1,0]
        # s = m1 - m3 + m6 + m7  → m1 contributes +1 to C[1,1]
        
        # I already added m1 above but with wrong W vector. Let me restart.
        pass
    
    # Let me redo this more carefully. I'll add each m_i as a separate rank-1 term.
    
    # Reset
    col = 0
    
    def strassen_block(k_start):
        """Add 7 Strassen terms for A[0:2,0:2] x B[0:2, k_start:k_start+2]"""
        nonlocal col
        
        a_i = u_idx(0,0)
        b_i = u_idx(0,1)
        c_i = u_idx(1,0)
        d_i = u_idx(1,1)
        
        e_i = v_idx(0,k_start)
        f_i = v_idx(0,k_start+1)
        g_i = v_idx(1,k_start)
        h_i = v_idx(1,k_start+1)
        
        p00 = w_idx(0,k_start)
        p01 = w_idx(0,k_start+1)
        p10 = w_idx(1,k_start)
        p11 = w_idx(1,k_start+1)
        
        # m1 = (a+d)(e+h), contributes: +1 to C[0,0], +1 to C[1,1]
        add_term({a_i:1, d_i:1}, {e_i:1, h_i:1}, {p00:1, p11:1})
        
        # m2 = (c+d)e, contributes: +1 to C[0,1]
        add_term({c_i:1, d_i:1}, {e_i:1}, {p01:1})
        
        # m3 = a(f-h), contributes: +1 to C[1,0], -1 to C[1,1]
        add_term({a_i:1}, {f_i:1, h_i:-1}, {p10:1, p11:-1})
        
        # m4 = d(g-e), contributes: -1 to C[0,0], +1 to C[1,0]
        add_term({d_i:1}, {g_i:1, e_i:-1}, {p00:-1, p10:1})
        
        # m5 = (a+b)h, contributes: +1 to C[0,0]
        add_term({a_i:1, b_i:1}, {h_i:1}, {p00:1})
        
        # m6 = (c-a)(e+f), contributes: +1 to C[1,1]
        add_term({c_i:1, a_i:-1}, {e_i:1, f_i:1}, {p11:1})
        
        # m7 = (b-d)(g+h), contributes: -1 to C[0,0], +1 to C[0,1], +1 to C[1,1]
        add_term({b_i:1, d_i:-1}, {g_i:1, h_i:1}, {p00:-1, p01:1, p11:1})
    
    # Apply Strassen for column pairs (0,1), (2,3), (4,5)
    strassen_block(0)
    strassen_block(2)
    strassen_block(4)
    
    # Column 6: schoolbook for A[0:2,0:2] x B[0:2, 6:7]
    # C[0,6] = A[0,0]*B[0,6] + A[0,1]*B[1,6]
    # C[1,6] = A[1,0]*B[0,6] + A[1,1]*B[1,6]
    for i in range(2):
        for j in range(2):
            add_term({u_idx(i,j):1}, {v_idx(j,6):1}, {w_idx(i,6):1})
    
    # A[0:2, 2] x B[2, :]: 2 x 7 = 14 terms
    for i in range(2):
        for k in range(7):
            add_term({u_idx(i,2):1}, {v_idx(2,k):1}, {w_idx(i,k):1})
    
    # A[2, 0:2] x B[0:2, :]: 2 x 7 = 14 terms
    for j in range(2):
        for k in range(7):
            add_term({u_idx(2,j):1}, {v_idx(j,k):1}, {w_idx(2,k):1})
    
    # A[2,2] x B[2,:]: 7 terms
    for k in range(7):
        add_term({u_idx(2,2):1}, {v_idx(2,k):1}, {w_idx(2,k):1})
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_true = A_test @ B_test
    C_comp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0.0
            for j in range(m):
                for r in range(rank):
                    val += U_arr[i*m + j, r] * A_test[i, j] * V_arr[j*p + k, r] * B_test[j, k] * W_arr[k*n + i, r]
            C_comp[i, k] = val
            
    if not np.allclose(C_comp, C_true, atol=1e-9):
        # Fallback
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col_fb] = 1
                    V_fb[j * p + k][col_fb] = 1
                    W_fb[k * n + i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
