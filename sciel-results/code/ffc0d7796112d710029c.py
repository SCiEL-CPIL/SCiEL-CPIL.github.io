import time
import numpy as np

def construct():
    n, m, p = 3, 3, 8
    start_time = time.time()
    DEADLINE = 130.0

    U_terms = []
    V_terms = []
    W_terms = []

    def add_term(u_list, v_list, w_list):
        U_terms.append(u_list)
        V_terms.append(v_list)
        W_terms.append(w_list)

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(i, k):
        return k * n + i

    def strassen_block(i_off, k_off):
        """
        Strassen for A[i_off:i_off+2, 0:2] * B[0:2, k_off:k_off+2] -> C[i_off:i_off+2, k_off:k_off+2]
        Here i_off=0, so A[0:2, 0:2] * B[0:2, k_off:k_off+2] -> C[0:2, k_off:k_off+2]
        """
        a = (u_idx(i_off, 0), 1)
        b = (u_idx(i_off, 1), 1)
        c = (u_idx(i_off + 1, 0), 1)
        d = (u_idx(i_off + 1, 1), 1)
        
        e = (v_idx(0, k_off), 1)
        f = (v_idx(0, k_off + 1), 1)
        g = (v_idx(1, k_off), 1)
        h = (v_idx(1, k_off + 1), 1)
        
        C00 = (w_idx(i_off, k_off), 1)
        C01 = (w_idx(i_off, k_off + 1), 1)
        C10 = (w_idx(i_off + 1, k_off), 1)
        C11 = (w_idx(i_off + 1, k_off + 1), 1)
        
        # m1 = (a+d)(e+h)
        add_term([a, d], [e, h], [C00, C11])
        # m2 = (c+d)*e
        add_term([c, d], [e], [C10, (w_idx(i_off + 1, k_off + 1), -1)])
        # m3 = a*(f-h)
        add_term([a], [f, (v_idx(1, k_off + 1), -1)], [C01, C11])
        # m4 = d*(g-e)
        add_term([d], [g, (v_idx(0, k_off), -1)], [C00, C10])
        # m5 = (a+b)*h
        add_term([a, b], [h], [(w_idx(i_off, k_off), -1), C01])
        # m6 = (c-a)*(e+f)
        add_term([c, (u_idx(i_off, 0), -1)], [e, f], [C11])
        # m7 = (b-d)*(g+h)
        add_term([b, (u_idx(i_off + 1, 1), -1)], [g, h], [C00])

    for k0 in range(0, p, 2):
        if time.time() > DEADLINE:
            break
        # Strassen: A[0:2, 0:2] * B[0:2, k0:k0+2] -> C[0:2, k0:k0+2]
        strassen_block(0, k0)
        
        # j in {0,1}, i=2: A[2,0:2] * B[0:2, k0:k0+2] -> C[2, k0:k0+2]
        # Single rank-1 term
        u_part = [(u_idx(2, 0), 1), (u_idx(2, 1), 1)]
        v_part = [(v_idx(0, k0), 1), (v_idx(1, k0), 1), (v_idx(0, k0+1), 1), (v_idx(1, k0+1), 1)]
        w_part = [(w_idx(2, k0), 1), (w_idx(2, k0+1), 1)]
        add_term(u_part, v_part, w_part)
        
        # j=2, i in {0,1,2}: A[0:3, 2] * B[2, k0:k0+2] -> C[0:3, k0:k0+2]
        # Single rank-1 term
        u_part = [(u_idx(0, 2), 1), (u_idx(1, 2), 1), (u_idx(2, 2), 1)]
        v_part = [(v_idx(2, k0), 1), (v_idx(2, k0+1), 1)]
        w_part = [(w_idx(0, k0), 1), (w_idx(1, k0), 1), (w_idx(2, k0), 1),
                  (w_idx(0, k0+1), 1), (w_idx(1, k0+1), 1), (w_idx(2, k0+1), 1)]
        add_term(u_part, v_part, w_part)
    
    rank = len(W_terms)
    
    # Build factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for idx in range(rank):
        for (flat_idx, coeff) in U_terms[idx]:
            U[flat_idx][idx] = coeff
        for (flat_idx, coeff) in V_terms[idx]:
            V[flat_idx][idx] = coeff
        for (flat_idx, coeff) in W_terms[idx]:
            W[flat_idx][idx] = coeff
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    def verify_decomp(Ua, Va, Wa):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_decomp(U_arr, V_arr, W_arr):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
