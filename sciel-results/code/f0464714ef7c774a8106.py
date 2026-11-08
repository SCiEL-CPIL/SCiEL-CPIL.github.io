import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen vectors for 2x2x2
    # u, v, w are length 4
    s_vecs = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, 0])  # M7
    ]
    
    # Terms for 4x4x4
    # (A_blocks, B_blocks, C_blocks)
    # Block (br, bc) is 2x2 at rows 2br..2br+1, cols 2bc..2bc+1
    terms_4x4 = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_vec(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
        
    # Helper to map local 2x2 index to global
    def map_u(br, bc, lr, lc):
        gr = 2*br + lr
        gc = 2*bc + lc
        return gr * m + gc
        
    def map_v(br, bc, lr, lc, col_off):
        gr = 2*br + lr
        gc = col_off + 2*bc + lc
        return gr * p + gc
        
    def map_w(br, bc, lr, lc, col_off):
        gr = 2*br + lr
        gc = col_off + 2*bc + lc
        return gc * n + gr
        
    # Process first 4 columns (4x4x4)
    for term in terms_4x4:
        a_b, b_b, c_b = term
        for u_s, v_s, w_s in s_vecs:
            u_g = [0.0] * (n*m)
            v_g = [0.0] * (m*p)
            w_g = [0.0] * (p*n)
            
            for br, bc, s in a_b:
                for lr in range(2):
                    for lc in range(2):
                        val = u_s[lr*2+lc]
                        if val != 0:
                            idx = map_u(br, bc, lr, lc)
                            u_g[idx] += s * val
                            
            for br, bc, s in b_b:
                for lr in range(2):
                    for lc in range(2):
                        val = v_s[lr*2+lc]
                        if val != 0:
                            idx = map_v(br, bc, lr, lc, 0)
                            v_g[idx] += s * val
                            
            for br, bc, s in c_b:
                for lr in range(2):
                    for lc in range(2):
                        val = w_s[lr*2+lc]
                        if val != 0:
                            idx = map_w(br, bc, lr, lc, 0)
                            w_g[idx] += s * val
                            
            add_vec(u_g, v_g, w_g)
            
    # Process next 4 columns (4x4x4)
    for term in terms_4x4:
        a_b, b_b, c_b = term
        for u_s, v_s, w_s in s_vecs:
            u_g = [0.0] * (n*m)
            v_g = [0.0] * (m*p)
            w_g = [0.0] * (p*n)
            
            for br, bc, s in a_b:
                for lr in range(2):
                    for lc in range(2):
                        val = u_s[lr*2+lc]
                        if val != 0:
                            idx = map_u(br, bc, lr, lc)
                            u_g[idx] += s * val
                            
            for br, bc, s in b_b:
                for lr in range(2):
                    for lc in range(2):
                        val = v_s[lr*2+lc]
                        if val != 0:
                            idx = map_v(br, bc, lr, lc, 4)
                            v_g[idx] += s * val
                            
            for br, bc, s in c_b:
                for lr in range(2):
                    for lc in range(2):
                        val = w_s[lr*2+lc]
                        if val != 0:
                            idx = map_w(br, bc, lr, lc, 4)
                            w_g[idx] += s * val
                            
            add_vec(u_g, v_g, w_g)
            
    # Process last 1 column (4x4x1)
    # Standard algorithm: 16 terms
    for i in range(n):
        for j in range(m):
            u_g = [0.0] * (n*m)
            v_g = [0.0] * (m*p)
            w_g = [0.0] * (p*n)
            
            u_g[i*m + j] = 1.0
            v_g[j*p + 8] = 1.0
            w_g[8*n + i] = 1.0
            
            add_vec(u_g, v_g, w_g)
            
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return {"U": U, "V": V, "W": W}
