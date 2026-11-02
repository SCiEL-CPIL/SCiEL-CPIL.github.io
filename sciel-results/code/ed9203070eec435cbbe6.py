"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Index helpers
    def u_idx(i, j): return i * m + j  # A[i,j] -> index in U
    def v_idx(j, k): return j * p + k  # B[j,k] -> index in V
    def w_idx(k, i): return k * n + i  # C[k,i] -> index in W (trace convention W[k*n+i])
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def make_u(indices):
        vec = [0.0] * (n * m)
        for idx, val in indices:
            vec[idx] += val
        return vec
    
    def make_v(indices):
        vec = [0.0] * (m * p)
        for idx, val in indices:
            vec[idx] += val
        return vec
    
    def make_w(indices):
        vec = [0.0] * (p * n)
        for idx, val in indices:
            vec[idx] += val
        return vec
    
    # Strassen for 2x2 * 2x2:
    def strassen_2x2_2x2(ai, aj, bj, bk):
        """Add 7 Strassen terms for A[ai:ai+2, aj:aj+2] * B[bj:bj+2, bk:bk+2] -> C[ai:ai+2, bk:bk+2]"""
        # M1 = (a+d)(e+h)
        u = make_u([(u_idx(ai,aj),1), (u_idx(ai+1,aj+1),1)])
        v = make_v([(v_idx(bj,bk),1), (v_idx(bj+1,bk+1),1)])
        w = make_w([(w_idx(bk,ai),1), (w_idx(bk+1,ai+1),1)])
        add_term(u, v, w)
        
        # M2 = (c+d)e
        u = make_u([(u_idx(ai+1,aj),1), (u_idx(ai+1,aj+1),1)])
        v = make_v([(v_idx(bj,bk),1)])
        w = make_w([(w_idx(bk,ai+1),1), (w_idx(bk+1,ai+1),-1)])
        add_term(u, v, w)
        
        # M3 = a(f-h)
        u = make_u([(u_idx(ai,aj),1)])
        v = make_v([(v_idx(bj,bk+1),1), (v_idx(bj+1,bk+1),-1)])
        w = make_w([(w_idx(bk+1,ai),1), (w_idx(bk+1,ai+1),1)])
        add_term(u, v, w)
        
        # M4 = d(g-e)
        u = make_u([(u_idx(ai+1,aj+1),1)])
        v = make_v([(v_idx(bj+1,bk),1), (v_idx(bj,bk),-1)])
        w = make_w([(w_idx(bk,ai),1), (w_idx(bk,ai+1),1)])
        add_term(u, v, w)
        
        # M5 = (a+b)h
        u = make_u([(u_idx(ai,aj),1), (u_idx(ai,aj+1),1)])
        v = make_v([(v_idx(bj+1,bk+1),1)])
        w = make_w([(w_idx(bk,ai),-1), (w_idx(bk+1,ai),1)])
        add_term(u, v, w)
        
        # M6 = (c-a)(e+f)
        u = make_u([(u_idx(ai+1,aj),1), (u_idx(ai,aj),-1)])
        v = make_v([(v_idx(bj,bk),1), (v_idx(bj,bk+1),1)])
        w = make_w([(w_idx(bk+1,ai+1),1)])
        add_term(u, v, w)
        
        # M7 = (b-d)(g+h)
        u = make_u([(u_idx(ai,aj+1),1), (u_idx(ai+1,aj+1),-1)])
        v = make_v([(v_idx(bj+1,bk),1), (v_idx(bj+1,bk+1),1)])
        w = make_w([(w_idx(bk,ai),1)])
        add_term(u, v, w)
    
    # Schoolbook for 2x2 * 2x1:
    def schoolbook_2x2_2x1(ai, aj, bj, bk):
        """Add 2 terms for A[ai:ai+2, aj:aj+2] * B[bj:bj+2, bk] -> C[ai:ai+2, bk]"""
        # C[ai,bk] = A[ai,aj]*B[bj,bk] + A[ai,aj+1]*B[bj+1,bk]
        # C[ai+1,bk] = A[ai+1,aj]*B[bj,bk] + A[ai+1,aj+1]*B[bj+1,bk]
        
        # Term 1: (A[ai,aj]+A[ai+1,aj]) * B[bj,bk] -> C[ai,bk] += A[ai,aj]*B[bj,bk], C[ai+1,bk] += A[ai+1,aj]*B[bj,bk]
        u1 = make_u([(u_idx(ai,aj),1), (u_idx(ai+1,aj),1)])
        v1 = make_v([(v_idx(bj,bk),1)])
        w1 = make_w([(w_idx(bk,ai),1), (w_idx(bk,ai+1),1)])
        add_term(u1, v1, w1)
        
        # Term 2: (A[ai,aj+1]+A[ai+1,aj+1]) * B[bj+1,bk] -> C[ai,bk] += A[ai,aj+1]*B[bj+1,bk], C[ai+1,bk] += A[ai+1,aj+1]*B[bj+1,bk]
        u2 = make_u([(u_idx(ai,aj+1),1), (u_idx(ai+1,aj+1),1)])
        v2 = make_v([(v_idx(bj+1,bk),1)])
        w2 = make_w([(w_idx(bk,ai),1), (w_idx(bk,ai+1),1)])
        add_term(u2, v2, w2)
    
    # Build the full decomposition:
    # A is 4x4, split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]
    # B is 4x5, split into blocks:
    # B = [[B00, B01, B02], [B10, B11, B12]]
    # C = [[C00, C01, C02], [C10, C11, C12]]
    
    # C00 = A00*B00 + A01*B10
    strassen_2x2_2x2(0, 0, 0, 0)  # A00 * B00
    strassen_2x2_2x2(0, 2, 2, 0)  # A01 * B10
    
    # C01 = A00*B01 + A01*B11
    strassen_2x2_2x2(0, 0, 0, 2)  # A00 * B01
    strassen_2x2_2x2(0, 2, 2, 2)  # A01 * B11
    
    # C02 = A00*B02 + A01*B12
    schoolbook_2x2_2x1(0, 0, 0, 4)  # A00 * B02
    schoolbook_2x2_2x1(0, 2, 2, 4)  # A01 * B12
    
    # C10 = A10*B00 + A11*B10
    strassen_2x2_2x2(2, 0, 0, 0)  # A10 * B00
    strassen_2x2_2x2(2, 2, 2, 0)  # A11 * B10
    
    # C11 = A10*B01 + A11*B11
    strassen_2x2_2x2(2, 0, 0, 2)  # A10 * B01
    strassen_2x2_2x2(2, 2, 2, 2)  # A11 * B11
    
    # C12 = A10*B02 + A11*B12
    schoolbook_2x2_2x1(2, 0, 0, 4)  # A10 * B02
    schoolbook_2x2_2x1(2, 2, 2, 4)  # A11 * B12
    
    # Stack into matrices
    rank = len(U_cols)
    U = np.array(U_cols, dtype=float).T  # shape (n*m, rank)
    V = np.array(V_cols, dtype=float).T  # shape (m*p, rank)
    W = np.array(W_cols, dtype=float).T  # shape (p*n, rank)
    
    # Verify
    def verify(U, V, W):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False, (i, j, k, val)
        return True, None
    
    valid, err = verify(U, V, W)
    
    if not valid:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n*m, rank_fb))
        V_fb = np.zeros((m*p, rank_fb))
        W_fb = np.zeros((p*n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        U, V, W = U_fb, V_fb, W_fb
        rank = rank_fb
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
