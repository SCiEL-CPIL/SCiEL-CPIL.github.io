import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 42 construction using Strassen blocks
    # n=2, m=4, p=6
    # Split m into two blocks of 2 (s=0,1)
    # Split p into three blocks of 2 (t=0,1,2)
    # Each block is a 2x2x2 multiplication, rank 7
    # Total rank = 2 * 3 * 7 = 42
    
    R = 42
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    # Strassen coefficients for 2x2x2
    # U local indices: i*2 + u
    # V local indices: v*2 + w
    # W local indices: w*2 + i
    # 
    # M1: U(0,0)=1, U(1,1)=1; V(0,0)=1, V(1,1)=1; W(0,0)=1, W(1,1)=1
    # M2: U(1,0)=1, U(1,1)=1; V(0,0)=1; W(0,1)=1
    # M3: U(0,0)=1; V(1,0)=1, V(1,1)=-1; W(1,0)=1
    # M4: U(1,1)=1; V(0,1)=1, V(0,0)=-1; W(0,0)=1, W(0,1)=1
    # M5: U(0,0)=1, U(1,0)=1; V(1,1)=1; W(0,0)=-1, W(1,0)=1
    # M6: U(1,0)=1, U(0,0)=-1; V(0,0)=1, V(0,1)=1; W(1,1)=-1
    # M7: U(0,1)=1, U(1,1)=-1; V(1,0)=1, V(1,1)=1; W(0,0)=-1, W(1,1)=1
    
    strassen_terms = [
        # U: [(idx, coeff)], V: [(idx, coeff)], W: [(idx, coeff)]
        # Local indices: U: i*2+u, V: v*2+w, W: w*2+i
        {
            'U': [(0, 1), (3, 1)], # 00, 11
            'V': [(0, 1), (3, 1)], # 00, 11
            'W': [(0, 1), (3, 1)]  # 00 (w=0,i=0), 11 (w=1,i=1) -> idx 0, 3
        },
        {
            'U': [(2, 1), (3, 1)], # 10, 11
            'V': [(0, 1)],         # 00
            'W': [(1, 1)]          # 01 (w=0,i=1) -> idx 1
        },
        {
            'U': [(0, 1)],         # 00
            'V': [(2, 1), (3, -1)],# 10, 11
            'W': [(2, 1)]          # 10 (w=1,i=0) -> idx 2
        },
        {
            'U': [(3, 1)],         # 11
            'V': [(1, 1), (0, -1)],# 01, 00
            'W': [(0, 1), (1, 1)]  # 00, 01 -> idx 0, 1
        },
        {
            'U': [(0, 1), (2, 1)], # 00, 10
            'V': [(3, 1)],         # 11
            'W': [(0, -1), (2, 1)] # 00, 10 -> idx 0, 2
        },
        {
            'U': [(2, 1), (0, -1)],# 10, 00
            'V': [(0, 1), (1, 1)], # 00, 01
            'W': [(3, -1)]         # 11 -> idx 3
        },
        {
            'U': [(1, 1), (3, -1)],# 01, 11
            'V': [(2, 1), (3, 1)], # 10, 11
            'W': [(0, -1), (3, 1)] # 00, 11 -> idx 0, 3
        }
    ]
    
    col = 0
    for s in range(2): # m blocks
        for t in range(3): # p blocks
            for r in range(7): # strassen terms
                term = strassen_terms[r]
                
                # Fill U
                for local_idx, coeff in term['U']:
                    i = local_idx // 2
                    u = local_idx % 2
                    j_global = 2 * s + u
                    u_idx = i * m + j_global
                    U[u_idx, col] += coeff
                
                # Fill V
                for local_idx, coeff in term['V']:
                    v = local_idx // 2
                    w = local_idx % 2
                    j_global = 2 * s + v
                    k_global = 2 * t + w
                    v_idx = j_global * p + k_global
                    V[v_idx, col] += coeff
                
                # Fill W
                for local_idx, coeff in term['W']:
                    w = local_idx // 2
                    i = local_idx % 2
                    k_global = 2 * t + w
                    w_idx = k_global * n + i
                    W[w_idx, col] += coeff
                
                col += 1
    
    # Verify
    U_t = U.reshape(n, m, R)
    V_t = V.reshape(m, p, R)
    W_t = W.reshape(p, n, R)
    T = np.einsum('ijr,jkr,ki->ijk', U_t, V_t, W_t)
    expected = np.ones((n, m, p))
    
    if not np.allclose(T, expected, atol=1e-9):
        # Fallback to trivial rank 48
        R_fb = n * m * p
        U_fb = np.zeros((n * m, R_fb), dtype=int)
        V_fb = np.zeros((m * p, R_fb), dtype=int)
        W_fb = np.zeros((p * n, R_fb), dtype=int)
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col_fb] = 1
                    V_fb[j * p + k, col_fb] = 1
                    W_fb[k * n + i, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
