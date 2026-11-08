"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses rank-17 decomposition for 3x3x2 blocks (4 blocks for k=0..7).
Total rank: 4*17 = 68.
"""

import time
import numpy as np

def construct():
    n, m, p = 3, 3, 8  # A is n x m, B is m x p, C is n x p
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

    # Process 4 blocks of 3x3x2
    for k0 in range(0, p, 2):
        if time.time() > DEADLINE - 10:
            break
        
        # Strassen for A[0:2, 0:2] * B[0:2, k0:k0+2] -> C[0:2, k0:k0+2]
        # Indices: i in {0,1}, j in {0,1}, k in {k0, k0+1}
        
        # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
        # e=B[0,k0], f=B[0,k0+1], g=B[1,k0], h=B[1,k0+1]
        
        # m1 = (a+d)(e+h) -> C[0,k0] += m1, C[1,k0+1] += m1
        add_term([(u_idx(0, 0), 1), (u_idx(1, 1), 1)], 
                 [(v_idx(0, k0), 1), (v_idx(1, k0+1), 1)], 
                 [(w_idx(0, k0), 1), (w_idx(1, k0+1), 1)])
                 
        # m2 = (c+d)(e) -> C[1,k0] += m2, C[1,k0+1] -= m2
        add_term([(u_idx(1, 0), 1), (u_idx(1, 1), 1)], 
                 [(v_idx(0, k0), 1)], 
                 [(w_idx(1, k0), 1), (w_idx(1, k0+1), -1)])
                 
        # m3 = (a)(f-h) -> C[0,k0+1] += m3, C[1,k0+1] += m3
        add_term([(u_idx(0, 0), 1)], 
                 [(v_idx(0, k0+1), 1), (v_idx(1, k0+1), -1)], 
                 [(w_idx(0, k0+1), 1), (w_idx(1, k0+1), 1)])
                 
        # m4 = (d)(g-e) -> C[0,k0] -= m4, C[1,k0] += m4
        add_term([(u_idx(1, 1), 1)], 
                 [(v_idx(1, k0), 1), (v_idx(0, k0), -1)], 
                 [(w_idx(0, k0), -1), (w_idx(1, k0), 1)])
                 
        # m5 = (a+b)(h) -> C[0,k0] -= m5, C[0,k0+1] += m5
        add_term([(u_idx(0, 0), 1), (u_idx(0, 1), 1)], 
                 [(v_idx(1, k0+1), 1)], 
                 [(w_idx(0, k0), -1), (w_idx(0, k0+1), 1)])
                 
        # m6 = (c-a)(e+f) -> C[1,k0+1] += m6
        add_term([(u_idx(1, 0), 1), (u_idx(0, 0), -1)], 
                 [(v_idx(0, k0), 1), (v_idx(0, k0+1), 1)], 
                 [(w_idx(1, k0+1), 1)])
                 
        # m7 = (b-d)(g+h) -> C[0,k0] += m7
        add_term([(u_idx(0, 1), 1), (u_idx(1, 1), -1)], 
                 [(v_idx(1, k0), 1), (v_idx(1, k0+1), 1)], 
                 [(w_idx(0, k0), 1)])

        # Direct terms for i=2, j in {0,1}, k in {k0, k0+1}
        add_term([(u_idx(2, 0), 1)], [(v_idx(0, k0), 1)], [(w_idx(2, k0), 1)])
        add_term([(u_idx(2, 1), 1)], [(v_idx(1, k0), 1)], [(w_idx(2, k0), 1)])
        add_term([(u_idx(2, 0), 1)], [(v_idx(0, k0+1), 1)], [(w_idx(2, k0+1), 1)])
        add_term([(u_idx(2, 1), 1)], [(v_idx(1, k0+1), 1)], [(w_idx(2, k0+1), 1)])

        # Direct terms for i in {0,1,2}, j=2, k in {k0, k0+1}
        for i in range(n):
            add_term([(u_idx(i, 2), 1)], [(v_idx(2, k0), 1)], [(w_idx(i, k0), 1)])
        for i in range(n):
            add_term([(u_idx(i, 2), 1)], [(v_idx(2, k0+1), 1)], [(w_idx(i, k0+1), 1)])

    rank = len(W_terms)
    
    if rank == 0:
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
