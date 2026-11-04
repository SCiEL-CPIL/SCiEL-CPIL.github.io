import time
import sys

def construct():
    n, m, p = 4, 4, 7
    
    terms = []
    
    def add_term(u_idx, v_idx, w_idx, u_coeff, v_coeff, w_coeff):
        terms.append((u_idx, v_idx, w_idx, u_coeff, v_coeff, w_coeff))
    
    # [4,4,4] part using Strassen
    # Strassen terms for 2x2x2
    strassen = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
    ]
    
    for A_comb, B_comb in strassen:
        L_A = {}
        for I, J, c in A_comb:
            L_A[(I,J)] = L_A.get((I,J), 0) + c
        L_B = {}
        for J, K, c in B_comb:
            L_B[(J,K)] = L_B.get((J,K), 0) + c
        
        for i_b in range(2):
            for j_b in range(2):
                for k_b in range(2):
                    if (i_b, j_b) not in L_A or (j_b, k_b) not in L_B:
                        continue
                    uc = L_A[(i_b, j_b)]
                    vc = L_B[(j_b, k_b)]
                    for ii in range(2):
                        for jj in range(2):
                            for jj2 in range(2):
                                for kk in range(2):
                                    gi = i_b*2 + ii
                                    gj = j_b*2 + jj
                                    gj2 = j_b*2 + jj2
                                    gk = k_b*2 + kk
                                    add_term(gi*m + gj, gj2*p + gk, gk*n + gi, uc, vc, 1)
    
    # [4,4,2] trivial
    for i in range(4):
        for j in range(4):
            for k in [4, 5]:
                add_term(i*m + j, j*p + k, k*n + i, 1, 1, 1)
    
    # [4,4,1] trivial
    for i in range(4):
        for j in range(4):
            k = 6
            add_term(i*m + j, j*p + k, k*n + i, 1, 1, 1)
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t, (u_idx, v_idx, w_idx, u_coeff, v_coeff, w_coeff) in enumerate(terms):
        U[u_idx][t] = u_coeff
        V[v_idx][t] = v_coeff
        W[w_idx][t] = w_coeff
    
    return {"U": U, "V": V, "W": W}
