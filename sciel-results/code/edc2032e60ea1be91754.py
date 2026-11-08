import numpy as np
import time

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    def u_idx(i, j):
        return i * 4 + j
    
    def v_idx(j, k):
        return j * 4 + k
    
    def w_idx(k, i):
        return k * 4 + i
    
    # Recursive Strassen: 4x4 -> 7 block products of 2x2, each with 7 scalar mults
    # Block (bi,bj) of A: rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    
    def a_key(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    def b_key(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    def c_key(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    # 7 block-level Strassen terms
    # Each term: L (2x2 of A linear combos), R (2x2 of B linear combos), contrib (output mapping)
    
    terms = []
    
    # Term 0: P1 = (A11 + A22)(B11 + B22)
    L0 = [[{} for _ in range(2)] for _ in range(2)]
    R0 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L0[r][c] = {a_key(0,0,r,c): 1, a_key(1,1,r,c): 1}
            R0[r][c] = {b_key(0,0,r,c): 1, b_key(1,1,r,c): 1}
    contrib0 = {}
    for r in range(2):
        for c in range(2):
            contrib0[(r,c)] = [(c_key(0,0,r,c), 1), (c_key(1,1,r,c), 1)]
    terms.append((L0, R0, contrib0))
    
    # Term 1: P2 = (A21 + A22)B11
    L1 = [[{} for _ in range(2)] for _ in range(2)]
    R1 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L1[r][c] = {a_key(1,0,r,c): 1, a_key(1,1,r,c): 1}
            R1[r][c] = {b_key(0,0,r,c): 1}
    contrib1 = {}
    for r in range(2):
        for c in range(2):
            contrib1[(r,c)] = [(c_key(1,0,r,c), 1), (c_key(1,1,r,c), -1)]
    terms.append((L1, R1, contrib1))
    
    # Term 2: P3 = A11(B12 - B22)
    L2 = [[{} for _ in range(2)] for _ in range(2)]
    R2 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L2[r][c] = {a_key(0,0,r,c): 1}
            R2[r][c] = {b_key(0,1,r,c): 1, b_key(1,1,r,c): -1}
    contrib2 = {}
    for r in range(2):
        for c in range(2):
            contrib2[(r,c)] = [(c_key(0,1,r,c), 1), (c_key(1,1,r,c), 1)]
    terms.append((L2, R2, contrib2))
    
    # Term 3: P4 = A22(B21 - B11)
    L3 = [[{} for _ in range(2)] for _ in range(2)]
    R3 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L3[r][c] = {a_key(1,1,r,c): 1}
            R3[r][c] = {b_key(1,0,r,c): 1, b_key(0,0,r,c): -1}
    contrib3 = {}
    for r in range(2):
        for c in range(2):
            contrib3[(r,c)] = [(c_key(0,0,r,c), 1), (c_key(1,0,r,c), 1)]
    terms.append((L3, R3, contrib3))
    
    # Term 4: P5 = (A11 + A12)B22
    L4 = [[{} for _ in range(2)] for _ in range(2)]
    R4 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L4[r][c] = {a_key(0,0,r,c): 1, a_key(0,1,r,c): 1}
            R4[r][c] = {b_key(1,1,r,c): 1}
    contrib4 = {}
    for r in range(2):
        for c in range(2):
            contrib4[(r,c)] = [(c_key(0,0,r,c), -1), (c_key(0,1,r,c), 1)]
    terms.append((L4, R4, contrib4))
    
    # Term 5: P6 = (A21 - A11)(B11 + B12)
    L5 = [[{} for _ in range(2)] for _ in range(2)]
    R5 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L5[r][c] = {a_key(1,0,r,c): 1, a_key(0,0,r,c): -1}
            R5[r][c] = {b_key(0,0,r,c): 1, b_key(0,1,r,c): 1}
    contrib5 = {}
    for r in range(2):
        for c in range(2):
            contrib5[(r,c)] = [(c_key(1,1,r,c), 1)]
    terms.append((L5, R5, contrib5))
    
    # Term 6: P7 = (A12 - A22)(B21 + B22)
    L6 = [[{} for _ in range(2)] for _ in range(2)]
    R6 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L6[r][c] = {a_key(0,1,r,c): 1, a_key(1,1,r,c): -1}
            R6[r][c] = {b_key(1,0,r,c): 1, b_key(1,1,r,c): 1}
    contrib6 = {}
    for r in range(2):
        for c in range(2):
            contrib6[(r,c)] = [(c_key(0,0,r,c), 1)]
    terms.append((L6, R6, contrib6))
    
    # Strassen sub-product contributions to 2x2 result
    # M1 -> (0,0):+1, (1,1):+1
    # M2 -> (1,0):+1, (1,1):-1
    # M3 -> (0,1):+1, (1,1):+1
    # M4 -> (0,0):+1, (1,0):+1
    # M5 -> (0,0):-1, (0,1):+1
    # M6 -> (1,1):+1
    # M7 -> (0,0):+1
    sub_contrib = [
        [((0,0),1), ((1,1),1)],
        [((1,0),1), ((1,1),-1)],
        [((0,1),1), ((1,1),1)],
        [((0,0),1), ((1,0),1)],
        [((0,0),-1), ((0,1),1)],
        [((1,1),1)],
        [((0,0),1)]
    ]
    
    for t in range(7):
        L_t, R_t, contrib_t = terms[t]
        a, b, c, d = L_t[0][0], L_t[0][1], L_t[1][0], L_t[1][1]
        e, f, g, h = R_t[0][0], R_t[0][1], R_t[1][0], R_t[1][1]
        
        # 7 sub-products
        def add_dicts(x, y, sign=1):
            d = dict(x)
            for k, v in y.items():
                d[k] = d.get(k, 0) + sign * v
            return d
        
        sub_products = [
            (add_dicts(a, d), add_dicts(e, h)),           # M1
            (add_dicts(c, d), dict(e)),                    # M2
            (dict(a), add_dicts(f, h, -1)),               # M3
            (dict(d), add_dicts(g, e, -1)),               # M4
            (add_dicts(a, b), dict(h)),                   # M5
            (add_dicts(c, a, -1), add_dicts(e, f)),       # M6
            (add_dicts(b, d, -1), add_dicts(g, h))        # M7
        ]
        
        for k in range(7):
            x_k, y_k = sub_products[k]
            rank_idx = t * 7 + k
            
            for (row, col), coeff in x_k.items():
                U[u_idx(row, col)][rank_idx] += coeff
            
            for (row, col), coeff in y_k.items():
                V[v_idx(row, col)][rank_idx] += coeff
            
            for (pr, pc), sub_coeff in sub_contrib[k]:
                if (pr, pc) in contrib_t:
                    for (out_key, block_coeff) in contrib_t[(pr, pc)]:
                        out_row, out_col = out_key
                        W[w_idx(out_col, out_row)][rank_idx] += sub_coeff * block_coeff
    
    return {"U": U, "V": V, "W": W}
