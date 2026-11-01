import numpy as np
import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=13
    
    # We will use Strassen for 2x2 blocks and Schoolbook for the remaining 2x1 block.
    # p = 13 = 2*6 + 1.
    # 6 blocks of size 2x2, 1 block of size 2x1.
    
    rank = 0
    U = []
    V = []
    W = []
    
    def add_term(u, v, w):
        nonlocal rank
        U.append(u)
        V.append(v)
        W.append(w)
        rank += 1

    # Strassen decomposition for M_2,2,2
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # C = [[C00, C01], [C10, C11]]
    
    # u vectors (coeffs of a, b, c, d)
    # v vectors (coeffs of e, f, g, h)
    # w vectors (coeffs of C00, C01, C10, C11)
    
    # M1 = (a+d)(e+h)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, -1] # C00 += 1, C11 -= 1
    
    # M2 = (c+d)e
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, -1] # C10 += 1, C11 -= 1
    
    # M3 = a(f-h)
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 0, 1] # C01 += 1, C11 += 1
    
    # M4 = d(g-e)
    u4 = [0, 0, 0, 1]
    v4 = [0, 0, 1, -1] # e=0, f=0, g=1, h=-1? No.
    # B = [[e, f], [g, h]]
    # v4 should pick g and -e.
    # e is idx 0, f is idx 1, g is idx 2, h is idx 3.
    # So v4 = [-1, 0, 1, 0]
    v4 = [-1, 0, 1, 0]
    w4 = [1, 0, 1, 0] # C00 += 1, C10 += 1
    
    # M5 = (a+b)h
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 1, 0, 0] # C00 -= 1, C01 += 1
    
    # M6 = (c-a)(e+f)
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1] # C11 += 1
    
    # M7 = (b-d)(g+h)
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [1, 0, 0, 0] # C00 += 1
    
    strassen_terms = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]
    
    # Process 6 blocks of size 2x2
    for block_idx in range(6):
        k_start = block_idx * 2
        k_end = k_start + 2
        
        for u, v, w in strassen_terms:
            # Map v to full V vector
            full_v = [0] * (m * p)
            # B indices for this block:
            # j=0, k=k_start -> 0*p + k_start
            # j=0, k=k_start+1 -> 0*p + k_start+1
            # j=1, k=k_start -> 1*p + k_start
            # j=1, k=k_start+1 -> 1*p + k_start+1
            full_v[0 * p + k_start] = v[0]
            full_v[0 * p + k_start + 1] = v[1]
            full_v[1 * p + k_start] = v[2]
            full_v[1 * p + k_start + 1] = v[3]
            
            # Map w to full W vector
            full_w = [0] * (p * n)
            # C indices for this block:
            # i=0, k=k_start -> k_start*n + 0
            # i=1, k=k_start -> k_start*n + 1
            # i=0, k=k_start+1 -> (k_start+1)*n + 0
            # i=1, k=k_start+1 -> (k_start+1)*n + 1
            full_w[k_start * n + 0] = w[0]
            full_w[k_start * n + 1] = w[1]
            full_w[(k_start + 1) * n + 0] = w[2]
            full_w[(k_start + 1) * n + 1] = w[3]
            
            add_term(u, full_v, full_w)
            
    # Process 1 block of size 2x1 (k=12)
    k_start = 12
    # Schoolbook for M_2,2,1
    # C0 = a*e + b*g
    # C1 = c*e + d*g
    # e = B_0,12, g = B_1,12
    
    # Term 1: a * B_0,12 * C_0,12
    u_t1 = [1, 0, 0, 0]
    v_t1 = [0] * (m * p)
    v_t1[0 * p + k_start] = 1
    w_t1 = [0] * (p * n)
    w_t1[k_start * n + 0] = 1
    add_term(u_t1, v_t1, w_t1)
    
    # Term 2: b * B_1,12 * C_0,12
    u_t2 = [0, 1, 0, 0]
    v_t2 = [0] * (m * p)
    v_t2[1 * p + k_start] = 1
    w_t2 = [0] * (p * n)
    w_t2[k_start * n + 0] = 1
    add_term(u_t2, v_t2, w_t2)
    
    # Term 3: c * B_0,12 * C_1,12
    u_t3 = [0, 0, 1, 0]
    v_t3 = [0] * (m * p)
    v_t3[0 * p + k_start] = 1
    w_t3 = [0] * (p * n)
    w_t3[k_start * n + 1] = 1
    add_term(u_t3, v_t3, w_t3)
    
    # Term 4: d * B_1,12 * C_1,12
    u_t4 = [0, 0, 0, 1]
    v_t4 = [0] * (m * p)
    v_t4[1 * p + k_start] = 1
    w_t4 = [0] * (p * n)
    w_t4[k_start * n + 1] = 1
    add_term(u_t4, v_t4, w_t4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
