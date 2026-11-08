import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 4, 4, 7
    terms = []

    def add_term(u, v, w):
        terms.append((u, v, w))

    def strassen_2x2_block(i_off, j_off, k_off, a_scale=1, b_scale=1, c_scale=1):
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        # M1
        u[i_off * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v[j_off * p + k_off] = b_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w[k_off * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u[:], v[:], w[:])
        
        u = [0] * (n * m); v = [0] * (m * p); w = [0] * (p * n)
        # M2
        u[(i_off+1) * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v[j_off * p + k_off] = b_scale
        w[k_off * n + (i_off+1)] = c_scale
        w[(k_off+1) * n + (i_off+1)] = -c_scale
        add_term(u[:], v[:], w[:])

        u = [0] * (n * m); v = [0] * (m * p); w = [0] * (p * n)
        # M3
        u[i_off * m + j_off] = a_scale
        v[j_off * p + (k_off+1)] = b_scale
        v[(j_off+1) * p + (k_off+1)] = -b_scale
        w[(k_off+1) * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u[:], v[:], w[:])

        u = [0] * (n * m); v = [0] * (m * p); w = [0] * (p * n)
        # M4
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v[(j_off+1) * p + k_off] = b_scale
        v[j_off * p + k_off] = -b_scale
        w[k_off * n + i_off] = c_scale
        w[k_off * n + (i_off+1)] = c_scale
        add_term(u[:], v[:], w[:])

        u = [0] * (n * m); v = [0] * (m * p); w = [0] * (p * n)
        # M5
        u[i_off * m + j_off] = a_scale
        u[i_off * m + (j_off+1)] = a_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w[(k_off+1) * n + i_off] = c_scale
        w[k_off * n + i_off] = -c_scale
        add_term(u[:], v[:], w[:])

        u = [0] * (n * m); v = [0] * (m * p); w = [0] * (p * n)
        # M6
        u[(i_off+1) * m + j_off] = a_scale
        u[i_off * m + j_off] = -a_scale
        v[j_off * p + k_off] = b_scale
        v[j_off * p + (k_off+1)] = b_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u[:], v[:], w[:])

        u = [0] * (n * m); v = [0] * (m * p); w = [0] * (p * n)
        # M7
        u[i_off * m + (j_off+1)] = a_scale
        u[(i_off+1) * m + (j_off+1)] = -a_scale
        v[(j_off+1) * p + k_off] = b_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w[k_off * n + i_off] = c_scale
        add_term(u[:], v[:], w[:])

    def strassen_4x4_head():
        # Standard recursive Strassen for 4x4x4
        # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
        u1 = [0]*16; u1[0]=1; u1[13]=1
        v1 = [0]*28; v1[0]=1; v1[11]=1
        w1 = [0]*28; w1[0]=1; w1[13]=1
        
        u2 = [0]*16; u2[8]=1; u2[13]=1
        v2 = [0]*28; v2[0]=1
        w2 = [0]*28; w2[8]=1; w2[13]=-1

        u3 = [0]*16; u3[0]=1
        v3 = [0]*28; v3[1]=1; v3[11]=-1
        w3 = [0]*28; w3[4]=1; w3[13]=1

        u4 = [0]*16; u4[13]=1
        v4 = [0]*28; v4[2]=1; v4[0]=-1
        w4 = [0]*28; w4[0]=1; w4[8]=1

        u5 = [0]*16; u5[0]=1; u5[1]=1
        v5 = [0]*28; v5[11]=1
        w5 = [0]*28; w5[4]=1; w5[0]=-1

        u6 = [0]*16; u6[8]=1; u6[0]=-1
        v6 = [0]*28; v6[0]=1; v6[1]=1
        w6 = [0]*28; w6[13]=1

        u7 = [0]*16; u7[1]=1; u7[13]=-1
        v7 = [0]*28; v7[2]=1; v7[11]=1
        w7 = [0]*28; w7[0]=1

        def block_op(u, v, w, i_off, j_off, k_off):
             # u is 4x4 block indices, v is 4x2 block indices, w is 4x4 block indices? No.
             # For 4x4 * 4x4, indices are 0..3.
             # For our 4x4 * 4x7, the 'B' part is 4x4 (cols 0..3) and 'C' part is 4x4 (cols 0..3).
             # So we can just map the 4x4 indices directly.
             nu = [0]*16
             nv = [0]*28
             nw = [0]*28
             
             for r in range(4):
                 for c in range(4):
                     idx_u = r*4+c
                     if u[idx_u] != 0:
                         # Map to global A index: row i, col j
                         # Here i,j are 0..3
                         nu[r*4+c] = u[idx_u]
                         
             # B mapping: rows 0..3, cols 0..3 -> global j, k
             for r in range(4):
                 for c in range(4):
                     idx_v = r*4+c
                     if v[idx_v] != 0:
                         # global col k is c
                         # global row j is r
                         nv[r*7+c] = v[idx_v]
                         
             # C mapping: rows 0..3 (k), cols 0..3 (i) -> global k, i
             for r in range(4):
                 for c in range(4):
                     idx_w = r*4+c
                     if w[idx_w] != 0:
                         # global row i is c
                         # global col k is r
                         nw[r*4+c] = w[idx_w]
                         
             add_term(nu, nv, nw)

        # Apply the 7 Strassen terms for the 4x4x4 head
        block_op(u1, v1, w1, 0, 0, 0)
        block_op(u2, v2, w2, 0, 0, 0)
        block_op(u3, v3, w3, 0, 0, 0)
        block_op(u4, v4, w4, 0, 0, 0)
        block_op(u5, v5, w5, 0, 0, 0)
        block_op(u6, v6, w6, 0, 0, 0)
        block_op(u7, v7, w7, 0, 0, 0)
        
        # Wait, the above block_op just places the 4x4 Strassen pattern in the top-left 4x4 of A, top-left 4x4 of B, top-left 4x4 of C.
        # But Strassen 4x4 is not just placing 7 terms of 2x2. It involves 49 terms if we recurse, or 7 terms if we treat the 2x2 blocks as atomic?
        # No, Strassen complexity is log. 
        # If we use the explicit 49 terms (7^2), that's expensive.
        # The previous program used `strassen_4x4_head` which generated 49 terms?
        # Let's check the previous program's `strassen_4x4_head`.
        # It iterated outer_terms (7) and inner_terms (7) and did add_term. So 49 terms.
        # 49 terms for 4x4x4 is correct for Strassen recursion depth 2 (4=2^2, 7^2=49).
        # My block_op above only adds 7 terms, which is WRONG for 4x4. It would only handle 2x2x2 if the inputs were 2x2.
        # I need to generate the 49 terms properly.

    def strassen_4x4_head_correct():
        # Recursively generate Strassen terms for 4x4x4
        def rec(i_off, j_off, k_off, size):
            if size == 2:
                strassen_2x2_block(i_off, j_off, k_off)
                return
            half = size // 2
            rec(i_off, j_off, k_off, half)           # M1
            rec(i_off+half, j_off, k_off, half)     # M2
            rec(i_off, j_off+half, k_off+half, half) # M3
            rec(i_off+half, j_off+half, k_off, half) # M4
            rec(i_off, j_off, k_off+half, half)     # M5
            rec(i_off+half, j_off, k_off, half)     # M6
            rec(i_off, j_off+half, k_off+half, half) # M7

        rec(0, 0, 0, 4)

    def strassen_4x4x2_tail():
        # 2 blocks of 2x2 * 2x2 for columns 4,5
        for i_b in range(2):
            for j_b in range(2):
                i_off = i_b * 2
                j_off = j_b * 2
                k_off = 4
                strassen_2x2_block(i_off, j_off, k_off)

    def schoolbook_4x4x1_tail():
        # 16 terms for column 6
        for i in range(n):
            for j in range(m):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + 6] = 1
                w = [0] * (p * n)
                w[6 * n + i] = 1
                add_term(u, v, w)

    strassen_4x4_head_correct()
    strassen_4x4x2_tail()
    schoolbook_4x4x1_tail()

    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0:
                W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}
