import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 * 2x2
    # Returns list of (u, v, w) where u, v, w are lists of length 4
    # u: A[i, j] -> i*2+j
    # v: B[i, j] -> i*2+j
    # w: C[i, j] -> i*2+j (Note: C is 2x2 here, but in global it's different)
    # Wait, w in tensor is C[k, i]? No, W[k*n+i, r].
    # In 2x2 case, W indices are k*2+i.
    
    def get_strassen_2x2():
        # u, v, w are 4-vectors
        # A = [[a0, a1], [a2, a3]] -> u = [a0, a1, a2, a3]
        # B = [[b0, b1], [b2, b3]] -> v = [b0, b1, b2, b3]
        # C = [[c0, c1], [c2, c3]] -> w = [c0, c1, c2, c3]
        
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # We need the linear combinations for u, v, w.
        # u is coefficients for A.
        # v is coefficients for B.
        # w is coefficients for C.
        
        terms = []
        
        # M1
        u1 = [1, 0, 0, 1] # A00 + A11
        v1 = [1, 0, 0, 1] # B00 + B11
        # C00 += M1, C11 += M1
        w1 = [1, 0, 0, 1]
        terms.append((u1, v1, w1))
        
        # M2
        u2 = [0, 0, 1, 1] # A10 + A11
        v2 = [1, 0, 0, 0] # B00
        # C10 += M2, C11 -= M2
        w2 = [0, 0, 1, -1]
        terms.append((u2, v2, w2))
        
        # M3
        u3 = [1, 0, 0, 0] # A00
        v3 = [0, 1, 0, -1] # B01 - B11
        # C01 += M3, C11 += M3
        w3 = [0, 1, 0, 1]
        terms.append((u3, v3, w3))
        
        # M4
        u4 = [0, 0, 0, 1] # A11
        v4 = [0, 0, 1, -1] # B10 - B00
        # C00 += M4, C10 += M4
        w4 = [1, 0, 1, 0]
        terms.append((u4, v4, w4))
        
        # M5
        u5 = [1, 1, 0, 0] # A00 + A01
        v5 = [0, 0, 0, 1] # B11
        # C00 -= M5, C01 += M5
        w5 = [-1, 1, 0, 0]
        terms.append((u5, v5, w5))
        
        # M6
        u6 = [-1, 0, 1, 0] # A10 - A00
        v6 = [1, 1, 0, 0] # B00 + B01
        # C11 += M6
        w6 = [0, 0, 0, 1]
        terms.append((u6, v6, w6))
        
        # M7
        u7 = [0, 1, 0, -1] # A01 - A11
        v7 = [0, 0, 1, 1] # B10 + B11
        # C00 += M7
        w7 = [1, 0, 0, 0]
        terms.append((u7, v7, w7))
        
        return terms

    strassen_terms = get_strassen_2x2()
    
    col = 0
    # Part 1: A cols 0,1 ; B rows 0,1
    # A local index (i, jl) -> global U[i*4 + jl]
    # B local index (jl, k) -> global V[jl*8 + k]  (jl is 0,1)
    # C local index (k, i) -> global W[k*2 + i]
    
    for block in range(4): # 4 blocks of 2x2 in B/C
        for (u_s, v_s, w_s) in strassen_terms:
            # u_s is 4-vector [A00, A01, A10, A11]
            # Map to U
            for i in range(2):
                for jl in range(2):
                    val = u_s[i * 2 + jl]
                    if val != 0:
                        U[i * 4 + jl][col] += val
                        
            # v_s is 4-vector [B00, B01, B10, B11]
            # Map to V. B rows 0,1. Cols 2*block, 2*block+1
            for jl in range(2):
                for kl in range(2):
                    val = v_s[jl * 2 + kl]
                    if val != 0:
                        k_global = 2 * block + kl
                        V[jl * 8 + k_global][col] += val
                        
            # w_s is 4-vector [C00, C01, C10, C11]
            # Map to W. C rows 2*block, 2*block+1. Cols 0,1
            for kl in range(2):
                for i in range(2):
                    val = w_s[kl * 2 + i]
                    if val != 0:
                        k_global = 2 * block + kl
                        W[k_global * 2 + i][col] += val
                        
            col += 1
            
    # Part 2: A cols 2,3 ; B rows 2,3
    # A local index (i, jl) -> global U[i*4 + (2+jl)]
    # B local index (jl, k) -> global V[(2+jl)*8 + k]
    # C local index (k, i) -> global W[k*2 + i]
    
    for block in range(4):
        for (u_s, v_s, w_s) in strassen_terms:
            # u_s
            for i in range(2):
                for jl in range(2):
                    val = u_s[i * 2 + jl]
                    if val != 0:
                        U[i * 4 + (2 + jl)][col] += val
                        
            # v_s
            for jl in range(2):
                for kl in range(2):
                    val = v_s[jl * 2 + kl]
                    if val != 0:
                        k_global = 2 * block + kl
                        V[(2 + jl) * 8 + k_global][col] += val
                        
            # w_s
            for kl in range(2):
                for i in range(2):
                    val = w_s[kl * 2 + i]
                    if val != 0:
                        k_global = 2 * block + kl
                        W[k_global * 2 + i][col] += val
                        
            col += 1

    return {"U": U, "V": V, "W": W}
