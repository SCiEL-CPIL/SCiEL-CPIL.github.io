"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition block by block.
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    # We use a list of tuples to store the terms first, then assemble.
    # Each term is (U_dict, V_dict, W_dict) where keys are (i,j), (j,k), (k,i)
    # But we need to map to flat indices.
    # U_idx = i*m + j
    # V_idx = j*p + k
    # W_idx = k*n + i
    
    terms = []
    
    def add_strassen(i0, j0, k0):
        # Adds Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        # Local indices 0,1 map to global i0+0, i0+1, etc.
        
        # m1 = (a+d)(e+h)
        u1 = { (i0, j0): 1, (i0+1, j0+1): 1 }
        v1 = { (j0, k0): 1, (j0+1, k0+1): 1 }
        w1 = { (k0, i0): 1, (k0+1, i0+1): 1 }
        terms.append((u1, v1, w1))
        
        # m2 = (c+d)e
        u2 = { (i0+1, j0): 1, (i0+1, j0+1): 1 }
        v2 = { (j0, k0): 1 }
        w2 = { (k0, i0+1): 1 }
        terms.append((u2, v2, w2))
        
        # m3 = a(f-h)
        u3 = { (i0, j0): 1 }
        v3 = { (j0, k0+1): 1, (j0+1, k0+1): -1 }
        w3 = { (k0+1, i0): 1, (k0+1, i0+1): 1 }
        terms.append((u3, v3, w3))
        
        # m4 = d(g-e)
        u4 = { (i0+1, j0+1): 1 }
        v4 = { (j0+1, k0): 1, (j0, k0): -1 }
        w4 = { (k0, i0): 1, (k0, i0+1): 1 }
        terms.append((u4, v4, w4))
        
        # m5 = (a+b)h
        u5 = { (i0, j0): 1, (i0, j0+1): 1 }
        v5 = { (j0+1, k0+1): 1 }
        w5 = { (k0+1, i0): -1, (k0, i0): 1 } # C00 gets -m5, C11 gets +m5?
        # Check Strassen:
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        # So for m5:
        # W(k0, i0) should be -1? No, C00 is (k0, i0).
        # C00 term: -m5. So W(k0, i0) = -1.
        # C01 term: +m5. So W(k0+1, i0) = 1.
        w5 = { (k0, i0): -1, (k0+1, i0): 1 }
        terms.append((u5, v5, w5))
        
        # m6 = (c-a)(e+f)
        u6 = { (i0+1, j0): 1, (i0, j0): -1 }
        v6 = { (j0, k0): 1, (j0, k0+1): 1 }
        w6 = { (k0+1, i0+1): 1 }
        terms.append((u6, v6, w6))
        
        # m7 = (b-d)(g+h)
        u7 = { (i0, j0+1): 1, (i0+1, j0+1): -1 }
        v7 = { (j0+1, k0): 1, (j0+1, k0+1): 1 }
        w7 = { (k0, i0): 1, (k0+1, i0+1): 1 }
        terms.append((u7, v7, w7))

    def add_schoolbook(i0, j0, k0, di=1, dj=1, dk=1):
        # Adds schoolbook for A[i0:i0+di, j0:j0+dj] * B[j0:j0+dj, k0:k0+dk]
        for i in range(i0, i0+di):
            for j in range(j0, j0+dj):
                for k in range(k0, k0+dk):
                    u = { (i, j): 1 }
                    v = { (j, k): 1 }
                    w = { (k, i): 1 }
                    terms.append((u, v, w))

    # Block 1: i in [0,1], j in [0,1], k in [0,1] -> Strassen
    add_strassen(0, 0, 0)
    
    # Block 2: i in [0,1], j in [0,1], k in [2,3] -> Strassen
    add_strassen(0, 0, 2)
    
    # Block 3: i in [0,1], j in [0,1], k in [4,4] -> Schoolbook (2x2x1)
    add_schoolbook(0, 0, 4, 2, 2, 1)
    
    # Block 4: i in [0,1], j in [2,2], k in [0,1] -> Schoolbook (2x1x2)
    add_schoolbook(0, 2, 0, 2, 1, 2)
    
    # Block 5: i in [0,1], j in [2,2], k in [2,3] -> Schoolbook (2x1x2)
    add_schoolbook(0, 2, 2, 2, 1, 2)
    
    # Block 6: i in [0,1], j in [2,2], k in [4,4] -> Schoolbook (2x1x1)
    add_schoolbook(0, 2, 4, 2, 1, 1)

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[i * m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][c] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
