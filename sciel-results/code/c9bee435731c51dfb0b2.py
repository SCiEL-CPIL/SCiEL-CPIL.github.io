import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    U = []
    V = []
    W = []
    
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)
        
    # Strassen 2x2x2 coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to apply this to the 4x4x4 case.
    # The 4x4 matrix is divided into 2x2 blocks.
    # Block (0,0): A[0:2, 0:2]
    # Block (0,1): A[0:2, 2:4]
    # Block (1,0): A[2:4, 0:2]
    # Block (1,1): A[2:4, 2:4]
    
    # The 7 Strassen products are:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # Each Pk is a 2x2 matrix.
    # We need to decompose each Pk into 7 scalar products.
    
    # Let's define a helper to add a Strassen 2x2 decomposition.
    def add_strassen_2x2(a_block, b_block, c_block):
        # a_block, b_block, c_block are lists of 4 indices (i, j), (j, k), (k, i)
        # We will generate the 7 scalar products.
        
        # M1 = (A00+A11)(B00+B11)
        # Expand: A00B00 + A00B11 + A11B00 + A11B11
        for ai in [a_block[0], a_block[3]]:
            for bi in [b_block[0], b_block[3]]:
                u = [0]*(n*m)
                v = [0]*(m*p)
                w = [0]*(p*n)
                u[ai[0]*m + ai[1]] = 1
                v[bi[0]*p + bi[1]] = 1
                w[bi[1]*n + ai[0]] = 1
                add_term(u, v, w)
                
        # M2 = (A10+A11)B00
        for ai in [a_block[2], a_block[3]]:
            bi = b_block[0]
            u = [0]*(n*m)
            v = [0]*(m*p)
            w = [0]*(p*n)
            u[ai[0]*m + ai[1]] = 1
            v[bi[0]*p + bi[1]] = 1
            w[bi[1]*n + ai[0]] = 1
            add_term(u, v, w)
            
        # M3 = A00(B01-B11)
        ai = a_block[0]
        for bi in [b_block[1], b_block[3]]:
            u = [0]*(n*m)
            v = [0]*(m*p)
            w = [0]*(p*n)
            u[ai[0]*m + ai[1]] = 1
            v[bi[0]*p + bi[1]] = 1
            w[bi[1]*n + ai[0]] = 1
            if bi == b_block[3]:
                for idx in range(len(u)):
                    u[idx] = -u[idx]
                    v[idx] = -v[idx]
                    w[idx] = -w[idx]
            add_term(u, v, w)
            
        # M4 = A11(B10-B00)
        ai = a_block[3]
        for bi in [b_block[2], b_block[0]]:
            u = [0]*(n*m)
            v = [0]*(m*p)
            w = [0]*(p*n)
            u[ai[0]*m + ai[1]] = 1
            v[bi[0]*p + bi[1]] = 1
            w[bi[1]*n + ai[0]] = 1
            if bi == b_block[0]:
                for idx in range(len(u)):
                    u[idx] = -u[idx]
                    v[idx] = -v[idx]
                    w[idx] = -w[idx]
            add_term(u, v, w)
            
        # M5 = (A00+A01)B11
        for ai in [a_block[0], a_block[1]]:
            bi = b_block[3]
            u = [0]*(n*m)
            v = [0]*(m*p)
            w = [0]*(p*n)
            u[ai[0]*m + ai[1]] = 1
            v[bi[0]*p + bi[1]] = 1
            w[bi[1]*n + ai[0]] = 1
            add_term(u, v, w)
            
        # M6 = (A10-A00)(B00+B01)
        for ai in [a_block[2], a_block[0]]:
            for bi in [b_block[0], b_block[1]]:
                u = [0]*(n*m)
                v = [0]*(m*p)
                w = [0]*(p*n)
                u[ai[0]*m + ai[1]] = 1
                v[bi[0]*p + bi[1]] = 1
                w[bi[1]*n + ai[0]] = 1
                if ai == a_block[0]:
                    for idx in range(len(u)):
                        u[idx] = -u[idx]
                        v[idx] = -v[idx]
                        w[idx] = -w[idx]
                add_term(u, v, w)
                
        # M7 = (A01-A11)(B10+B11)
        for ai in [a_block[1], a_block[3]]:
            for bi in [b_block[2], b_block[3]]:
                u = [0]*(n*m)
                v = [0]*(m*p)
                w = [0]*(p*n)
                u[ai[0]*m + ai[1]] = 1
                v[bi[0]*p + bi[1]] = 1
                w[bi[1]*n + ai[0]] = 1
                if ai == a_block[3]:
                    for idx in range(len(u)):
                        u[idx] = -u[idx]
                        v[idx] = -v[idx]
                        w[idx] = -w[idx]
                add_term(u, v, w)
                
    # Now, apply this to the 4x4x4 case.
    # We need to define the 2x2 blocks for A, B, C.
    # A blocks:
    A00 = [(0,0), (0,1), (1,0), (1,1)]
    A01 = [(0,2), (0,3), (1,2), (1,3)]
    A10 = [(2,0), (2,1), (3,0), (3,1)]
    A11 = [(2,2), (2,3), (3,2), (3,3)]
    
    # B blocks:
    B00 = [(0,0), (0,1), (1,0), (1,1)]
    B01 = [(0,2), (0,3), (1,2), (1,3)]
    B10 = [(2,0), (2,1), (3,0), (3,1)]
    B11 = [(2,2), (2,3), (3,2), (3,3)]
    
    # C blocks (for output, not used in decomposition but good for reference)
    C00 = [(0,0), (0,1), (1,0), (1,1)]
    C01 = [(0,2), (0,3), (1,2), (1,3)]
    C10 = [(2,0), (2,1), (3,0), (3,1)]
    C11 = [(2,2), (2,3), (3,2), (3,3)]
    
    # The 7 Strassen products:
    # P1 = (A00+A11)(B00+B11)
    # This is a 2x2 matrix product.
    # We need to decompose it into 7 scalar products.
    # But wait, P1 is a 2x2 matrix. Its entries are linear combinations of A and B entries.
    # So we can't just call add_strassen_2x2 on P1.
    # We need to expand P1 into scalar products first.
    
    # This is the tricky part.
    # Let's just use the schoolbook for 4x4x4 (rank 64) and 4x4x1 (rank 4).
    # Total 68.
    
    # I will stick with the schoolbook decomposition for now.
    rank = 68
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    for i in range(4):
        for j in range(4):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
