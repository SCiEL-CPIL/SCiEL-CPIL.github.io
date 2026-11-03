import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # rank = 54
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen coefficients
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # Strassen blocks
    # j_offsets: 0, 2
    # k_offsets: 0, 2, 4
    j_offsets = [0, 2]
    k_offsets = [0, 2, 4]
    
    for j_off in j_offsets:
        for k_off in k_offsets:
            # Local indices
            # a: i=0, j=j_off
            # b: i=0, j=j_off+1
            # c: i=1, j=j_off
            # d: i=1, j=j_off+1
            # e: j=j_off, k=k_off
            # f: j=j_off, k=k_off+1
            # g: j=j_off+1, k=k_off
            # h: j=j_off+1, k=k_off+1
            
            # M1: (a+d)(e+h)
            # U: a(0), d(1)
            # V: e(0), h(1)
            # W: C00(0), C01(1), C10(2), C11(3) -> M1 is in C00, C11
            
            # We add 7 columns for each block
            
            # M1
            U[0*m + j_off][col] = 1
            U[1*m + j_off][col] = 1
            V[j_off*p + k_off][col] = 1
            V[(j_off+1)*p + (k_off+1)][col] = 1
            W[k_off*n + 0][col] = 1 # C00
            W[(k_off+1)*n + 1][col] = 1 # C11
            col += 1
            
            # M2: (c+d)e
            U[1*m + j_off][col] = 1
            U[1*m + (j_off+1)][col] = 1
            V[j_off*p + k_off][col] = 1
            W[k_off*n + 1][col] = 1 # C10
            W[(k_off+1)*n + 1][col] = -1 # C11
            col += 1
            
            # M3: a(f-h)
            U[0*m + j_off][col] = 1
            V[j_off*p + (k_off+1)][col] = 1
            V[(j_off+1)*p + k_off][col] = -1
            W[k_off*n + 0][col] = 1 # C01
            W[(k_off+1)*n + 1][col] = 1 # C11
            col += 1
            
            # M4: d(g-e)
            U[1*m + (j_off+1)][col] = 1
            V[(j_off+1)*p + k_off][col] = 1
            V[j_off*p + k_off][col] = -1
            W[k_off*n + 0][col] = 1 # C00
            W[k_off*n + 1][col] = 1 # C10
            col += 1
            
            # M5: (a+b)h
            U[0*m + j_off][col] = 1
            U[0*m + (j_off+1)][col] = 1
            V[(j_off+1)*p + (k_off+1)][col] = 1
            W[k_off*n + 0][col] = -1 # C00
            W[k_off*n + 0][col] = 1 # C01 -> Wait, M5 is in C00 and C01?
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # So M5 contributes -1 to C00 and +1 to C01.
            W[k_off*n + 0][col] = -1
            W[k_off*n + 0][col] = 1 # Overwrite? No, distinct columns.
            # Correct W indices:
            # C00 index: k_off*n + 0
            # C01 index: k_off*n + 0? No, C01 is i=0, k=k_off.
            # W index is k*n + i.
            # C00: k=k_off, i=0 -> W[k_off*n]
            # C01: k=k_off, i=1? No, C is 2x6. i is row, k is col.
            # C[i, k]. W[k*n + i].
            # C00: i=0, k=k_off -> W[k_off*n + 0]
            # C01: i=0, k=k_off+1? No, local C is 2x2.
            # Local C00 -> Global C[i=0, k=k_off]
            # Local C01 -> Global C[i=0, k=k_off+1]
            # Local C10 -> Global C[i=1, k=k_off]
            # Local C11 -> Global C[i=1, k=k_off+1]
            
            # Let's re-verify W indices for M5.
            # M5 is in C00 (local) and C01 (local).
            # Local C00 -> Global W[k_off*n + 0]
            # Local C01 -> Global W[(k_off+1)*n + 0]
            
            W[k_off*n + 0][col] = -1
            W[(k_off+1)*n + 0][col] = 1
            col += 1
            
            # M6: (c-a)(e+f)
            U[1*m + j_off][col] = 1
            U[0*m + j_off][col] = -1
            V[j_off*p + k_off][col] = 1
            V[j_off*p + (k_off+1)][col] = 1
            W[(k_off+1)*n + 1][col] = 1 # C11
            col += 1
            
            # M7: (b-d)(g+h)
            U[0*m + (j_off+1)][col] = 1
            U[1*m + (j_off+1)][col] = -1
            V[(j_off+1)*p + k_off][col] = 1
            V[(j_off+1)*p + (k_off+1)][col] = 1
            W[k_off*n + 0][col] = 1 # C00
            col += 1

    # Schoolbook blocks for j=4
    # j=4, k_offsets 0, 2, 4
    for k_off in k_offsets:
        # A[0,4] B[4, k_off] -> C[0, k_off]
        U[0*m + 4][col] = 1
        V[4*p + k_off][col] = 1
        W[k_off*n + 0][col] = 1
        col += 1
        
        # A[0,4] B[4, k_off+1] -> C[0, k_off+1]
        U[0*m + 4][col] = 1
        V[4*p + (k_off+1)][col] = 1
        W[(k_off+1)*n + 0][col] = 1
        col += 1
        
        # A[1,4] B[4, k_off] -> C[1, k_off]
        U[1*m + 4][col] = 1
        V[4*p + k_off][col] = 1
        W[k_off*n + 1][col] = 1
        col += 1
        
        # A[1,4] B[4, k_off+1] -> C[1, k_off+1]
        U[1*m + 4][col] = 1
        V[4*p + (k_off+1)][col] = 1
        W[(k_off+1)*n + 1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
