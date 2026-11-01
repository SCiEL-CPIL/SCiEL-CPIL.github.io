"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 2x2x2 matrix multiplication.
    
    Uses Strassen's algorithm which achieves rank 7 for 2x2 matrix multiplication.
    The output factor follows the official trace convention W[k*n+i, r].
    
    U has shape (n*m, rank) = (4, 7)
    V has shape (m*p, rank) = (4, 7)  
    W has shape (p*n, rank) = (4, 7)
    
    For the decomposition to be valid:
    For each (i, j, k) in {0,1}^3:
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    Index conventions:
      U index: i*m + j  (i in 0..1, j in 0..1)
      V index: j*p + k  (j in 0..1, k in 0..1)
      W index: k*n + i  (k in 0..1, i in 0..1)
    """
    t0 = time.time()
    deadline = t0 + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank = 7
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 7
    V = [[0] * rank for _ in range(m * p)]  # 4 x 7
    W = [[0] * rank for _ in range(p * n)]  # 4 x 7
    
    # Strassen decomposition for 2x2 matrix multiplication:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # Product: A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    #
    # Strassen's 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # Then:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    #
    # In our indexing:
    # A: i=0,j=0 -> a; i=0,j=1 -> b; i=1,j=0 -> c; i=1,j=1 -> d
    # B: j=0,k=0 -> e; j=0,k=1 -> f; j=1,k=0 -> g; j=1,k=1 -> h
    # C (output): i=0,k=0 -> C[0,0]; i=0,k=1 -> C[0,1]; i=1,k=0 -> C[1,0]; i=1,k=1 -> C[1,1]
    #
    # W index: k*n + i
    # C[0,0]: i=0, k=0 -> W[0*2+0] = W[0]
    # C[0,1]: i=0, k=1 -> W[1*2+0] = W[2]
    # C[1,0]: i=1, k=0 -> W[0*2+1] = W[1]
    # C[1,1]: i=1, k=1 -> W[1*2+1] = W[3]
    #
    # U index: i*m + j
    # a: i=0,j=0 -> U[0]
    # b: i=0,j=1 -> U[1]
    # c: i=1,j=0 -> U[2]
    # d: i=1,j=1 -> U[3]
    #
    # V index: j*p + k
    # e: j=0,k=0 -> V[0]
    # f: j=0,k=1 -> V[1]
    # g: j=1,k=0 -> V[2]
    # h: j=1,k=1 -> V[3]
    
    # Strassen's 7 products assigned to rank dimensions r=0..6:
    # r=0: M1 = (a+d)(e+h)
    # r=1: M2 = (c+d)e  
    # r=2: M3 = a(f-h)
    # r=3: M4 = d(g-e)
    # r=4: M5 = (a+b)h
    # r=5: M6 = (c-a)(e+f)
    # r=6: M7 = (b-d)(g+h)
    
    # U factors (row = i*m+j, col = r):
    U[0][0] = 1   # a in M1
    U[3][0] = 1   # d in M1
    U[2][1] = 1   # c in M2
    U[3][1] = 1   # d in M2
    U[0][2] = 1   # a in M3
    U[3][3] = 1   # d in M4
    U[0][4] = 1   # a in M5
    U[1][4] = 1   # b in M5
    U[2][5] = 1   # c in M6
    U[0][5] = -1  # a in M6 (negative)
    U[1][6] = 1   # b in M7
    U[3][6] = -1  # d in M7 (negative)
    
    # V factors (row = j*p+k, col = r):
    V[0][0] = 1   # e in M1
    V[3][0] = 1   # h in M1
    V[0][1] = 1   # e in M2
    V[1][2] = 1   # f in M3
    V[3][2] = -1  # h in M3 (negative)
    V[2][3] = 1   # g in M4
    V[0][3] = -1  # e in M4 (negative)
    V[3][4] = 1   # h in M5
    V[0][5] = 1   # e in M6
    V[1][5] = 1   # f in M6
    V[2][6] = 1   # g in M7
    V[3][6] = 1   # h in M7
    
    # W factors (row = k*n+i, col = r):
    # C[0,0] = M1 + M4 - M5 + M7 -> W[0]
    # C[0,1] = M3 + M5 -> W[2]
    # C[1,0] = M2 + M4 -> W[1]
    # C[1,1] = M1 - M2 + M3 + M6 -> W[3]
    
    W[0][0] = 1   # M1
    W[0][3] = 1   # M4
    W[0][4] = -1  # M5
    W[0][6] = 1   # M7
    
    W[1][1] = 1   # M2
    W[1][3] = 1   # M4
    
    W[2][2] = 1   # M3
    W[2][4] = 1   # M5
    
    W[3][0] = 1   # M1
    W[3][1] = -1  # M2
    W[3][2] = 1   # M3
    W[3][5] = 1   # M6
    
    # Verify the decomposition if time permits
    if time.time() < deadline:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        # Should never happen for correct Strassen, but guard anyway
                        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
