"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 3
    
    # Rank-17 decomposition for 2x3 * 3x3
    # Strategy: Use Strassen on the 2x2x2 sub-block and direct for the rest.
    # 
    # The 18 schoolbook terms are (i,j,k) for i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # 
    # For j in {0,1} and k in {0,1}: 8 terms forming a 2x2x2 tensor
    # Strassen reduces 8 -> 7, saving 1
    # For j=2: 2x1x3 = 6 terms (can't reduce below 6? Actually 2x1x3 has rank 2)
    # Wait, j=2 means A[:,2] (2x1) times B[2,:] (1x3)
    # That's a rank-2 tensor (outer product structure), so 2 terms suffice
    # 
    # But we also have j in {0,1} with k=2: 2x2x1 = 4 terms, rank 4? 
    # Actually 2x2x1: C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
    # This is 2 independent dot products, rank 2? No, it's 4 terms but structured.
    # 
    # Let me just build it and verify.
    
    # I'll construct a rank-17 solution and verify it.
    
    rank = 17
    U = [[0]*rank for _ in range(n*m)]  # 6 x 17
    V = [[0]*rank for _ in range(m*p)]  # 9 x 17
    W = [[0]*rank for _ in range(p*n)]  # 6 x 17
    
    # Terms 0-6: Strassen for the 2x2x2 block (i in 0,1; j in 0,1; k in 0,1)
    # Standard Strassen for 2x2 * 2x2:
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[p,q],[r,s]]
    # p = a*e + b*g
    # q = a*f + b*h
    # r = c*e + d*g
    # s = c*f + d*h
    #
    # Strassen:
    # M1 = a*e
    # M2 = a*f
    # M3 = (a+b)*(g+h)
    # M4 = b*g
    # M5 = (c+d)*e
    # M6 = d*(f+h)
    # M7 = (a+c)*(f-h)
    #
    # p = M1 + M4
    # q = M2 + M6
    # r = M5 + M7
    # s = M3 - M1 + M2 + M7
    #
    # In our indexing:
    # A[i,j] -> U[3*i+j]
    # B[j,k] -> V[3*j+k]
    # C[i,k] -> W[2*k+i]
    #
    # a=A[0,0]=U[0], b=A[0,1]=U[1], c=A[1,0]=U[3], d=A[1,1]=U[4]
    # e=B[0,0]=V[0], f=B[0,1]=V[1], g=B[1,0]=V[3], h=B[1,1]=V[4]
    # p=C[0,0]=W[0], q=C[0,1]=W[2], r=C[1,0]=W[1], s=C[1,1]=W[3]
    
    # M1 = a*e: U[0]=1, V[0]=1, W[0]=1
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    
    # M2 = a*f: U[0]=1, V[1]=1, W[2]=1
    U[0][1] = 1; V[1][1] = 1; W[2][1] = 1
    
    # M3 = (a+b)*(g+h): U[0]=1, U[1]=1, V[3]=1, V[4]=1
    # Contributes to s = C[1,1] = W[3]
    U[0][2] = 1; U[1][2] = 1; V[3][2] = 1; V[4][2] = 1; W[3][2] = 1
    
    # M4 = b*g: U[1]=1, V[3]=1, W[0]=1
    U[1][3] = 1; V[3][3] = 1; W[0][3] = 1
    
    # M5 = (c+d)*e: U[3]=1, U[4]=1, V[0]=1, W[1]=1
    U[3][4] = 1; U[4][4] = 1; V[0][4] = 1; W[1][4] = 1
    
    # M6 = d*(f+h): U[4]=1, V[1]=1, V[4]=1, W[2]=1
    U[4][5] = 1; V[1][5] = 1; V[4][5] = 1; W[2][5] = 1
    
    # M7 = (a+c)*(f-h): U[0]=1, U[3]=1, V[1]=1, V[4]=-1, W[1]=1, W[3]=1
    U[0][6] = 1; U[3][6] = 1; V[1][6] = 1; V[4][6] = -1; W[1][6] = 1; W[3][6] = 1
    
    # Now verify: C[0,0]=W[0] should equal a*e + b*g = M1 + M4
    # W[0] = 1*U[0]*V[0]*W[0] + 1*U[1]*V[3]*W[0] = 1*1*1*1 + 1*1*1*1 = 2? No!
    # W[0] gets contributions from term 0 and term 3.
    # Term 0: U[0][0]*V[0][0]*W[0][0] = 1*1*1 = 1
    # Term 3: U[0][3]*V[0][3]*W[0][3] = 0*0*1 = 0 (U[0][3]=0)
    # Wait, I need to check all terms for each (i,j,k)
    
    # Let me verify computationally after construction.
    
    # Terms 7-10: Handle k=2 for j in {0,1}: C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
    # That's 4 terms: (i=0,j=0,k=2), (i=0,j=1,k=2), (i=1,j=0,k=2), (i=1,j=1,k=2)
    # These are 4 separate rank-1 terms
    for idx, (i, j) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
        col = 7 + idx
        U[3*i+j][col] = 1
        V[3*j+2][col] = 1
        W[2*2+i][col] = 1  # k=2, so W[4+i]
    
    # Terms 11-16: Handle j=2: C[i,k] = A[i,2]*B[2,k] for i in {0,1}, k in {0,1,2}
    # That's 6 terms
    for idx, (i, k) in enumerate([(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]):
        col = 11 + idx
        U[3*i+2][col] = 1
        V[3*2+k][col] = 1
        W[2*k+i][col] = 1
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(len(U[0])):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if total != 1:
                        return False, (i, j, k, total)
        return True, None
    
    valid, err = verify(U, V, W, n, m, p)
    if not valid:
        # Fall back to schoolbook rank 18
        U = [[0]*18 for _ in range(n*m)]
        V = [[0]*18 for _ in range(m*p)]
        W = [[0]*18 for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
