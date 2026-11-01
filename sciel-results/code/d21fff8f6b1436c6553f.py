"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    # Rank 14 decomposition using Strassen on each 2x2 block of B
    # A is 2x2: A[0][0]=a, A[0][1]=b, A[1][0]=c, A[1][1]=d
    # B is 2x4: B[0][0]=e, B[0][1]=f, B[1][0]=g, B[1][1]=h,
    #            B[0][2]=i, B[0][3]=j, B[1][2]=k, B[1][3]=l
    # C is 2x4: C[0][0], C[0][1], C[1][0], C[1][1], C[0][2], C[0][3], C[1][2], C[1][3]
    # W convention: W[k*n+i, r] where k is B-row, i is C-row... 
    # Actually W has shape (p*n, rank) = (4*2, rank) = (8, rank)
    # W index: the output C has shape n x p = 2 x 4
    # W[k*n+i, r] - this seems to be W indexed by (output column k, output row i)?
    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # C[i][k] for i in range(n), k in range(p)
    # W[k*n+i, r] means W is indexed with k first then i? That would be W[k*2+i, r]
    # So W[k*2+i, r] corresponds to C[i][k]
    
    n, m, p = DIMENSIONS
    # U has shape (n*m, rank) = (4, rank)
    # V has shape (m*p, rank) = (8, rank)  
    # W has shape (p*n, rank) = (8, rank)
    
    rank = 14
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # A indices in U: U[i*m+j, r] where i is row of A, j is col of A
    # A[0][0] -> U[0, r], A[0][1] -> U[1, r], A[1][0] -> U[2, r], A[1][1] -> U[3, r]
    
    # B indices in V: V[j*p+k, r] where j is row of B, k is col of B
    # B[0][0] -> V[0, r], B[0][1] -> V[1, r], B[0][2] -> V[2, r], B[0][3] -> V[3, r]
    # B[1][0] -> V[4, r], B[1][1] -> V[5, r], B[1][2] -> V[6, r], B[1][3] -> V[7, r]
    
    # W indices: W[k*n+i, r] where k is col of C, i is row of C
    # C[0][0] -> W[0*2+0, r] = W[0, r]
    # C[0][1] -> W[1*2+0, r] = W[2, r]
    # C[0][2] -> W[2*2+0, r] = W[4, r]
    # C[0][3] -> W[3*2+0, r] = W[6, r]
    # C[1][0] -> W[0*2+1, r] = W[1, r]
    # C[1][1] -> W[1*2+1, r] = W[3, r]
    # C[1][2] -> W[2*2+1, r] = W[5, r]
    # C[1][3] -> W[3*2+1, r] = W[7, r]
    
    # For block 1 (B columns 0,1): e=B[0][0], f=B[0][1], g=B[1][0], h=B[1][1]
    # Strassen multipliers:
    # M1 = (a+c)(e+g)      -> U: a+c, V: e+g
    # M2 = (a-c)(g)        -> U: a-c, V: g
    # M3 = (a+b)(e)        -> U: a+b, V: e
    # M4 = (c-b)(e+g)      -> U: c-b, V: e+g
    # M5 = (a+c)(h)        -> U: a+c, V: h
    # M6 = (b-a)(e+g)      -> U: b-a, V: e+g
    # M7 = (c-b)(h)        -> U: c-b, V: h
    
    # C[0][0] = M1 - M2 + M3 - M6
    # C[0][1] = M5 - M3 + M7 - M6  ... let me verify
    # C[1][0] = M2 + M4
    # C[1][1] = M1 - M5 + M4 + M7
    
    # Actually let me use the standard Strassen:
    # C[0][0] = M1 - M2 + M3 - M6
    # C[0][1] = M5 - M3 + M7 - M6
    # C[1][0] = M2 + M4
    # C[1][1] = M1 - M5 + M4 + M7
    
    # For block 2 (B columns 2,3): i=B[0][2], j=B[0][3], k=B[1][2], l=B[1][3]
    # M8  = (a+c)(i+k)
    # M9  = (a-c)(k)
    # M10 = (a+b)(i)
    # M11 = (c-b)(i+k)
    # M12 = (a+c)(l)
    # M13 = (b-a)(i+k)
    # M14 = (c-b)(l)
    
    # C[0][2] = M8 - M9 + M10 - M13
    # C[0][3] = M12 - M10 + M14 - M13
    # C[1][2] = M9 + M11
    # C[1][3] = M8 - M12 + M11 + M14
    
    # Define the 14 multipliers
    # col 0: M1
    U[0][0] = 1  # a
    U[2][0] = 1  # c
    V[0][0] = 1  # e
    V[4][0] = 1  # g
    # W for C[0][0] += 1*M1, C[0][1] -= 1*M1, C[1][0] -= 1*M1, C[1][1] += 1*M1
    W[0][0] = 1   # C[0][0]
    W[2][0] = -1  # C[0][1]
    W[1][0] = -1  # C[1][0]
    W[3][0] = 1   # C[1][1]
    
    # col 1: M2
    U[0][1] = 1  # a
    U[2][1] = -1 # c
    V[4][1] = 1  # g
    # W: C[0][0] -= 1, C[1][0] += 1
    W[0][1] = -1
    W[1][1] = 1
    
    # col 2: M3
    U[0][2] = 1  # a
    U[1][2] = 1  # b
    V[0][2] = 1  # e
    # W: C[0][0] += 1, C[0][1] -= 1
    W[0][2] = 1
    W[2][2] = -1
    
    # col 3: M4
    U[2][3] = 1  # c
    U[1][3] = -1 # b
    V[0][3] = 1  # e
    V[4][3] = 1  # g
    # W: C[1][0] += 1, C[1][1] += 1
    W[1][3] = 1
    W[3][3] = 1
    
    # col 4: M5
    U[0][4] = 1  # a
    U[2][4] = 1  # c
    V[5][4] = 1  # h
    # W: C[0][1] += 1, C[1][1] -= 1
    W[2][4] = 1
    W[3][4] = -1
    
    # col 5: M6
    U[1][5] = 1  # b
    U[0][5] = -1 # a
    V[0][5] = 1  # e
    V[4][5] = 1  # g
    # W: C[0][0] -= 1, C[0][1] -= 1
    W[0][5] = -1
    W[2][5] = -1
    
    # col 6: M7
    U[2][6] = 1  # c
    U[1][6] = -1 # b
    V[5][6] = 1  # h
    # W: C[0][1] += 1, C[1][1] += 1
    W[2][6] = 1
    W[3][6] = 1
    
    # col 7: M8
    U[0][7] = 1  # a
    U[2][7] = 1  # c
    V[2][7] = 1  # i
    V[6][7] = 1  # k
    # W: C[0][2] += 1, C[0][3] -= 1, C[1][2] -= 1, C[1][3] += 1
    W[4][7] = 1   # C[0][2]
    W[6][7] = -1  # C[0][3]
    W[5][7] = -1  # C[1][2]
    W[7][7] = 1   # C[1][3]
    
    # col 8: M9
    U[0][8] = 1  # a
    U[2][8] = -1 # c
    V[6][8] = 1  # k
    # W: C[0][2] -= 1, C[1][2] += 1
    W[4][8] = -1
    W[5][8] = 1
    
    # col 9: M10
    U[0][9] = 1  # a
    U[1][9] = 1  # b
    V[2][9] = 1  # i
    # W: C[0][2] += 1, C[0][3] -= 1
    W[4][9] = 1
    W[6][9] = -1
    
    # col 10: M11
    U[2][10] = 1  # c
    U[1][10] = -1 # b
    V[2][10] = 1  # i
    V[6][10] = 1  # k
    # W: C[1][2] += 1, C[1][3] += 1
    W[5][10] = 1
    W[7][10] = 1
    
    # col 11: M12
    U[0][11] = 1  # a
    U[2][11] = 1  # c
    V[7][11] = 1  # l
    # W: C[0][3] += 1, C[1][3] -= 1
    W[6][11] = 1
    W[7][11] = -1
    
    # col 12: M13
    U[1][12] = 1  # b
    U[0][12] = -1 # a
    V[2][12] = 1  # i
    V[6][12] = 1  # k
    # W: C[0][2] -= 1, C[0][3] -= 1
    W[4][12] = -1
    W[6][12] = -1
    
    # col 13: M14
    U[2][13] = 1  # c
    U[1][13] = -1 # b
    V[7][13] = 1  # l
    # W: C[0][3] += 1, C[1][3] += 1
    W[6][13] = 1
    W[7][13] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
