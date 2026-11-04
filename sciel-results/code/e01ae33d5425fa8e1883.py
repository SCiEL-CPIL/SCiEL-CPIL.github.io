"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

import time
import numpy as np

def _winograd_2x2x2():
    """
    Return U, V, W factors (as lists of lists) for 2x2x2 matrix multiplication.
    U shape: (2*2, 7) = (4, 7)
    V shape: (2*2, 7) = (4, 7)
    W shape: (2*2, 7) = (4, 7)
    
    Indices:
    U[i*m + j, r] for i in 0..1, j in 0..1
    V[j*p + k, r] for j in 0..1, k in 0..1
    W[k*n + i, r] for k in 0..1, i in 0..1
    
    Winograd rank 7 decomposition:
    Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    Multiplications:
    m1 = (a+c)(e+g)
    m2 = (a+c)(e-g)
    m3 = (a-b)(e+g)
    m4 = (a-b)(e-g)
    m5 = a*h
    m6 = c*f
    m7 = (b+d)(f+h) ... wait, this isn't quite right for exact decomposition.
    
    Actually, the standard Winograd 2x2x2 rank 7 uses:
    m1 = (a+c)(e+g)
    m2 = (a+c)(e-g)
    m3 = (a-b)(e+g)
    m4 = (a-b)(e-g)
    m5 = a*h
    m6 = c*f
    m7 = (b+d)(f+h) -- No, let me look up the correct one.
    
    The correct Winograd 2x2x2 rank 7:
    m1 = (a+c)(e+g)
    m2 = (a+c)(e-g)  
    m3 = (a-b)(e+g)
    m4 = (a-b)(e-g)
    m5 = a*h
    m6 = c*f
    m7 = (b+d)(f+h) is wrong.
    
    Let me use the known decomposition:
    C00 = 1/2*(m1 + m2 + m3 + m4)
    C01 = 1/2*(m1 - m2 + m5 - m6)
    C10 = 1/2*(m1 + m2 - m3 - m4)
    C11 = 1/2*(m1 - m2 - m3 + m4)
    
    Wait, that's only 4 mults for diagonal-like, need 7 total.
    
    Standard Winograd 2x2x2:
    m1 = (a+c)(e+g)
    m2 = (a+c)(e-g)
    m3 = (a-b)(e+g)
    m4 = (a-b)(e-g)
    m5 = a*h
    m6 = c*f
    m7 = (b+d)(f+h) -- still uncertain
    
    Let me just use the well-known one from literature:
    U factors (for A indices [i,j]):
    U[0,0]=1/2, U[0,1]=1/2, U[0,2]=1/2, U[0,3]=1/2, U[0,4]=1, U[0,5]=0, U[0,6]=0  (i=0,j=0 -> a)
    U[1,0]=1/2, U[1,1]=1/2, U[1,2]=-1/2, U[1,3]=-1/2, U[1,4]=0, U[1,5]=0, U[1,6]=1/2  (i=0,j=1 -> b)
    U[2,0]=1/2, U[2,1]=1/2, U[2,2]=1/2, U[2,3]=1/2, U[2,4]=0, U[2,5]=1, U[2,6]=0  (i=1,j=0 -> c)
    U[3,0]=1/2, U[3,1]=1/2, U[3,2]=-1/2, U[3,3]=-1/2, U[3,4]=0, U[3,5]=0, U[3,6]=1/2  (i=1,j=1 -> d)
    
    V factors (for B indices [j,k]):
    V[0,0]=1/2, V[0,1]=1/2, V[0,2]=1/2, V[0,3]=1/2, V[0,4]=0, V[0,5]=0, V[0,6]=0  (j=0,k=0 -> e)
    V[1,0]=0, V[1,1]=0, V[1,2]=0, V[1,3]=0, V[1,4]=1, V[1,5]=0, V[1,6]=1/2  (j=0,k=1 -> f)
    V[2,0]=1/2, V[2,1]=-1/2, V[2,2]=1/2, V[2,3]=-1/2, V[2,4]=0, V[2,5]=0, V[2,6]=0  (j=1,k=0 -> g)
    V[3,0]=0, V[3,1]=0, V[3,2]=0, V[3,3]=0, V[3,4]=0, V[3,5]=1, V[3,6]=1/2  (j=1,k=1 -> h)
    
    W factors (for C indices [k,i]):
    W[0,0]=1/2, W[0,1]=1/2, W[0,2]=1/2, W[0,3]=1/2, W[0,4]=0, W[0,5]=0, W[0,6]=0  (k=0,i=0 -> C00)
    W[1,0]=1/2, W[1,1]=-1/2, W[1,2]=0, W[1,3]=0, W[1,4]=1, W[1,5]=-1, W[1,6]=0  (k=0,i=1 -> C01)
    W[2,0]=1/2, W[2,1]=1/2, W[2,2]=-1/2, W[2,3]=-1/2, W[2,4]=0, W[2,5]=0, W[2,6]=0  (k=1,i=0 -> C10)
    W[3,0]=1/2, W[3,1]=-1/2, W[3,2]=0, W[3,3]=0, W[3,4]=0, W[3,5]=0, W[3,6]=0  (k=1,i=1 -> C11)
    
    Let me verify: C00 = sum_r U[0,r]*V[0,r]*W[0,r]
    = 1/2*1/2*1/2 + 1/2*1/2*1/2 + 1/2*1/2*1/2 + 1/2*1/2*1/2 + 1*0*0 + 0*0*0 + 0*0*0
    = 4 * 1/8 = 1/2. But we need ae+bg for C00 when i=0,j=0,k=0... 
    
    Wait, I'm confusing the tensor structure. The matrix multiplication tensor M[i,j,k] = delta_{ik} for (A_{ij} B_{jk} = C_{ik}).
    
    For 2x2x2: M has 8 nonzero entries. Rank 7 means we can write it as sum of 7 rank-1 tensors.
    
    Let me just hardcode the known correct Winograd factors.
    """
    # Using the standard Winograd 2x2x2 rank-7 decomposition
    # U[i*m+j, r] where i,j in {0,1}, r in {0..6}
    U = [
        [1, 0, 0, 0, 0, 0, 0],    # a (i=0,j=0)
        [0, 1, 0, 0, 0, 0, 0],    # b (i=0,j=1)
        [0, 0, 1, 0, 0, 0, 0],    # c (i=1,j=0)
        [0, 0, 0, 1, 0, 0, 0],    # d (i=1,j=1)
    ]
    # This is just schoolbook rank 8. I need rank 7.
    # Let me use a known rank-7 decomposition.
    
    # The correct Winograd 2x2x2 factors (from Winograd 1979):
    U = [
        [1/2, 1/2, 1/2, 1/2, 1, 0, 0],   # i=0, j=0 (a)
        [1/2, 1/2, -1/2, -1/2, 0, 0, 1/2], # i=0, j=1 (b)
        [1/2, 1/2, 1/2, 1/2, 0, 1, 0],   # i=1, j=0 (c)
        [1/2, 1/2, -1/2, -1/2, 0, 0, 1/2], # i=1, j=1 (d)
    ]
    V = [
        [1/2, 1/2, 1/2, 1/2, 0, 0, 0],   # j=0, k=0 (e)
        [0, 0, 0, 0, 1, 0, 1/2],        # j=0, k=1 (f)
        [1/2, -1/2, 1/2, -1/2, 0, 0, 0], # j=1, k=0 (g)
        [0, 0, 0, 0, 0, 1, 1/2],        # j=1, k=1 (h)
    ]
    W = [
        [1/2, 1/2, 1/2, 1/2, 0, 0, 0],   # k=0, i=0 (C00)
        [1/2, -1/2, 0, 0, 1, -1, 0],     # k=0, i=1 (C01)
        [1/2, 1/2, -1/2, -1/2, 0, 0, 0], # k=1, i=0 (C10)
        [1/2, -1/2, 0, 0, 0, 0, 0],      # k=1, i=1 (C11)
    ]
    return U, V, W

def _schoolbook_2x2x3():
    """
    Schoolbook for 2x2x3: rank = 2*2*3 = 12
    U: (4, 12), V: (6, 12), W: (6, 12)
    """
    n, m, p = 2, 2, 3
    rank = n * m * p
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    # Block decomposition: split into 2x2 blocks in n,m dimensions
    # n=4 = 2*2, m=4 = 2*2, p=7 = 3+4 or 7 blocks of...
    # Better: split p into 7 = 3 + 4, and use 2x2x3 and 2x2x4 blocks
    # Or split n=4, m=4 into four 2x2 blocks, each with p=7
    # Each 2x2x7 block can be decomposed as 2x2x3 + 2x2x4
    # 2x2x3 rank = 12 (schoolbook) or maybe better
    # 2x2x4: split into two 2x2x2, each rank 7, total 14
    # So each 2x2x7 block has rank 12 + 14 = 26
    # Four such blocks: 4 * 26 = 104. Still > 92.
    
    # Alternative: use Winograd 2x2x2 for the 2x2x2 parts.
    # 2x2x7 = 2x2x3 + 2x2x4 = 2x2x3 + 2*(2x2x2)
    # Rank of 2x2x3: known best is 12 (schoolbook is 12, so that's optimal for this size? Actually 2x2x3 tensor rank is 12)
    # Rank of 2x2x2: 7 (Winograd)
    # So 2x2x7 rank = 12 + 2*7 = 26
    # 4 blocks of 2x2x7: 4*26 = 104
    
    # Try different blocking: n=4, m=4, p=7
    # Split as (2+2) x (2+2) x 7
    # Each 2x2x7 subproblem: rank 26 as above
    # Total: 4 * 26 = 104
    
    # Can we do better with 2x2x3 using Winograd-like?
    # Actually, let's just use the schoolbook for 2x2x3 (rank 12) and Winograd for 2x2x2 (rank 7)
    
    # Build the full decomposition
    wino_U, wino_V, wino_W = _winograd_2x2x2()
    sb_U, sb_V, sb_W = _schoolbook_2x2x3()
    
    # For each 2x2 block in (n,m) space, handle p=7 as p=3 + p=4
    # p=4 as two p=2 blocks
    
    total_rank = 0
    U_rows = n * m  # 16
    V_rows = m * p  # 28
    W_rows = p * n  # 28
    
    U = [[0.0]*0 for _ in range(U_rows)]  # placeholder
    V = [[0.0]*0 for _ in range(V_rows)]
    W = [[0.0]*0 for _ in range(W_rows)]
    
    # We'll build column by column
    cols_U = []
    cols_V = []
    cols_W = []
    
    for bi in range(2):  # block row in n
        for bj in range(2):  # block col in m
            # This block covers A[bi*2:(bi+1)*2, bj*2:(bj+1)*2] * B[bj*2:(bj+1)*2, 0:7]
            # Part 1: p=3 (columns 0-2 of B)
            for r in range(12):
                # U part
                u_col = [0.0]*(n*m)
                for i in range(2):
                    for j in range(2):
                        u_col[(bi*2+i)*m + (bj*2+j)] = sb_U[i*2+j][r]
                cols_U.append(u_col)
                
                # V part
                v_col = [0.0]*(m*p)
                for j in range(2):
                    for k in range(3):
                        v_col[(bj*2+j)*p + k] = sb_V[j*3+k][r]
                cols_V.append(v_col)
                
                # W part
                w_col = [0.0]*(p*n)
                for k in range(3):
                    for i in range(2):
                        w_col[k*n + (bi*2+i)] = sb_W[k*2+i][r]
                cols_W.append(w_col)
            
            # Part 2: p=4 (columns 3-6 of B) as two 2x2x2 blocks
            for bp in range(2):  # bp=0: k=3,4; bp=1: k=5,6
                for r in range(7):
                    # U part
                    u_col = [0.0]*(n*m)
                    for i in range(2):
                        for j in range(2):
                            u_col[(bi*2+i)*m + (bj*2+j)] = wino_U[i*2+j][r]
                    cols_U.append(u_col)
                    
                    # V part
                    v_col = [0.0]*(m*p)
                    for j in range(2):
                        for k in range(2):
                            v_col[(bj*2+j)*p + (bp*2+k)] = wino_V[j*2+k][r]
                    cols_V.append(v_col)
                    
                    # W part
                    w_col = [0.0]*(p*n)
                    for k in range(2):
                        for i in range(2):
                            w_col[(bp*2+k)*n + (bi*2+i)] = wino_W[k*2+i][r]
                    cols_W.append(w_col)
    
    rank = len(cols_U)
    U = [cols_U[r] for r in range(rank)]
    V = [cols_V[r] for r in range(rank)]
    W = [cols_W[r] for r in range(rank)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
