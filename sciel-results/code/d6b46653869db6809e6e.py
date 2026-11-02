"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    # We'll build a rank-55 decomposition using Strassen-like identities on the 2x2 sub-blocks
    # Strategy: For the 2x3 * 3x10, split middle dim into 2+1.
    # The 2x2 part (A rows 0,1 cols 0,1 * B rows 0,1 cols 0..9) uses Strassen per column pair.
    # Actually, let's use a known construction.
    
    # Schoolbook gives rank 60. We want to find a rank-55 construction.
    # 
    # Key idea: For 2x3 * 3x10, we can group the 10 columns of B into 5 pairs.
    # For each pair of columns (k, k+1), the 2x3 * 3x2 multiplication can be done
    # with fewer than 12 scalar multiplications using bilinear identities.
    #
    # Actually, let me use a simpler approach: direct rank minimization via 
    # structured decomposition.
    #
    # For 2x3 * 3x1: 6 mults schoolbook. Can we do better? No, it's rank 6 (it's 
    # a 2x3x1 tensor which is just 6 independent terms).
    #
    # For 2x2 * 2x2: Strassen does 7 instead of 4.
    #
    # Let me think of 2x3 * 3x10 as: 
    #   C[i,k] = sum_j A[i,j] B[j,k]
    #   = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # Group j=0,1 together: use Strassen-like for the 2x2 block A[i,0:2] * B[0:2,k]
    # But that's 2x2 * 2x1, not 2x2*2x2.
    #
    # Alternative: view as 2x(2+1) * (2+1)x10. 
    # The 2x2 * 2x10 part: for each column pair in B, use Strassen.
    # 
    # Let me just try: use the fact that we can write the product using 55 terms
    # by exploiting that some combinations are redundant.
    
    # Let me try a different structural approach:
    # 2x3 * 3x10 = [2x2 | 2x1] * [[2x10], [1x10]]
    # = 2x2 * 2x10 + 2x1 * 1x10
    # 
    # 2x2 * 2x10: This is like 5 independent 2x2*2x2 problems (grouping cols in pairs)
    #   Each 2x2*2x2 uses Strassen rank 7, so 5*7 = 35
    #   But the last group is 2x2*2x0 if 10 is even... 10/2 = 5 pairs, so 5 * 7 = 35
    # 2x1 * 1x10: This is 2*10 = 20 scalar mults
    # Total: 35 + 20 = 55!
    
    # So the construction is:
    # - Split B's 10 columns into 5 pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
    # - For each pair (k1, k2), compute A[:,0:2] * B[0:2, [k1,k2]] using Strassen (7 mults)
    # - Compute A[:,2] * B[2, :] using 20 scalar mults (2 rows * 10 cols)
    # Total rank: 5*7 + 20 = 55
    
    # Strassen for 2x2 * 2x2:
    # Given P = [[a,b],[c,d]] * [[e,f],[g,h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e  
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # P[0,0] = m1 + m4 - m5 + m7
    # P[0,1] = m3 + m5
    # P[1,0] = m2 + m4
    # P[1,1] = m1 - m2 + m3 + m6
    
    # For our case, for each column pair (k1, k2):
    # A_sub = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]  (2x2)
    # B_sub = [[B[0,k1], B[0,k2]], [B[1,k1], B[1,k2]]]  (2x2)
    # Result: 2x2 matrix added to C[:, [k1,k2]]
    
    # The 7 Strassen mults for pair t (columns 2t, 2t+1):
    # m1_t = (A[0,0]+A[1,0]) * (B[0,2t]+B[1,2t])  -- wait, this isn't quite right
    # 
    # Actually Strassen is for P = X*Y where X,Y are 2x2.
    # X = A[:, 0:2] = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # Y = B[0:2, [2t, 2t+1]] = [[B[0,2t], B[0,2t+1]], [B[1,2t], B[1,2t+1]]]
    #
    # m1 = (X[0,0]+X[1,1]) * (Y[0,0]+Y[1,1]) = (A[0,0]+A[1,1]) * (B[0,2t]+B[1,2t+1])
    # m2 = (X[1,0]+X[1,1]) * Y[0,0] = (A[1,0]+A[1,1]) * B[0,2t]
    # m3 = X[0,0] * (Y[0,1]-Y[1,1]) = A[0,0] * (B[0,2t+1]-B[1,2t+1])
    # m4 = X[1,1] * (Y[1,0]-Y[0,0]) = A[1,1] * (B[1,2t]-B[0,2t])
    # m5 = (X[0,0]+X[0,1]) * Y[1,1] = (A[0,0]+A[0,1]) * B[1,2t+1]
    # m6 = (X[1,0]-X[0,0]) * (Y[0,0]+Y[0,1]) = (A[1,0]-A[0,0]) * (B[0,2t]+B[0,2t+1])
    # m7 = (X[0,1]-X[1,1]) * (Y[1,0]+Y[1,1]) = (A[0,1]-A[1,1]) * (B[1,2t]+B[1,2t+1])
    #
    # Result[0,0] = m1 + m4 - m5 + m7  -> C[0, 2t]
    # Result[0,1] = m3 + m5           -> C[0, 2t+1]
    # Result[1,0] = m2 + m4           -> C[1, 2t]
    # Result[1,1] = m1 - m2 + m3 + m6 -> C[1, 2t+1]
    #
    # Plus for each k: C[i,k] += A[i,2] * B[2,k]
    
    rank = 55
    U = [[0]*rank for _ in range(n*m)]  # 6 x 55
    V = [[0]*rank for _ in range(m*p)]  # 30 x 55
    W = [[0]*rank for _ in range(p*n)]  # 20 x 55
    
    col = 0
    # For each of the 5 column pairs, add 7 Strassen terms
    for t in range(5):
        k1 = 2*t
        k2 = 2*t+1
        # m1: (A[0,0]+A[1,1]) * (B[0,k1]+B[1,k2])
        # U entries: A[0,0] and A[1,1]
        # V entries: B[0,k1] and B[1,k2]
        U[0*m+0][col] = 1   # A[0,0]
        U[1*m+1][col] = 1   # A[1,1]
        V[0*p+k1][col] = 1  # B[0,k1]
        V[1*p+k2][col] = 1  # B[1,k2]
        # W: C[0,k1] += 1, C[1,k1] += 1, C[1,k2] += 1
        W[k1*n+0][col] = 1  # C[0,k1]
        W[k1*n+1][col] = 1  # C[1,k1]
        W[k2*n+1][col] = 1  # C[1,k2]
        col += 1
        
        # m2: (A[1,0]+A[1,1]) * B[0,k1]
        U[1*m+0][col] = 1   # A[1,0]
        U[1*m+1][col] = 1   # A[1,1]
        V[0*p+k1][col] = 1  # B[0,k1]
        # W: C[1,k1] += 1
        W[k1*n+1][col] = 1
        col += 1
        
        # m3: A[0,0] * (B[0,k2]-B[1,k2])
        U[0*m+0][col] = 1   # A[0,0]
        V[0*p+k2][col] = 1  # B[0,k2]
        V[1*p+k2][col] = -1 # B[1,k2]
        # W: C[0,k2] += 1, C[1,k2] += 1
        W[k2*n+0][col] = 1
        W[k2*n+1][col] = 1
        col += 1
        
        # m4: A[1,1] * (B[1,k1]-B[0,k1])
        U[1*m+1][col] = 1   # A[1,1]
        V[1*p+k1][col] = 1  # B[1,k1]
        V[0*p+k1][col] = -1 # B[0,k1]
        # W: C[0,k1] += 1, C[1,k1] += 1
        W[k1*n+0][col] = 1
        W[k1*n+1][col] = 1
        col += 1
        
        # m5: (A[0,0]+A[0,1]) * B[1,k2]
        U[0*m+0][col] = 1   # A[0,0]
        U[0*m+1][col] = 1   # A[0,1]
        V[1*p+k2][col] = 1  # B[1,k2]
        # W: C[0,k1] -= 1, C[0,k2] += 1
        W[k1*n+0][col] = -1
        W[k2*n+0][col] = 1
        col += 1
        
        # m6: (A[1,0]-A[0,0]) * (B[0,k1]+B[0,k2])
        U[1*m+0][col] = 1   # A[1,0]
        U[0*m+0][col] = -1  # A[0,0]
        V[0*p+k1][col] = 1  # B[0,k1]
        V[0*p+k2][col] = 1  # B[0,k2]
        # W: C[1,k2] += 1
        W[k2*n+1][col] = 1
        col += 1
        
        # m7: (A[0,1]-A[1,1]) * (B[1,k1]+B[1,k2])
        U[0*m+1][col] = 1   # A[0,1]
        U[1*m+1][col] = -1  # A[1,1]
        V[1*p+k1][col] = 1  # B[1,k1]
        V[1*p+k2][col] = 1  # B[1,k2]
        # W: C[0,k1] += 1
        W[k1*n+0][col] = 1
        col += 1
    
    # Now add the 20 scalar mults for A[:,2] * B[2,:]
    for i in range(2):
        for k in range(10):
            U[i*m+2][col] = 1   # A[i,2]
            V[2*p+k][col] = 1   # B[2,k]
            W[k*n+i][col] = 1   # C[i,k]
            col += 1
    
    assert col == rank, f"Column count {col} != rank {rank}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
