import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # We'll build the decomposition and verify it
    
    # Strategy: A is 2x3, B is 3x5
    # Split A = [A1 | a] where A1 is 2x2 (cols 0,1) and a is 2x1 (col 2)
    # Split B = [B1; b] where B1 is 2x5 (rows 0,1) and b is 1x5 (row 2)
    # C = A1 @ B1 + a @ b (outer product)
    
    # a @ b contributes 2*5 = 10 terms
    # A1 @ B1: 2x2 @ 2x5
    
    # For 2x2 @ 2x5, use Strassen on overlapping 2x2 blocks
    # B1 columns: [c0, c1, c2, c3, c4]
    # We can group as [c0,c1], [c2,c3], [c4]
    # Strassen for 2x2@2x2 on [c0,c1]: 7 terms
    # Strassen for 2x2@2x2 on [c2,c3]: 7 terms  
    # Direct for 2x2@2x1 on [c4]: 4 terms
    # Total for A1@B1: 18, plus 10 for a@b = 28
    
    # Try to reduce: can we share Strassen products between the two 2x2 blocks?
    # The Strassen products for block [c0,c1] use:
    #   M1=(a+b)(c0+c1), M2=(c+d)c0, M3=a(c1-c0), M4=d(c0-c1) wait let me be careful
    
    # Let A1 = [[a,b],[c,d]]
    # For columns [x,y] of B1:
    # Strassen:
    #   P1 = (a+b)(x+y)
    #   P2 = (c+d)x  
    #   P3 = a(y-x)
    #   P4 = d(x-y)  # wait, standard Strassen for 2x2:
    #   Actually standard Strassen for [[a,b],[c,d]] @ [[x,y],[z,w]]:
    #   P1=(a+b)(x+z)
    #   P2=(c+d)x
    #   P3=a(w-z)
    #   P4=d(y-w)  # hmm let me just use the known 7
    #   P5=(a-b)(w+z)
    #   P6=(c+a)y
    #   P7=(b+d)z
    #   Result: [[P1-P4+P7-P5, P5+P4], [P2+P5, P6-P2+P3-P7]] wait this is getting confusing.
    
    # Let me just use a verified Strassen and try to find if rank 27 is possible.
    # For now, let me construct rank 28 correctly and verify.
    
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0.0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0.0] * rank for _ in range(p * n)]  # 10 x 28
    
    r = 0
    
    # Part 1: a @ b (outer product) - 10 terms
    # a = [A[0,2], A[1,2]], b = [B[2,0], B[2,1], B[2,2], B[2,3], B[2,4]]
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1.0
            V[2 * p + k][r] = 1.0
            W[k * n + i][r] = 1.0
            r += 1
    # r = 10
    
    # Part 2: A1 @ B1 using Strassen on column pairs
    # A1 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]] = [[a, b], [c, d]]
    # B1 = [[B[0,0], B[0,1], B[0,2], B[0,3], B[0,4]],
    #       [B[1,0], B[1,1], B[1,2], B[1,3], B[1,4]]]
    
    # For columns 0,1 (B1[:,0:2]):
    # x=B[0,0], y=B[1,0], z=B[0,1], w=B[1,1]
    # Strassen 7 products:
    # P1 = (a+b)(x+z)
    U[0*m+0][r] = 1.0; U[0*m+1][r] = 1.0
    V[0*p+0][r] = 1.0; V[0*p+1][r] = 1.0
    W[0*n+0][r] = 1.0; W[0*n+1][r] = 1.0; W[1*n+0][r] = 1.0; W[1*n+1][r] = 1.0
    r += 1  # P1
    
    # P2 = (c+d)x
    U[1*m+0][r] = 1.0; U[1*m+1][r] = 1.0
    V[0*p+0][r] = 1.0
    W[0*n+1][r] = 1.0; W[1*n+1][r] = 1.0
    r += 1  # P2
    
    # P3 = a(w-z)
    U[0*m+0][r] = 1.0
    V[1*p+1][r] = 1.0; V[0*p+1][r] = -1.0
    W[0*n+0][r] = 1.0; W[1*n+0][r] = 1.0
    r += 1  # P3
    
    # P4 = d(y-w)
    U[1*m+1][r] = 1.0
    V[1*p+0][r] = 1.0; V[1*p+1][r] = -1.0
    W[0*n+1][r] = 1.0; W[1*n+1][r] = -1.0
    r += 1  # P4
    
    # P5 = (a-b)(w+z)
    U[0*m+0][r] = 1.0; U[0*m+1][r] = -1.0
    V[1*p+1][r] = 1.0; V[0*p+1][r] = 1.0
    W[0*n+0][r] = 1.0; W[0*n+1][r] = -1.0; W[1*n+0][r] = 1.0; W[1*n+1][r] = -1.0
    r += 1  # P5
    
    # P6 = (c+a)y
    U[1*m+0][r] = 1.0; U[0*m+0][r] = 1.0
    V[1*p+0][r] = 1.0
    W[1*n+1][r] = 1.0
    r += 1  # P6
    
    # P7 = (b+d)z
    U[0*m+1][r] = 1.0; U[1*m+1][r] = 1.0
    V[0*p+1][r] = 1.0
    W[1*n+1][r] = -1.0
    r += 1  # P7
    
    # For columns 2,3 (B1[:,2:4]):
    # x=B[0,2], y=B[1,2], z=B[0,3], w=B[1,3]
    # P1 = (a+b)(x+z)
    U[0*m+0][r] = 1.0; U[0*m+1][r] = 1.0
    V[0*p+2][r] = 1.0; V[0*p+3][r] = 1.0
    W[2*n+0][r] = 1.0; W[2*n+1][r] = 1.0; W[3*n+0][r] = 1.0; W[3*n+1][r] = 1.0
    r += 1  # P1
    
    # P2 = (c+d)x
    U[1*m+0][r] = 1.0; U[1*m+1][r] = 1.0
    V[0*p+2][r] = 1.0
    W[2*n+1][r] = 1.0; W[3*n+1][r] = 1.0
    r += 1  # P2
    
    # P3 = a(w-z)
    U[0*m+0][r] = 1.0
    V[1*p+3][r] = 1.0; V[0*p+3][r] = -1.0
    W[2*n+0][r] = 1.0; W[3*n+0][r] = 1.0
    r += 1  # P3
    
    # P4 = d(y-w)
    U[1*m+1][r] = 1.0
    V[1*p+2][r] = 1.0; V[1*p+3][r] = -1.0
    W[2*n+1][r] = 1.0; W[3*n+1][r] = -1.0
    r += 1  # P4
    
    # P5 = (a-b)(w+z)
    U[0*m+0][r] = 1.0; U[0*m+1][r] = -1.0
    V[1*p+3][r] = 1.0; V[0*p+3][r] = 1.0
    W[2*n+0][r] = 1.0; W[2*n+1][r] = -1.0; W[3*n+0][r] = 1.0; W[3*n+1][r] = -1.0
    r += 1  # P5
    
    # P6 = (c+a)y
    U[1*m+0][r] = 1.0; U[0*m+0][r] = 1.0
    V[1*p+2][r] = 1.0
    W[3*n+1][r] = 1.0
    r += 1  # P6
    
    # P7 = (b+d)z
    U[0*m+1][r] = 1.0; U[1*m+1][r] = 1.0
    V[0*p+3][r] = 1.0
    W[3*n+1][r] = -1.0
    r += 1  # P7
    
    # For column 4 (B1[:,4]): direct 4 terms
    # A1 @ [B[0,4], B[1,4]]^T
    # C[0,4] = a*B[0,4] + b*B[1,4]
    # C[1,4] = c*B[0,4] + d*B[1,4]
    U[0*m+0][r] = 1.0
    V[0*p+4][r] = 1.0
    W[4*n+0][r] = 1.0
    r += 1
    
    U[0*m+1][r] = 1.0
    V[1*p+4][r] = 1.0
    W[4*n+0][r] = 1.0
    r += 1
    
    U[1*m+0][r] = 1.0
    V[0*p+4][r] = 1.0
    W[4*n+1][r] = 1.0
    r += 1
    
    U[1*m+1][r] = 1.0
    V[1*p+4][r] = 1.0
    W[4*n+1][r] = 1.0
    r += 1
    
    # Now verify the decomposition
    # The tensor T[i,j,k] = delta_{j,k} for matrix mult
    # Our decomposition: sum_r U[i,r] * V[j,r] * W[k,r] should equal T[i,j,k]
    # where i indexes A (n*m=6), j indexes B (m*p=15), k indexes C (p*n=10)
    
    # Let's verify numerically
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # For random A and B, check A@B == sum_r (A_flat * U[:,r]) * (B_flat * V[:,r]) * W[:,r]
    # Actually: C[k*n+i] = sum_{j=0}^{m-1} A[i*m+j] * B[j*p+k]
    # Our formula: C[k*n+i] = sum_r U[i*m, r] * V[j*p... wait
    
    # The tensor for matrix mult: T[i,j,k] = 1 if j==k... no.
    # T[i,j,k] where i=(i_A, j_A) for A element, j=(j_B, k_B) for B element, k=(k_C, i_C) for C element
    # T = 1 if j_A == j_B and i_C == i_A and k_C == k_B
    
    # In our indexing:
    # U index: i*m + j where i in [0,n), j in [0,m) -> A[i,j]
    # V index: j*p + k where j in [0,m), k in [0,p) -> B[j,k]  
    # W index: k*n + i where k in [0,p), i in [0,n) -> C[k,i]... wait C is p x n?
    
    # C = A @ B where A is n x m, B is m x p, so C is n x p.
    # But the problem says output factor follows W[k*n+i, r] convention.
    # So C is stored as k*n + i where k is column (0..p-1) and i is row (0..n-1).
    # So C[k,i] = A[i,:] @ B[:,k] = sum_j A[i,j] * B[j,k]
    
    # So W[k*n + i, r] corresponds to C[k,i]
    
    # Verification: for each (i,k), C[k,i] = sum_j A[i,j] * B[j,k]
    # Our decomposition: C[k,i] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * ... 
    # No, the tensor decomposition is: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where T[i,j,k] = 1 if this combination contributes to C.
    
    # Actually for matrix multiplication tensor:
    # T[A_idx, B_idx, C_idx] = 1 if A_idx=(i,j), B_idx=(j,k), C_idx=(k,i) in our convention
    # i.e., T[i*m+j, j*p+k, k*n+i] = 1
    
    # Let me verify by checking the tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Our decomposition gives: T_decomp[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    T_decomp = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
    
    if not np.allclose(T, T_decomp, atol=1e-10):
        # Decomposition is wrong, fall back to trivial rank 30
        rank = 30
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][r] = 1.0
                    V[j*p+k][r] = 1.0
                    W[k*n+i][r] = 1.0
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
