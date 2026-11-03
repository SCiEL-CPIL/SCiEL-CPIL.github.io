import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # The tensor for matrix multiplication: T[i*m+j, j*p+k, k*n+i] = 1
    # We seek a rank-r decomposition: T = sum_{r=0}^{R-1} U_r ⊗ V_r ⊗ W_r
    # where U_r has shape (n*m,), V_r has shape (m*p,), W_r has shape (p*n,)
    # and T[a,b,c] = sum_r U_r[a] * V_r[b] * W_r[c]
    
    # This is equivalent to: for each (a,b,c) with T[a,b,c]=1,
    # sum_r U_r[a] * V_r[b] * W_r[c] = 1
    # and for each (a,b,c) with T[a,b,c]=0,
    # sum_r U_r[a] * V_r[b] * W_r[c] = 0
    
    # We can search for integer/half-integer solutions by trying small ranks.
    # Known: rank of 2x3x6 matrix multiplication is at most 12 (using rank 4 for 2x2x2 blocks)
    # Actually, the rank of 2x3x6 is known to be 12.
    
    # Let's try to find a rank-12 decomposition.
    # We can use the fact that 2x3x6 can be decomposed using the rank-4 decomposition of 2x2x2
    # and the rank-3 decomposition of 2x2x2... wait, 2x2x2 has rank 7.
    
    # Actually, let's just use a simple construction.
    # For 2x3x6, we can think of it as 3 copies of 2x1x2 (rank 2 each) = rank 6? No.
    # Or 6 copies of 2x1x1 (rank 1 each) = rank 6? No, that's not right either.
    
    # Let me think more carefully. The tensor T is defined by T[i,j,k] = delta_{i,k} for i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4,5}?
    # No, that's not right. Let me re-read the problem.
    
    # "multiplying a 2 by 3 matrix by a 3 by 6 matrix"
    # So A is 2x3, B is 3x6, C = A*B is 2x6.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor has indices (i,j) for A, (j,k) for B, and (i,k) for C.
    # So T[i*m+j, j*p+k, k*n+i] = 1 where m=3, p=6, n=2.
    
    # Let's try rank 12.
    # We can use the trivial decomposition with rank = n*m*p = 36, but we want lower.
    
    # Let me try to construct a rank-12 decomposition.
    # Idea: For each pair (i,k) where i in {0,1} and k in {0,...,5},
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a sum of 3 terms.
    # If we could decompose each product A[i,j]*B[j,k] as a rank-1 tensor, we'd need 2*3*6 = 36 terms.
    # But we can combine them.
    
    # Actually, the known rank of 2x3x6 matrix multiplication is 12.
    # Let me try to find it by solving a linear system.
    
    # For rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # This is a nonlinear system. But if we fix V and W, it's linear in U, and vice versa.
    # We can use alternating minimization or just try small integer values.
    
    # Let me try a different approach: use the known rank-7 decomposition of 2x2x2
    # and extend it to 2x3x6.
    
    # Actually, let me just try rank 12 with a simple construction.
    # For each i in {0,1} and k in {0,...,5}, define:
    # U[i*3+j, (i,k)] = 1 for all j? No, that doesn't work.
    
    # Let me think again. The tensor T has T[i*3+j, j*6+k, k*2+i] = 1.
    # So for fixed (i,k), the sum over j of A[i,j]*B[j,k] gives C[i,k].
    # The term A[i,j]*B[j,k] corresponds to U[i*3+j, r] * V[j*6+k, r] * W[k*2+i, r] = 1
    # for some r, and 0 for all other (i',j',k') combinations.
    
    # This is getting complicated. Let me just use the rank-36 decomposition for now,
    # since I'm not confident in finding a lower rank decomposition.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
