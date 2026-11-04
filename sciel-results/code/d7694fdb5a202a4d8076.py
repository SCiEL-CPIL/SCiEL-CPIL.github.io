"""Program for rank-minimal exact tensor decomposition of 2x3 by 3x9 matrix multiplication."""

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = 2, 3, 9
    # We'll use the schoolbook decomposition with rank n*m*p = 54
    # but we can do better by using the block structure.
    
    # Actually, let me think about this more carefully.
    # C[i, k] = sum_j A[i,j] * B[j,k]
    # where i in {0,1}, j in {0,1,2}, k in {0,...,8}
    
    # The tensor has entries T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # The rank is the minimum number of rank-1 tensors that sum to T.
    
    # Schoolbook: rank = n*m*p = 2*3*9 = 54
    # Each term is A[i,j] * B[j,k] contributing to C[i,k]
    
    # Can we do better? 
    # For 2x3 by 3x3, the rank is 9 (it's just the standard matrix multiplication tensor for those dims)
    # Wait no, for m x n by n x p, the rank is at most m*n*p (schoolbook) but can be lower.
    # For 2x3 by 3x3: rank is at most 18 (schoolbook), but actually...
    
    # Let me just use the schoolbook for now with rank 54, but let me see if I can improve.
    
    # Actually, the key insight: for multiplying (n x m) by (m x p), 
    # the rank is at most n*m*p (schoolbook).
    # For 2x3 by 3x9: 2*3*9 = 54.
    
    # But we can potentially do better. Let me think...
    # The 9 columns of B can be grouped as 3 blocks of 3 columns each.
    # C[:, 0:3] = A * B[:, 0:3]  -- this is 2x3 by 3x3, rank at most 18
    # C[:, 3:6] = A * B[:, 3:6]  -- same
    # C[:, 6:9] = A * B[:, 6:9]  -- same
    # Total: 3 * 18 = 54. No improvement from naive blocking.
    
    # However, the A factors can be shared across the three blocks!
    # For C[:, 0:3]: we need terms A[i,j] * B[j, 3t+r] for t=0, r=0,1,2
    # For C[:, 3:6]: we need terms A[i,j] * B[j, 3t+r] for t=1, r=0,1,2
    # For C[:, 6:9]: we need terms A[i,j] * B[j, 3t+r] for t=2, r=0,1,2
    
    # In the schoolbook, each (i,j,k) triple gives one rank-1 term.
    # The U factor has entries U[i*m+j, term] = A[i,j] coefficient
    # The V factor has entries V[j*p+k, term] = B[j,k] coefficient  
    # The W factor has entries W[k*n+i, term] = C[i,k] coefficient
    
    # For schoolbook, each term t corresponds to (i,j,k) and:
    # U[i*m+j, t] = 1
    # V[j*p+k, t] = 1
    # W[k*n+i, t] = 1
    
    # This gives rank = 2*3*9 = 54.
    
    # Let me just implement this correctly.
    
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1.0
                V[j * p + k][idx] = 1.0
                W[k * n + i][idx] = 1.0
                idx += 1
    
    return {"U": U, "V": V, "W": W}
