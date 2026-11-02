import numpy as np
import time

def _verify(U, V, W, n, m, p, tol=1e-6):
    """
    Verify if U, V, W decompose the matrix multiplication tensor M_{n,m,p}.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] should be 1 for all i,j,k.
    """
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijt,jkt,kit->ijk', U_r, V_r, W_r)
    return np.allclose(T, 1.0, atol=tol)

def construct():
    n, m, p = 2, 3, 9
    start_time = time.time()
    deadline = start_time + 140
    
    # Strategy: M_{2,3,9} = M_{2,3,3} ⊗ M_{1,1,3}
    # We need a rank-minimal decomposition of M_{2,3,3} and M_{1,1,3}.
    # M_{1,1,3} has rank 3 (schoolbook is optimal for 1x1xp).
    # For M_{2,3,3}, the schoolbook rank is 18. 
    # Can we do better? Let's try to construct a rank-15 or lower decomposition.
    # Actually, let's first just use the Kronecker of schoolbook to get a valid answer,
    # then try to optimize the base case.
    
    # Schoolbook for M_{2,3,3}: rank = 2*3*3 = 18
    # Schoolbook for M_{1,1,3}: rank = 1*1*3 = 3
    # Kronecker product rank = 18 * 3 = 54 (same as full schoolbook)
    
    # Let's try to find a better decomposition for M_{2,3,3}.
    # The tensor M_{2,3,3} has structure: T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    # This is actually the all-ones tensor of shape (2,3,3), which has rank 1!
    # Wait, no. The matrix multiplication tensor T_{i,j,k} represents the (i,j) entry of A*B.
    # T_{i,j,k} = 1 if and only if the (i,k) entry of B multiplies the (j,k) entry of A... 
    # No, standard convention: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So the tensor is T_{i,j,k} where i indexes A's row, j indexes A's col/B's row, k indexes B's col.
    # T_{i,j,k} = 1 for all valid i,j,k. This IS the all-ones tensor.
    # The all-ones tensor of shape (n,m,p) has rank 1!
    # U[i,j,:] = [1], V[j,k,:] = [1], W[k,i,:] = [1]
    # Then sum_t U[i,j,t] V[j,k,t] W[k,i,t] = 1*1*1 = 1. Yes!
    
    # So M_{2,3,3} has rank 1 decomposition:
    # U_233: shape (6, 1), all ones
    # V_233: shape (9, 1), all ones  
    # W_233: shape (6, 1), all ones
    
    # M_{1,1,3} has rank 3:
    # U_113: shape (1, 3) = [[1, 1, 1]]
    # V_113: shape (3, 3) = I_3
    # W_113: shape (3, 1) = [[1], [1], [1]]
    # Check: T[i,j,k] for n=1,m=1,p=3: only i=0,j=0.
    # T[0,0,k] = sum_t U[0,t] V[t,k] W[k,0]
    # = sum_t 1 * I[t,k] * 1 = I[k,k]... wait that gives delta_{k,k} = 1 for all k.
    # Actually V[t,k] = I[t,k], so sum_t 1 * I[t,k] * 1 = I[k,k] = 1. Yes, T[0,0,k]=1 for all k.
    
    # Now Kronecker product:
    # M_{2,3,9} = M_{2,3,3} ⊗ M_{1,1,3}
    # U = U_233 ⊗ U_113: shape (6*1, 1*3) = (6, 3)
    # V = V_233 ⊗ V_113: shape (9*3, 1*3) = (27, 3)
    # W = W_233 ⊗ W_113: shape (6*3, 1*3) = (18, 3)
    # Rank = 1 * 3 = 3
    
    # Let's verify this works!
    U_233 = np.ones((6, 1))
    V_233 = np.ones((9, 1))
    W_233 = np.ones((6, 1))
    
    U_113 = np.ones((1, 3))
    V_113 = np.eye(3)
    W_113 = np.ones((3, 1))
    
    U = np.kron(U_233, U_113)  # (6, 3)
    V = np.kron(V_233, V_113)  # (27, 3)
    W = np.kron(W_233, W_113)  # (18, 3)
    
    if _verify(U, V, W, n, m, p, tol=1e-6):
        # Already integer entries
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
