"""Matrix multiplication 2x4x4 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    # Try to find the minimum rank
    for target_rank in range(16, 33):
        if time.time() > deadline - 10:
            break
        result = _find_decomposition_als(n, m, p, target_rank)
        if result is not None:
            return result
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
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

def _find_decomposition_als(n, m, p, target_rank, max_iters=500, tol=1e-8):
    """Find a rank decomposition using alternating least squares."""
    # Target tensor: T[(i,j), (j,k), (k,i)] = 1 for all valid i,j,k
    # In flattened form: index1 = i*m+j, index2 = j*p+k, index3 = k*n+i
    
    # Build the target tensor as a 3D array
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Initialize U, V, W
    np.random.seed(0)
    U = np.random.randn(n*m, target_rank)
    V = np.random.randn(m*p, target_rank)
    W = np.random.randn(p*n, target_rank)
    
    # Alternating least squares
    for iteration in range(max_iters):
        # Update U: for each column c, U[:,c] should minimize ||T - sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]||
        # Fix V and W, solve for U
        # T_unfolded along mode 1: for each (a,b), T[a,b,:] = sum_c U[a,c] * (V[b,c] * W[:,c])
        # This is a linear system
        
        # Mode-1 unfolding: T1 has shape (n*m, (m*p)*(p*n))
        T1 = T.reshape(n*m, -1)
        # For each c, the contribution to T1 is U[:,c] ⊗ (V[:,c] ⊗ W[:,c])
        # So T1 = sum_c U[:,c] * (V[:,c] ⊗ W[:,c]).T
        # Let M_c = V[:,c] ⊗ W[:,c], then T1 = U @ (M_c stacked).T
        
        # Build the matrix: each column c contributes U[:,c] * (V[:,c] ⊗ W[:,c])
        # So T1 = U @ X where X[:,c] = V[:,c] ⊗ W[:,c]
        X = np.zeros((m*p*p*n, target_rank))
        for c in range(target_rank):
            X[:,c] = np.kron(V[:,c], W[:,c])
        
        # Solve U @ X = T1 => U = T1 @ pinv(X)
        try:
            U_new = T1 @ np.linalg.pinv(X)
        except:
            break
        
        # Update V: fix U and W
        T2 = T.transpose(1,0,2).reshape(m*p, -1)
        Y = np.zeros((n*m*p*n, target_rank))
        for c in range(target_rank):
            Y[:,c] = np.kron(U[:,c], W[:,c])
        try:
            V_new = T2 @ np.linalg.pinv(Y)
        except:
            break
        
        # Update W: fix U and V
        T3 = T.transpose(2,0,1).reshape(p*n, -1)
        Z = np.zeros((n*m*m*p, target_rank))
        for c in range(target_rank):
            Z[:,c] = np.kron(U[:,c], V[:,c])
        try:
            W_new = T3 @ np.linalg.pinv(Z)
        except:
            break
        
        # Check convergence
        U, V, W = U_new, V_new, W_new
        
        # Compute error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    error += (val - 1.0)**2
        if error < tol:
            # Convert to list of lists
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
        
        if time.time() > deadline - 5:
            break
    
    return None
