import numpy as np

def construct():
    n, m, p = 3, 3, 7
    
    # We'll try to find a low-rank decomposition
    # T[i,j,k] = 1 for all i,j,k
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Try ranks from 20 to 63
    for rank in range(20, 64):
        # Set up the problem
        # Variables: U (9*rank), V (21*rank), W (21*rank)
        # Equations: for each (i,j,k): sum_r U[i*3+j,r]*V[j*7+k,r]*W[k*3+i,r] = 1
        
        # This is nonlinear. Let's use alternating minimization.
        # Fix V and W, solve for U (linear)
        # Fix U and W, solve for V (linear)
        # Fix U and V, solve for W (linear)
        
        np.random.seed(42)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        for iteration in range(200):
            # Solve for U
            # For each (i,j): sum_r U[i*m+j,r] * (V[j*p:..., r] * W[... , r]) = 1
            # More precisely: for each i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # Group by (i,j): sum_k [sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]] = sum_k 1 = p
            # But we need each (i,j,k) separately.
            
            # For fixed (i,j), the equation for each k is:
            # sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1
            # This is a linear system in U[i*m+j,:]:
            # Let A[i,j,k,r] = V[j*p+k,r] * W[k*n+i,r]
            # Then sum_r U[i*m+j,r] * A[i,j,k,r] = 1 for each k
            
            # So for each (i,j), we have a system: M[i,j] @ U[i*m+j,:] = ones(p)
            # where M[i,j][k,r] = V[j*p+k,r] * W[k*n+i,r]
            
            for i in range(n):
                for j in range(m):
                    # Build M: p x rank
                    M = np.zeros((p, rank))
                    for k in range(p):
                        for r in range(rank):
                            M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # Solve M @ U[i*m+j,:] = ones(p)
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Solve for V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, rank))
                    for i in range(n):
                        for r in range(rank):
                            M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Solve for W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        for r in range(rank):
                            M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            # Compute error
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(rank):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        max_err = max(max_err, abs(val - 1))
            
            if max_err < 1e-8:
                # Found a valid decomposition
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fallback to schoolbook
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
