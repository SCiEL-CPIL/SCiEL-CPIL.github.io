import numpy as np

def construct():
    n, m, p = 2, 4, 7
    
    # Try different ranks
    for rank in [20, 22, 24, 26, 28, 30, 32, 36, 40, 44, 48, 52, 56]:
        # Initialize randomly
        np.random.seed(42)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Alternating minimization
        for it in range(1000):
            # Update U: for each row of U, solve least squares
            for i in range(n*m):
                # T[i, :, :] = U[i,:] * (V * W elementwise)
                # We need: sum_r U[i,r] V[j,r] W[k,r] = T[i,j,k]
                # For fixed i, this is: sum_r U[i,r] * (V[:,r] * W[:,r] reshaped) = T[i,:,:]
                # Build matrix M where M[j*7+k, r] = V[j*7+k, r] * W[k*2+i, r]
                M = np.zeros((m*p, rank))
                for j in range(m):
                    for k in range(p):
                        M[j*p+k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve M @ U[i,:] = T[i,:,:].reshape(-1)
                target = np.ones(m*p)  # T[i,j,k]=1 for all j,k
                U[i, :], _, _, _ = np.linalg.lstsq(M, target, rcond=None)
            
            # Similarly for V and W
            for j in range(m*p):
                M = np.zeros((n*m, rank))
                for ii in range(n):
                    for kk in range(p):
                        # j = j_mid*p + k, so j_mid = j//p, k = j%p
                    # Actually let me redo this
                pass
        
        # Check convergence
        # Build full tensor
        T = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        if np.allclose(T, 1.0, atol=1e-6):
            break
    
    # Return the best found
    ...
