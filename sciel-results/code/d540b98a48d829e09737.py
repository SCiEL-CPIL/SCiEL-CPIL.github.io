import numpy as np
import time

def construct():
    n, m, p = 3, 3, 5
    # Target: T[i,k,j] = 1 for all i in range(n), k in range(p), j in range(m)
    # Decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    best_result = None
    deadline = time.time() + 140  # 10s safety margin
    
    # Try ranks from 20 to 27
    for rank in range(20, 28):
        if time.time() > deadline - 5:
            break
        
        # Multiple random restarts
        for restart in range(50):
            if time.time() > deadline - 2:
                break
            
            # Initialize
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # ALS iterations
            converged = False
            for it in range(200):
                if time.time() > deadline - 2:
                    break
                
                # Update U: for each (i,j), solve for U[i*m+j, :]
                for i in range(n):
                    for j in range(m):
                        # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all k
                        # Let A_k = V[j,k,:] * W[k,i,:] (elementwise product, shape rank)
                        # Then sum_c U[i,j,c] * A_k[c] = 1 for all k
                        # This is: A @ U[i,j,:] = 1 where A[k,c] = V[j,k,c]*W[k,i,c]
                        A = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                        # A is p x rank, solve A @ u = ones(p)
                        try:
                            u, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[i*m+j, :] = u
                        except:
                            pass
                
                # Update V: for each (j,k), solve for V[j*p+k, :]
                for j in range(m):
                    for k in range(p):
                        # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i
                        A = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                        try:
                            v, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            V[j*p+k, :] = v
                        except:
                            pass
                
                # Update W: for each (k,i), solve for W[k*n+i, :]
                for k in range(p):
                    for i in range(n):
                        # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all j
                        A = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                        try:
                            w, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W[k*n+i, :] = w
                        except:
                            pass
                
                # Check convergence
                # Compute residual
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            max_err = max(max_err, abs(val - 1))
                
                if max_err < 1e-8:
                    converged = True
                    break
            
            if converged:
                # Found valid decomposition of this rank
                result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_result = result
                break
        
        if best_result is not None:
            break
    
    if best_result is None:
        # Fall back to schoolbook
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
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
