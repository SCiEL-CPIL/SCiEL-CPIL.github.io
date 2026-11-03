import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    # U: (n*m, rank) = (9, rank)
    # V: (m*p, rank) = (12, rank)
    # W: (p*n, rank) = (12, rank)
    # Tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    deadline = time.time() + 140  # 140s to leave safety margin
    
    # Build the list of (i,j,k) indices
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_eqs = len(indices)  # 36
    
    best_result = None
    
    # Try ranks from 35 down to some minimum
    for rank in range(35, 20, -1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline - 20:
                break
            
            # Random initialization
            np.random.seed(42 + rank * 100 + trial)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # ALS iterations
            for it in range(200):
                if time.time() > deadline - 20:
                    break
                
                # Update U: for each (i,j), U[i*m+j, :] = sum_k V[j*p+k,:] * W[k*n+i,:] / sum_k V[j*p+k,:]^2 * W[k*n+i,:]^2... 
                # Actually, this is a bilinear problem, not linear in each factor alone.
                # The equation is: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                # For fixed V,W, this is linear in U: for each (i,j,k), U[i*m+j,:] dot (V[j*p+k,:] * W[k*n+i,:]) = 1
                # So for fixed (i,j), we have p equations: U[i*m+j,:] dot (V[j*p+k,:] * W[k*n+i,:]) = 1 for k=0..p-1
                # This is an underdetermined system (rank equations, p constraints). We can use least squares.
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # For each k, we need: U[i*m+j,:] dot (V[j*p+k,:] * W[k*n+i,:]) = 1
                        # Let A_k = V[j*p+k,:] * W[k*n+i,:] (element-wise product), shape (rank,)
                        # We need U[i*m+j,:] dot A_k = 1 for all k
                        # This is: (stack of A_k) @ U[i*m+j,:] = ones(p)
                        # Let M be p x rank matrix with M[k,:] = V[j*p+k,:] * W[k*n+i,:]
                        # Then M @ U[i*m+j,:] = ones(p)
                        # Least squares solution: U[i*m+j,:] = pinv(M) @ ones(p)
                        M = np.zeros((p, rank))
                        for k in range(p):
                            M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        rhs = np.ones(p)
                        U[i * m + j, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        # For each i, we need: U[i*m+j,:] dot V[j*p+k,:] * W[k*n+i,:] = 1
                        # Let A_i = U[i*m+j,:] * W[k*n+i,:]
                        # We need V[j*p+k,:] dot A_i = 1 for all i
                        # M[i,:] = U[i*m+j,:] * W[k*n+i,:], shape (n, rank)
                        # M @ V[j*p+k,:] = ones(n)
                        M = np.zeros((n, rank))
                        for i in range(n):
                            M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        rhs = np.ones(n)
                        V[j * p + k, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        # For each j, we need: U[i*m+j,:] dot V[j*p+k,:] * W[k*n+i,:] = 1
                        # Let A_j = U[i*m+j,:] * V[j*p+k,:]
                        # We need W[k*n+i,:] dot A_j = 1 for all j
                        # M[j,:] = U[i*m+j,:] * V[j*p+k,:], shape (m, rank)
                        # M @ W[k*n+i,:] = ones(m)
                        M = np.zeros((m, rank))
                        for j in range(m):
                            M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        rhs = np.ones(m)
                        W[k * n + i, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
            
            # Check residual
            max_err = 0
            for (i, j, k) in indices:
                val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                err = abs(val - 1)
                if err > max_err:
                    max_err = err
            
            if max_err < 1e-8:
                # Try to rationalize
                U_r = np.round(U * 2) / 2  # Try half-integers
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Check if rationalized version is exact
                max_err_r = 0
                for (i, j, k) in indices:
                    val = np.sum(U_r[i * m + j, :] * V_r[j * p + k, :] * W_r[k * n + i, :])
                    err = abs(val - 1)
                    if err > max_err_r:
                        max_err_r = err
                
                if max_err_r < 1e-10:
                    # Found a valid solution!
                    U_list = [[float(x) for x in row] for row in U_r]
                    V_list = [[float(x) for x in row] for row in V_r]
                    W_list = [[float(x) for x in row] for row in W_r]
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        # If we get here, this rank didn't work, try next
    
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
    return {"U": U, "V": V, "W": W}
