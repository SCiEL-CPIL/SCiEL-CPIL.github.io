import numpy as np
import time

DIMENSIONS = [2, 2, 4]

def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0
    
    n, m, p = DIMENSIONS
    rank_trivial = n * m * p
    
    # Trivial decomposition
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_trivial, "V": V_trivial, "W": W_trivial}
    
    # Try to find a rank-12 decomposition using structured variables
    # We'll use numpy to solve the system
    
    # For rank r, we have r terms. Let's try r=12.
    # The equations are: for all (i,j,k) in [0,2)x[0,2)x[0,4):
    # sum_{c=0}^{r-1} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is a system of 16 equations in 4*r + 8*r + 8*r = 20*r unknowns.
    # For r=12, that's 240 unknowns, 16 equations - very underdetermined.
    # So solutions should exist. Let's find one.
    
    for target_rank in [12, 11, 10, 9]:
        if time.time() - start_time > deadline - safety_margin:
            break
        
        found = False
        for seed in range(20):
            if time.time() - start_time > deadline - safety_margin:
                break
            
            np.random.seed(seed)
            # Initialize with small random values
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Use iterative refinement: fix two matrices, solve for the third
            # This is a bilinear problem, so we can use alternating least squares
            
            for iteration in range(100):
                # Fix V and W, solve for U
                # For each (i,j): sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
                # This is a system of p equations in r unknowns for each (i,j)
                
                U_new = np.zeros((n*m, target_rank))
                for i in range(n):
                    for j in range(m):
                        # Build the p x r matrix A where A[k,c] = V[j*p+k,c] * W[k*n+i,c]
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve A @ U[i*m+j,:] = ones(p)
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U_new[i*m+j, :] = sol
                        except:
                            pass
                
                # Fix U and W, solve for V
                V_new = np.zeros((m*p, target_rank))
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, target_rank))
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            V_new[j*p+k, :] = sol
                        except:
                            pass
                
                # Fix U and V, solve for W
                W_new = np.zeros((p*n, target_rank))
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, target_rank))
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W_new[k*n+i, :] = sol
                        except:
                            pass
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] for c in range(target_rank))
                            max_err = max(max_err, abs(val - 1))
                
                if max_err < 1e-10:
                    found = True
                    break
            
            if found:
                break
        
        if found:
            # Round to reasonable precision
            U_list = [[round(float(U[i][c]), 10) for c in range(target_rank)] for i in range(n*m)]
            V_list = [[round(float(V[i][c]), 10) for c in range(target_rank)] for i in range(m*p)]
            W_list = [[round(float(W[i][c]), 10) for c in range(target_rank)] for i in range(p*n)]
            
            # Final verification
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U_list[i*m+j][c] * V_list[j*p+k][c] * W_list[k*n+i][c] for c in range(target_rank))
                        if abs(val - 1) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
    
    return best_result
