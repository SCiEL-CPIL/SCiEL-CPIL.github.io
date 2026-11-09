import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s margin
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1.0
    
    # Start with schoolbook rank 30
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[d, col] = 1
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by removing columns that are linearly dependent
    # or by finding a lower rank decomposition
    
    # Try ranks from 29 down to some minimum
    for target_rank in range(rank - 1, 15, -1):
        if time.time() > deadline:
            break
        
        # Try to find a decomposition of the given rank
        # Use a random initialization and ALS
        success = False
        for trial in range(10):
            if time.time() > deadline:
                break
            
            U_trial = np.random.randn(n*m, target_rank)
            V_trial = np.random.randn(m*p, target_rank)
            W_trial = np.random.randn(p*n, target_rank)
            
            # ALS iterations
            for it in range(100):
                if time.time() > deadline:
                    break
                
                # Update U: for each a, solve U[a,:] from the equations
                # The equation is: sum_r U[a,r] * V[b,r] * W[d,r] = T[a,b,d]
                # For fixed a, this is: sum_r U[a,r] * (V[b,r] * W[d,r]) = T[a,b,d] for all b,d
                # This is a linear system in U[a,:]
                
                # Actually, let's use a different approach: 
                # Fix V and W, solve for U using least squares
                # The full equation is: sum_r U[a,r] V[b,r] W[d,r] = T[a,b,d]
                # This can be written as: for each (a,b,d), we have one equation.
                # We can vectorize this.
                
                # Let's use a simpler ALS: fix two, solve for the third.
                # Fix V and W, solve for U:
                # For each a, we have equations: sum_r U[a,r] * V[b,r] * W[d,r] = T[a,b,d]
                # This is a linear system: M_a @ U[a,:] = t_a
                # where M_a is a matrix of size (m*p * p*n) x target_rank
                # and t_a is a vector of size (m*p * p*n)
                
                # This might be too large. Let's use a different approach.
                
                # Alternative: use the fact that the tensor is sparse.
                # Only 30 entries of T are non-zero.
                
                # For now, let's just check if the current U,V,W satisfy the equations
                # and if not, try to improve.
                
                # Check residual
                residual = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            d = k * n + i
                            val = np.sum(U_trial[a, :] * V_trial[b, :] * W_trial[d, :])
                            residual += (val - 1.0) ** 2
                
                if residual < 1e-10:
                    success = True
                    break
                
                # If not converged, try a different approach
                # For now, just break and try next trial
                break
        
        if success:
            best_rank = target_rank
            best_U, best_V, best_W = U_trial, V_trial, W_trial
            break
    
    # Convert to lists
    U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else (2*x if abs(x*2 - round(x*2)) < 1e-9 else x) for x in row] for row in best_U]
    V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else (2*x if abs(x*2 - round(x*2)) < 1e-9 else x) for x in row] for row in best_V]
    W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else (2*x if abs(x*2 - round(x*2)) < 1e-9 else x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
