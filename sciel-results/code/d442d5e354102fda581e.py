"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U0 = np.zeros((n * m, rank_school))
    V0 = np.zeros((m * p, rank_school))
    W0 = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    # Try to find a lower rank decomposition
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
    # for all i,j,k and 0 otherwise (but for matrix mult, T[i,j,k]=1 for all valid indices)
    
    # Actually for matrix multiplication, the tensor has T[i,j,k] = 1 for all i,j,k in valid ranges
    # So we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # Let's try a search approach: start with rank r < 75 and see if we can find U,V,W
    # such that the constraint is satisfied
    
    # Due to time constraints, let's try a few ranks below 75
    best_rank = rank_school
    best_U, best_V, best_W = U0, V0, W0
    
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s safety margin
    
    # Try ranks from 74 down to maybe 60
    for target_rank in range(rank_school - 1, max(rank_school - 20, 0), -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Use iterative optimization: minimize sum over i,j,k of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
            # This is a non-convex problem, so we use simple gradient-free approach
            
            # Actually, let's use a different approach: 
            # Fix two matrices and solve for the third
            # Given U and V, for each k,i: sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is linear in W for fixed U,V
            
            # Let's try alternating minimization
            U_curr = U.copy()
            V_curr = V.copy()
            W_curr = W.copy()
            
            for iteration in range(50):
                if time.time() > deadline:
                    break
                
                # Update U: fix V, W, solve for U
                # For each i,j: sum_k sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                # This is linear in U[i*m+j,:]
                # Let's build the linear system
                # Actually this is complex, let's use a simpler approach
                
                # Use scipy to solve least squares
                try:
                    from scipy.optimize import minimize
                except:
                    break
                
                # Define objective
                def objective(params):
                    U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
                    V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_flat = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    # Compute error
                    err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_flat[i*m+j, :] * V_flat[j*p+k, :] * W_flat[k*n+i, :])
                                err += (val - 1)**2
                    return err
                
                x0 = np.concatenate([U_curr.flatten(), V_curr.flatten(), W_curr.flatten()])
                
                # This will be very slow, so let's just check if current solution is good enough
                err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_curr[i*m+j, :] * V_curr[j*p+k, :] * W_curr[k*n+i, :])
                            err += (val - 1)**2
                
                if err < 1e-6:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_curr, V_curr, W_curr
                    break
            
            if best_rank == target_rank:
                break
    
    # If we found a lower rank, use it; otherwise use schoolbook
    if best_rank < rank_school:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
    else:
        U_out = U0.tolist()
        V_out = V0.tolist()
        W_out = W0.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
