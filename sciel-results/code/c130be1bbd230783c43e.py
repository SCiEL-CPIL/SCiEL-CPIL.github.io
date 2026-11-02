import numpy as np
import time
import random
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Fallback Schoolbook
    U_school = np.zeros((n*m, n*m*p))
    V_school = np.zeros((m*p, n*m*p))
    W_school = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1

    def compute_residuals(params, rank):
        u_len = n * m * rank
        v_len = m * p * rank
        
        U = params[:u_len].reshape(n, m, rank)
        V = params[u_len:u_len+v_len].reshape(m, p, rank)
        W = params[u_len+v_len:].reshape(p, n, rank)
        
        # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # Reshape W to align indices for multiplication
        # We want to sum over r.
        # U[i, j, r] * V[j, k, r] -> Shape (n, m, p, rank) if we broadcast V
        # V is (m, p, rank). We need it to be (1, m, p, rank) to broadcast with U (n, m, 1, rank)?
        # No, U is (n, m, rank).
        # Let's use einsum for clarity and speed.
        # 'ijr,jkr,kin->ijk'
        T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        
        return (T - 1).flatten()

    def verify_solution(U, V, W, rank):
        # Check if U, V, W are close to half-integers
        U_int = np.round(U * 2) / 2
        V_int = np.round(V * 2) / 2
        W_int = np.round(W * 2) / 2
        
        # Verify exactness with integers
        U_3d = U_int.reshape(n, m, rank)
        V_3d = V_int.reshape(m, p, rank)
        W_3d = W_int.reshape(p, n, rank)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        if np.allclose(T, 1, atol=1e-9):
            return {
                "U": U_int.reshape(n*m, rank).tolist(),
                "V": V_int.reshape(m*p, rank).tolist(),
                "W": W_int.reshape(p*n, rank).tolist()
            }
        return None

    # Search for minimal rank
    # Start from a reasonable lower bound. 
    # Rank 1 is impossible. Rank 14 was tried and failed.
    # Let's try ranks from 20 to 40.
    # We want the MINIMAL rank, so we iterate upwards.
    
    best_solution = None
    
    for rank in range(20, 41):
        if time.time() > deadline:
            break
            
        U_shape = (n*m, rank)
        V_shape = (m*p, rank)
        W_shape = (p*n, rank)
        
        # Try a few random restarts for this rank
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Random initialization with half-integers
            params = np.random.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], 
                                      size=(n*m*rank + m*p*rank + p*n*rank))
            
            try:
                result = least_squares(
                    lambda x: compute_residuals(x, rank), 
                    params, 
                    method='lm', 
                    max_nfev=2000
                )
                
                if result.cost < 1e-8:
                    # Extract solution
                    u_len = n * m * rank
                    v_len = m * p * rank
                    
                    U_sol = result.x[:u_len].reshape(n, m, rank)
                    V_sol = result.x[u_len:u_len+v_len].reshape(m, p, rank)
                    W_sol = result.x[u_len+v_len:].reshape(p, n, rank)
                    
                    sol = verify_solution(U_sol, V_sol, W_sol, rank)
                    if sol:
                        return sol
                        
            except Exception:
                continue

    # If no rank < 45 found, return schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
