import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Generate Schoolbook as fallback
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
    
    def compute_tensor(U, V, W, rank):
        """Compute the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]"""
        T = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            
            for j in range(m):
                outer = np.outer(u_c[:, j], v_c[j, :])
                T[:, j, :] += outer * w_c.T
        return T
    
    def verify_solution(U, V, W, rank, atol=1e-9):
        """Verify that the solution exactly decomposes the tensor"""
        T = compute_tensor(U, V, W, rank)
        return np.allclose(T, 1, atol=atol)
    
    def try_round_and_verify(U_sol, V_sol, W_sol, rank):
        """Try rounding to nearest half-integer and verify"""
        # Try rounding to nearest integer
        U_int = np.round(U_sol)
        V_int = np.round(V_sol)
        W_int = np.round(W_sol)
        
        if verify_solution(U_int, V_int, W_int, rank):
            return U_int, V_int, W_int
        
        # Try rounding to nearest half-integer
        U_half = np.round(U_sol * 2) / 2
        V_half = np.round(V_sol * 2) / 2
        W_half = np.round(W_sol * 2) / 2
        
        if verify_solution(U_half, V_half, W_half, rank):
            return U_half, V_half, W_half
        
        return None
    
    # Try to find low-rank decomposition
    from scipy.optimize import least_squares
    
    # Try ranks from low to high
    for target_rank in range(10, 25):
        if time.time() > deadline:
            break
        
        U_shape = (n*m, target_rank)
        V_shape = (m*p, target_rank)
        W_shape = (p*n, target_rank)
        
        def residuals(params, U_shape, V_shape, W_shape, target_rank):
            u_len = U_shape[0] * U_shape[1]
            v_len = V_shape[0] * V_shape[1]
            
            U = params[:u_len].reshape(U_shape)
            V = params[u_len:u_len+v_len].reshape(V_shape)
            W = params[u_len+v_len:].reshape(W_shape)
            
            T = compute_tensor(U, V, W, target_rank)
            return (T - 1).flatten()
        
        # Try multiple restarts for this rank
        for restart in range(5):
            if time.time() > deadline:
                break
            
            # Use structured initialization: start with schoolbook-like pattern
            U_init = np.zeros(U_shape)
            V_init = np.zeros(V_shape)
            W_init = np.zeros(W_shape)
            
            # Initialize with random half-integers
            for c in range(target_rank):
                U_init[:, c] = np.random.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], size=n*m)
                V_init[:, c] = np.random.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], size=m*p)
                W_init[:, c] = np.random.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], size=p*n)
            
            params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = least_squares(
                    lambda x: residuals(x, U_shape, V_shape, W_shape, target_rank),
                    params, 
                    method='trf', 
                    max_nfev=5000, 
                    xtol=1e-10, 
                    ftol=1e-10
                )
                
                if result.cost < 1e-8:
                    u_len = U_shape[0] * U_shape[1]
                    v_len = V_shape[0] * V_shape[1]
                    
                    U_sol = result.x[:u_len].reshape(U_shape)
                    V_sol = result.x[u_len:u_len+v_len].reshape(V_shape)
                    W_sol = result.x[u_len+v_len:].reshape(W_shape)
                    
                    rounded = try_round_and_verify(U_sol, V_sol, W_sol, target_rank)
                    if rounded is not None:
                        U_int, V_int, W_int = rounded
                        return {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
                    
                    # Polish step
                    result2 = least_squares(
                        lambda x: residuals(x, U_shape, V_shape, W_shape, target_rank),
                        result.x, 
                        method='trf', 
                        max_nfev=10000, 
                        xtol=1e-12, 
                        ftol=1e-12
                    )
                    
                    if result2.cost < 1e-10:
                        U_sol2 = result2.x[:u_len].reshape(U_shape)
                        V_sol2 = result2.x[u_len:u_len+v_len].reshape(V_shape)
                        W_sol2 = result2.x[u_len+v_len:].reshape(W_shape)
                        
                        rounded2 = try_round_and_verify(U_sol2, V_sol2, W_sol2, target_rank)
                        if rounded2 is not None:
                            U_int, V_int, W_int = rounded2
                            return {
                                "U": U_int.tolist(),
                                "V": V_int.tolist(),
                                "W": W_int.tolist()
                            }
            except Exception as e:
                pass
    
    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
