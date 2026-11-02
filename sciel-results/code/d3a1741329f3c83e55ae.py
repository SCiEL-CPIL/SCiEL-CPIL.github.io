import numpy as np
import scipy.optimize as opt

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Function to compute loss and gradient
    def loss_and_grad(params):
        # Unpack params into U, V, W
        # U: n*m x R, V: m*p x R, W: p*n x R
        size_U = n * m
        size_V = m * p
        size_W = p * n
        
        U_flat = params[0:size_U * R]
        V_flat = params[size_U * R : size_U * R + size_V * R]
        W_flat = params[size_U * R + size_V * R : ]
        
        U = U_flat.reshape(n, m, R)
        V = V_flat.reshape(m, p, R)
        W = W_flat.reshape(p, n, R)
        
        # Compute X_ijk = sum_r U_ijr V_jkr W_kir
        # U: (n, m, R), V: (m, p, R), W: (p, n, R)
        # We want to sum over r.
        # X_ijk = einsum('ijr,jkr,kin->ijk', U, V, W)
        X = np.einsum('ijr,jkr,kin->ijk', U, V, W)
        
        # Target is all 1s
        E = X - 1.0
        loss = np.sum(E * E)
        
        # Compute gradients
        # dL/dU_ijr = 2 * sum_k E_ijk * V_jkr * W_kir
        # dL/dV_jkr = 2 * sum_i E_ijk * U_ijr * W_kir
        # dL/dW_kir = 2 * sum_j E_ijk * U_ijr * V_jkr
        
        grad_U = 2.0 * np.einsum('ijk,jkr,kin->ijr', E, V, W)
        grad_V = 2.0 * np.einsum('ijk,ijr,kin->jkr', E, U, W)
        grad_W = 2.0 * np.einsum('ijk,ijr,jkr->kin', E, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return loss, grad

    # Try ranks from lower bound upwards
    min_rank = max(n * m, m * p, p * n)
    
    # We will try a few ranks. If we find an exact solution, return it.
    # If not, return schoolbook.
    
    # To save time, we'll limit the number of attempts per rank and total time.
    import time
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    for R in range(min_rank, schoolbook_rank + 1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random initializations for this rank
        # Small R might need specific structure, large R is easier.
        # We'll try 5 restarts for small R, fewer for larger R.
        num_restarts = 5 if R < 40 else 2
        
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            np.random.seed(1000 + R * 10 + restart)
            # Initialize with small random values
            # Scale: if we want final values ~1, starting small is good.
            scale = 0.1
            U_init = np.random.randn(n, m, R) * scale
            V_init = np.random.randn(m, p, R) * scale
            W_init = np.random.randn(p, n, R) * scale
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                # Use L-BFGS-B
                res = opt.minimize(
                    loss_and_grad, 
                    x0, 
                    jac=True, 
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                
                if res.success and res.fun < 1e-8:
                    # Extract U, V, W
                    U_opt = res.x[0 : n*m*R].reshape(n, m, R)
                    V_opt = res.x[n*m*R : n*m*R + m*p*R].reshape(m, p, R)
                    W_opt = res.x[n*m*R + m*p*R : ].reshape(p, n, R)
                    
                    # Round to half-integers
                    U_int = np.round(U_opt * 2) / 2
                    V_int = np.round(V_opt * 2) / 2
                    W_int = np.round(W_opt * 2) / 2
                    
                    # Verify exactness
                    X_check = np.einsum('ijr,jkr,kin->ijk', U_int, V_int, W_int)
                    if np.allclose(X_check, 1.0, atol=1e-9):
                        # Convert to required format
                        # U: (n*m, R)
                        # V: (m*p, R)
                        # W: (p*n, R)
                        U_list = U_int.reshape(n*m, R).tolist()
                        V_list = V_int.reshape(m*p, R).tolist()
                        W_list = W_int.reshape(p*n, R).tolist()
                        return {"U": U_list, "V": V_list, "W": W_list}
                        
            except Exception:
                pass

    # Fallback to Schoolbook
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
