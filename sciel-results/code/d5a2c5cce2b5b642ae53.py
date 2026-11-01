import numpy as np
import time

def construct():
    n, m, p = 2, 3, 3
    # Target tensor T of shape (n*m, m*p, p*n)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    U_size = n * m
    V_size = m * p
    W_size = p * n
    
    T = np.zeros((U_size, V_size, W_size))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    # Try ranks from 14 upwards
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_rank = n * m * p
    best_decomp = None
    
    # Function to verify exactness
    def verify(U, V, W):
        # U, V, W are lists of lists or numpy arrays
        # Compute tensor
        rank = U.shape[1]
        T_calc = np.zeros((U_size, V_size, W_size))
        for c in range(rank):
            T_calc += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        
        # Check if T_calc matches T
        # Allow small tolerance for floating point, then check if values are close to integers
        if not np.allclose(T_calc, T, atol=1e-9):
            return False
        return True

    def snap_and_verify(U, V, W):
        # Snap to nearest 0.5
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        if verify(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
        return None

    # Try Rank 14
    for rank in [14, 15, 16, 17]:
        if time.time() > deadline:
            break
            
        # Number of restarts depends on time left
        max_restarts = 50 if rank == 14 else 20
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Scale matters. Entries are around 1.
            U = np.random.randn(U_size, rank) * 0.5
            V = np.random.randn(V_size, rank) * 0.5
            W = np.random.randn(W_size, rank) * 0.5
            
            # Simple gradient descent or least squares alternating
            # Given the small size, we can use scipy.optimize
            # But scipy might be slow. Let's try a simple alternating minimization.
            
            # Flatten U, V, W
            x = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Objective function
            def objective(x):
                U_x = x[:U_size*rank].reshape(U_size, rank)
                V_x = x[U_size*rank:U_size*rank+V_size*rank].reshape(V_size, rank)
                W_x = x[U_size*rank+V_size*rank:].reshape(W_size, rank)
                
                T_calc = np.zeros((U_size, V_size, W_size))
                for c in range(rank):
                    T_calc += np.einsum('i,j,k->ijk', U_x[:, c], V_x[:, c], W_x[:, c])
                
                return np.sum((T_calc - T)**2)
            
            # Use scipy.optimize.minimize
            try:
                from scipy.optimize import minimize
                res = minimize(objective, x, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-8})
                x_opt = res.x
                U_opt = x_opt[:U_size*rank].reshape(U_size, rank)
                V_opt = x_opt[U_size*rank:U_size*rank+V_size*rank].reshape(V_size, rank)
                W_opt = x_opt[U_size*rank+V_size*rank:].reshape(W_size, rank)
                
                # Snap and verify
                result = snap_and_verify(U_opt, V_opt, W_opt)
                if result:
                    U_final, V_final, W_final = result
                    # Convert to list of lists for return
                    U_list = U_final.tolist()
                    V_list = V_final.tolist()
                    W_list = W_final.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
            except Exception:
                pass

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
