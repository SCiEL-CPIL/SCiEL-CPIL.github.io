import numpy as np
import time
import scipy.optimize as so

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = 120.0
    
    # Try to find a low rank decomposition for 2x3x3
    n, m, p = 2, 3, 3
    
    # Tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    T_flat = T.flatten()
    
    best_rank = n*m*p
    best_factors = None
    
    for r in range(14, n*m*p + 1):
        if time.time() > deadline - 5:
            break
            
        # Try to find decomposition of rank r
        # Parameters: U (6*r), V (9*r), W (6*r)
        # Total params: 21*r
        
        def func(params):
            U = params[:n*m*r].reshape(n*m, r)
            V = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W = params[n*m*r + m*p*r :].reshape(p*n, r)
            
            # Compute tensor
            # T_comp[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
            # This is a batched matrix multiplication / einsum
            # U: (6, r), V: (9, r), W: (6, r)
            # We want to form T_comp of shape (6, 9, 6)
            # T_comp[i, j, k] = sum_c U[i, c] * V[j, c] * W[k, c]
            # Note: indices are flattened.
            # i in 0..5, j in 0..8, k in 0..5
            
            # Use einsum
            # 'ic,jc,kc->ijk'
            T_comp = np.einsum('ic,jc,kc->ijk', U, V, W)
            return (T_comp.flatten() - T_flat).flatten()
            
        # Initial guess
        # Random
        np.random.seed(42)
        x0 = np.random.randn((n*m + m*p + p*n) * r) * 0.1
        
        try:
            result = so.least_squares(func, x0, method='lm', max_nfev=1000)
            if result.success and np.linalg.norm(result.fun) < 1e-6:
                # Found a solution
                # Snap to half-integers
                U = result.x[:n*m*r].reshape(n*m, r)
                V = result.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W = result.x[n*m*r + m*p*r :].reshape(p*n, r)
                
                U_snapped = np.round(U * 2) / 2.0
                V_snapped = np.round(V * 2) / 2.0
                W_snapped = np.round(W * 2) / 2.0
                
                # Verify
                T_comp = np.einsum('ic,jc,kc->ijk', U_snapped, V_snapped, W_snapped)
                if np.allclose(T_comp, T, atol=1e-9):
                    best_rank = r
                    best_factors = (U_snapped, V_snapped, W_snapped)
                    break
        except Exception:
            pass
            
    if best_factors is not None:
        # Extend to 2x3x9
        # n=2, m=3, p=9
        # The 2x3x3 solution has U (6, r), V (9, r), W (6, r)
        # We need to create U (6, 3r), V (27, 3r), W (18, 3r)
        
        U_full = np.zeros((6, 3 * best_rank))
        V_full = np.zeros((27, 3 * best_rank))
        W_full = np.zeros((18, 3 * best_rank))
        
        U_local, V_local, W_local = best_factors
        
        for b in range(3):
            # Block b corresponds to k in [3b, 3b+2]
            # U is the same for all blocks
            U_full[:, b*best_rank : (b+1)*best_rank] = U_local
            
            # V: local index j*3 + k' -> global index j*9 + (3b + k')
            # V_local has shape (9, r). Index j*3 + k'.
            # We need to map this to V_full index j*9 + (3b + k').
            for j in range(3):
                for k_prime in range(3):
                    local_idx = j * 3 + k_prime
                    global_idx = j * 9 + (3 * b + k_prime)
                    V_full[global_idx, b*best_rank : (b+1)*best_rank] = V_local[local_idx, :]
                    
            # W: local index k'*2 + i -> global index (3b + k')*2 + i
            # W_local has shape (6, r). Index k'*2 + i.
            for k_prime in range(3):
                for i in range(2):
                    local_idx = k_prime * 2 + i
                    global_idx = (3 * b + k_prime) * 2 + i
                    W_full[global_idx, b*best_rank : (b+1)*best_rank] = W_local[local_idx, :]
                    
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }
    else:
        # Fallback to schoolbook
        n, m, p = 2, 3, 9
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
