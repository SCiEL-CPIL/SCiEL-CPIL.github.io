import numpy as np
import time
import sys
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    n, m, p = 2, 2, 15
    
    def get_rank_53():
        rank = 53
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen for 2x2x2
        U_s = [
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ]
        V_s = [
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1]
        ]
        W_s = [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ]
        
        # Schoolbook for 2x2x1
        U_sc = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        V_sc = [
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ]
        W_sc = [
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        
        col = 0
        for b in range(7):
            k0 = 2 * b
            k1 = 2 * b + 1
            for r in range(7):
                c = col + r
                for i in range(2):
                    for j in range(2):
                        U[i*m + j][c] = U_s[i*2 + j][r]
                for j in range(2):
                    V[j*p + k0][c] = V_s[j*2 + 0][r]
                    V[j*p + k1][c] = V_s[j*2 + 1][r]
                for k in range(2):
                    kk = k0 + k
                    for i in range(2):
                        W[kk*n + i][c] = W_s[k*2 + i][r]
            col += 7
            
        k_last = 14
        for r in range(4):
            c = col + r
            for i in range(2):
                for j in range(2):
                    U[i*m + j][c] = U_sc[i*2 + j][r]
            for j in range(2):
                V[j*p + k_last][c] = V_sc[j][r]
            for i in range(2):
                W[k_last*n + i][c] = W_sc[i][r]
                
        return {"U": U, "V": V, "W": W}

    def try_optimized_rank(target_rank):
        """
        Try to find a decomposition of the given rank by fixing W to canonical
        basis vectors and solving for U and V using structured optimization.
        
        The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        For matrix multiplication: T[i,j,k] = 1 if the indices match the
        multiplication pattern, else 0.
        
        Strategy: Fix W to have one non-zero entry per rank term (canonical),
        then solve for U and V.
        """
        if time.time() > deadline - 30:
            return None
        
        # Build target tensor for M_{2,2,15}
        # A: 2x2, B: 2x15, C: 2x15
        # T[a_idx, b_idx, c_idx] where a_idx = i*m+j, b_idx = j*p+k, c_idx = k*n+i
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    T[a_idx, b_idx, c_idx] = 1.0
        
        # Try different W configurations
        np.random.seed(123)
        
        for attempt in range(5):
            if time.time() > deadline - 25:
                break
            
            # Initialize W with sparse random structure
            W = np.zeros((p*n, target_rank))
            for r in range(target_rank):
                # Each column of W has exactly one non-zero entry
                idx = np.random.randint(0, p*n)
                W[idx, r] = 1.0
            
            # Now solve for U and V
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For each c, let S_c = {r : W[c,r] != 0}
            # Then T[:, :, c] = sum_{r in S_c} U[:,r] * V[:,r] * W[c,r]
            
            # This is still hard. Let's try a different approach:
            # Use scipy to minimize ||T - sum_r U_r V_r^T W_r||_F^2
            
            def flatten_params(params):
                U_flat = params[:n*m*target_rank]
                V_flat = params[n*m*target_rank:n*m*target_rank + m*p*target_rank]
                return U_flat.reshape(n*m, target_rank), V_flat.reshape(m*p, target_rank)
            
            def objective(params):
                U, V = flatten_params(params)
                # Compute tensor: T_comp[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # Efficient computation using einsum
                # First compute UV: (a,b,rank)
                UV = np.einsum('ar,br->ab r', U, V)
                # Then contract with W: (a,b,c)
                T_comp = np.einsum('abr,cr->abc', UV, W)
                err = T_comp - T
                return np.sum(err**2)
            
            # Initial guess: random
            U_init = np.random.randn(n*m, target_rank) * 0.1
            V_init = np.random.randn(m*p, target_rank) * 0.1
            params_init = np.concatenate([U_init.flatten(), V_init.flatten()])
            
            # Use L-BFGS-B for efficiency
            result = minimize(objective, params_init, method='L-BFGS-B', 
                             options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if result.fun < 1e-8:
                U_opt, V_opt = flatten_params(result.x)
                
                # Try to round to half-integers
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                
                # Verify
                UV = np.einsum('ar,br->abr', U_rounded, V_rounded)
                T_comp = np.einsum('abr,cr->abc', UV, W)
                if np.linalg.norm(T_comp - T) < 1e-6:
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W.tolist()}
            
            # If exact rounding fails, try allowing more flexibility
            # Return the continuous solution if it's close enough
            if result.fun < 1e-6:
                # Convert to list format
                return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W.tolist()}
        
        return None

    # Try to find better ranks starting from 52 down to 45
    best_result = None
    for rank in range(52, 44, -1):
        if time.time() > deadline - 20:
            break
        
        result = try_optimized_rank(rank)
        if result is not None:
            # Verify the solution
            U = np.array(result['U'])
            V = np.array(result['V'])
            W = np.array(result['W'])
            
            # Build target tensor
            T = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a_idx = i * m + j
                        b_idx = j * p + k
                        c_idx = k * n + i
                        T[a_idx, b_idx, c_idx] = 1.0
            
            # Compute decomposition
            UV = np.einsum('ar,br->abr', U, V)
            T_comp = np.einsum('abr,cr->abc', UV, W)
            
            if np.linalg.norm(T_comp - T) < 1e-6:
                best_result = result
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback to rank 53
    return get_rank_53()
