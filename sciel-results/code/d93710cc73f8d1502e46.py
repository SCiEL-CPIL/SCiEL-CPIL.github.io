import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    def verify(U, V, W, T, tol=1e-9):
        D = np.einsum('at,bt,ct->abc', U, V, W)
        return np.allclose(D, T, atol=tol)

    def solve_subproblem(n_sub, m_sub, p_sub, target_rank, time_budget, rng_seed):
        """
        Solves for tensor of shape (n_sub*m_sub, m_sub*p_sub, p_sub*n_sub).
        Returns U, V, W if found, else None.
        """
        T_sub = np.zeros((n_sub*m_sub, m_sub*p_sub, p_sub*n_sub), dtype=float)
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    a = i * m_sub + j
                    b = j * p_sub + k
                    c = k * n_sub + i
                    T_sub[a, b, c] = 1.0
        
        def flatten(U, V, W):
            return np.concatenate((U.flatten(), V.flatten(), W.flatten()))
        
        def unflatten(x, r):
            size_u = n_sub*m_sub*r
            size_v = m_sub*p_sub*r
            size_w = p_sub*n_sub*r
            U = x[:size_u].reshape(n_sub*m_sub, r)
            V = x[size_u:size_u+size_v].reshape(m_sub*p_sub, r)
            W = x[size_u+size_v:].reshape(p_sub*n_sub, r)
            return U, V, W
            
        def objective(x, r):
            U, V, W = unflatten(x, r)
            D = np.einsum('at,bt,ct->abc', U, V, W)
            E = T_sub - D
            # Add regularization to prevent scale drift
            reg = 1e-6 * (np.sum(U**2) + np.sum(V**2) + np.sum(W**2))
            return np.sum(E**2) + reg
            
        def gradient(x, r):
            U, V, W = unflatten(x, r)
            D = np.einsum('at,bt,ct->abc', U, V, W)
            E = T_sub - D
            grad_U = -2.0 * np.einsum('abc,bt,ct->at', E, V, W)
            grad_V = -2.0 * np.einsum('abc,at,ct->bt', E, U, W)
            grad_W = -2.0 * np.einsum('abc,at,bt->ct', E, U, V)
            # Gradient of regularization
            grad_U += 2e-6 * U
            grad_V += 2e-6 * V
            grad_W += 2e-6 * W
            return flatten(grad_U, grad_V, grad_W)
        
        best_solution = None
        best_loss = float('inf')
        
        # Try multiple restarts
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit - 5:
                break
            rng = np.random.default_rng(rng_seed + restart)
            x0 = rng.normal(0, 1.0, size=(n_sub*m_sub*target_rank + m_sub*p_sub*target_rank + p_sub*n_sub*target_rank))
            
            try:
                res = minimize(
                    objective, 
                    x0, 
                    method='L-BFGS-B', 
                    jac=lambda x: gradient(x, target_rank),
                    args=(target_rank,),
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                continue
                
            current_loss = res.fun
            if current_loss < best_loss:
                best_loss = current_loss
                U, V, W = unflatten(res.x, target_rank)
                
                # Round to nearest 0.5
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if verify(U_rounded, V_rounded, W_rounded, T_sub):
                    return U_rounded, V_rounded, W_rounded
                
                # Keep best float solution for potential further refinement or return if close enough
                best_solution = (U_rounded, V_rounded, W_rounded)
                
        return best_solution

    # 1. Solve M_{3,3,1} (n=3, m=3, p=1)
    # Rank guess: 7 or 8
    U_331 = None
    V_331 = None
    W_331 = None
    for r_vec in [7, 8, 9]:
        if time.time() - start_time > time_limit - 20:
            break
        res_vec = solve_subproblem(3, 3, 1, r_vec, 10.0, 1234)
        if res_vec is not None:
            U_331, V_331, W_331 = res_vec
            break

    # 2. Solve M_{3,3,3} (n=3, m=3, p=3)
    # Rank guess: 19, 20, 21
    U_333 = None
    V_333 = None
    W_333 = None
    for r_333 in [19, 20, 21]:
        if time.time() - start_time > time_limit - 10:
            break
        res_333 = solve_subproblem(3, 3, 3, r_333, 10.0, 5678)
        if res_333 is not None:
            U_333, V_333, W_333 = res_333
            break

    # 3. Combine
    if U_331 is not None and U_333 is not None:
        # M_{3,3,3} W has shape (9, R333). Pad with 3 zeros to make (12, R333)
        W_333_padded = np.zeros((12, U_333.shape[1]))
        W_333_padded[:9, :] = W_333
        
        # M_{3,3,1} W has shape (3, R331). Pad with 9 zeros to make (12, R331)
        W_331_padded = np.zeros((12, U_331.shape[1]))
        W_331_padded[9:, :] = W_331
        
        # U and V for 333 have shapes (9, R333) and (9, R333)
        # U and V for 331 have shapes (9, R331) and (3, R331)
        # Wait, V for 331 is (m*p, R) = (3*1, R) = (3, R).
        # V for 333 is (m*p, R) = (3*3, R) = (9, R).
        # We need to align V dimensions for the full problem.
        # Full V dimension is m*p = 3*4 = 12.
        # V_333 corresponds to k=0,1,2. Indices 0..8.
        # V_331 corresponds to k=3? No.
        # In M_{3,3,1}, the "p" dimension is 1. The index k is 0.
        # But in the full problem, the "p" dimension is 4.
        # The V index is j*p + k.
        # For M_{3,3,3}, k=0,1,2. V indices: j*3 + k.
        # For M_{3,3,1}, we are computing the part for k=3.
        # So the V index should be j*4 + 3.
        # The V factor for M_{3,3,1} has shape (3, R). The rows correspond to j=0,1,2.
        # We need to place these into rows 3, 7, 11 of the full V matrix (since j*4+3 for j=0,1,2 is 3, 7, 11).
        
        V_333_padded = np.zeros((12, U_333.shape[1]))
        # V_333 rows are j*3+k.
        # For k=0,1,2, j=0,1,2.
        # We need to map these to the full indices j*4+k.
        # This mapping is not a simple padding if the strides are different.
        # V_333[row] corresponds to (j, k) where row = j*3 + k.
        # Full V[row_full] corresponds to (j, k) where row_full = j*4 + k.
        # So we need to permute/expand V_333.
        for j in range(3):
            for k in range(3):
                idx_333 = j * 3 + k
                idx_full = j * 4 + k
                V_333_padded[idx_full, :] = V_333[idx_333, :]
        
        V_331_padded = np.zeros((12, U_331.shape[1]))
        # V_331 rows correspond to j=0,1,2 (since p=1, index is j*1+0 = j).
        # In full problem, these correspond to k=3.
        # So idx_full = j*4 + 3.
        for j in range(3):
            idx_331 = j
            idx_full = j * 4 + 3
            V_331_padded[idx_full, :] = V_331[idx_331, :]
            
        # U factors
        # U_333 shape (9, R333). Indices i*m+j = i*3+j.
        # Full U indices i*3+j. Same.
        U_333_padded = U_333
        
        # U_331 shape (9, R331). Indices i*3+j.
        # Full U indices i*3+j. Same.
        U_331_padded = U_331
        
        # W factors
        # W_333 shape (9, R333). Indices k*n+i = k*3+i.
        # Full W indices k*3+i for k=0,1,2. Same.
        W_333_padded = np.zeros((12, U_333.shape[1]))
        W_333_padded[:9, :] = W_333
        
        # W_331 shape (3, R331). Indices k*n+i = 0*3+i = i.
        # Full W indices for k=3 are 3*3+i = 9+i.
        W_331_padded = np.zeros((12, U_331.shape[1]))
        W_331_padded[9:, :] = W_331
        
        # Concatenate
        U_total = np.hstack([U_333_padded, U_331_padded])
        V_total = np.hstack([V_333_padded, V_331_padded])
        W_total = np.hstack([W_333_padded, W_331_padded])
        
        # Verify
        T = np.zeros((9, 12, 12), dtype=float)
        for i in range(3):
            for j in range(3):
                for k in range(4):
                    a = i * 3 + j
                    b = j * 4 + k
                    c = k * 3 + i
                    T[a, b, c] = 1.0
                    
        if verify(U_total, V_total, W_total, T):
            return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

    # Fallback: General search or Schoolbook
    # If the block decomposition failed, try a general search for rank 25-35
    # Or just return schoolbook
    n, m, p = 3, 3, 4
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
