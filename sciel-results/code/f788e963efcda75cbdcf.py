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
        
        # Try multiple restarts with different initializations
        max_restarts = 30
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit - 5:
                break
            rng = np.random.default_rng(rng_seed + restart)
            
            # Different initialization strategies
            if restart % 3 == 0:
                x0 = rng.normal(0, 1.0, size=(n_sub*m_sub*target_rank + m_sub*p_sub*target_rank + p_sub*n_sub*target_rank))
            elif restart % 3 == 1:
                x0 = rng.normal(0, 0.1, size=(n_sub*m_sub*target_rank + m_sub*p_sub*target_rank + p_sub*n_sub*target_rank))
            else:
                x0 = rng.choice([-1, 1], size=(n_sub*m_sub*target_rank + m_sub*p_sub*target_rank + p_sub*n_sub*target_rank))
            
            try:
                res = minimize(
                    objective, 
                    x0, 
                    method='L-BFGS-B', 
                    jac=lambda x: gradient(x, target_rank),
                    args=(target_rank,),
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
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
    # Hardcode this trivial case: Rank 9
    U_331 = np.eye(9)
    V_331 = np.zeros((3, 9))
    W_331 = np.zeros((3, 9))
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            V_331[j, idx] = 1.0
            W_331[i, idx] = 1.0
            
    # Verify hardcoded 331
    T_331 = np.zeros((9, 3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            a = i * 3 + j
            b = j
            c = i
            T_331[a, b, c] = 1.0
    if not verify(U_331, V_331, W_331, T_331):
        # Fallback to schoolbook if hardcoded fails (should not happen)
        U_331 = np.zeros((9, 9))
        V_331 = np.zeros((3, 9))
        W_331 = np.zeros((3, 9))
        for i in range(3):
            for j in range(3):
                idx = i * 3 + j
                U_331[idx, idx] = 1.0
                V_331[j, idx] = 1.0
                W_331[i, idx] = 1.0

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
    if U_333 is not None:
        # M_{3,3,3} W has shape (9, R333). Pad with 3 zeros to make (12, R333)
        W_333_padded = np.zeros((12, U_333.shape[1]))
        W_333_padded[:9, :] = W_333
        
        # M_{3,3,1} W has shape (3, R331). Pad with 9 zeros to make (12, R331)
        W_331_padded = np.zeros((12, U_331.shape[1]))
        W_331_padded[9:, :] = W_331
        
        V_333_padded = np.zeros((12, U_333.shape[1]))
        # V_333 rows are j*3+k.
        # For k=0,1,2, j=0,1,2.
        # We need to map these to the full indices j*4+k.
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
        U_333_padded = U_333
        U_331_padded = U_331
        
        # W factors
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
