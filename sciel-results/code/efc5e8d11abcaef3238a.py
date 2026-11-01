import numpy as np
import time
from scipy.optimize import minimize

def construct():
    deadline = time.time() + 110.0
    n, m, p = 2, 3, 8
    
    # Target tensor for the full problem
    # U index: i*m+j (0..5)
    # V index: j*p+k (0..23)
    # W index: k*n+i (0..15)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # We aim for rank 44 by decomposing into 4 blocks of 2x3x2 rank 11.
    
    # Step 1: Find a rank-11 decomposition for 2x3x2
    n2, m2, p2 = 2, 3, 2
    dim_u2 = n2 * m2  # 6
    dim_v2 = m2 * p2  # 6
    dim_w2 = p2 * n2  # 4
    
    T2 = np.zeros((dim_u2, dim_v2, dim_w2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T2[i*m2+j, j*p2+k, k*n2+i] = 1.0
                
    def objective(params, r):
        # params is a flat array of size (dim_u2 + dim_v2 + dim_w2) * r
        U = params[:dim_u2*r].reshape(dim_u2, r)
        V = params[dim_u2*r : dim_u2*r + dim_v2*r].reshape(dim_v2, r)
        W = params[dim_u2*r + dim_v2*r :].reshape(dim_w2, r)
        
        recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = T2 - recon
        return np.sum(err**2)
        
    def objective_grad(params, r):
        U = params[:dim_u2*r].reshape(dim_u2, r)
        V = params[dim_u2*r : dim_u2*r + dim_v2*r].reshape(dim_v2, r)
        W = params[dim_u2*r + dim_v2*r :].reshape(dim_w2, r)
        
        recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = T2 - recon
        
        dU = np.einsum('ijk,jc,kc->ic', err, V, W)
        dV = np.einsum('ijk,ic,kc->jc', err, U, W)
        dW = np.einsum('ijk,ic,jc->kc', err, U, V)
        
        return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        
    found_sub = None
    r_sub = 11
    
    # Try multiple random starts
    np.random.seed(42)
    for attempt in range(50):
        if time.time() > deadline - 20:
            break
            
        U0 = np.random.randn(dim_u2, r_sub) * 0.5
        V0 = np.random.randn(dim_v2, r_sub) * 0.5
        W0 = np.random.randn(dim_w2, r_sub) * 0.5
        
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        res = minimize(
            objective, 
            params0, 
            method='BFGS', 
            jac=lambda p: objective_grad(p, r_sub),
            options={'maxiter': 1000, 'gtol': 1e-12, 'disp': False}
        )
        
        if res.success or res.fun < 1e-10:
            params_opt = res.x
            U_opt = params_opt[:dim_u2*r_sub].reshape(dim_u2, r_sub)
            V_opt = params_opt[dim_u2*r_sub : dim_u2*r_sub + dim_v2*r_sub].reshape(dim_v2, r_sub)
            W_opt = params_opt[dim_u2*r_sub + dim_v2*r_sub :].reshape(dim_w2, r_sub)
            
            # Round to half-integers
            U_round = np.round(2 * U_opt) / 2.0
            V_round = np.round(2 * V_opt) / 2.0
            W_round = np.round(2 * W_opt) / 2.0
            
            recon_round = np.einsum('ic,jc,kc->ijk', U_round, V_round, W_round)
            err_round = T2 - recon_round
            
            if np.linalg.norm(err_round) < 1e-9:
                found_sub = (U_round, V_round, W_round)
                break
            else:
                # If rounding fails, try integer rounding
                U_round = np.round(U_opt)
                V_round = np.round(V_opt)
                W_round = np.round(W_opt)
                recon_round = np.einsum('ic,jc,kc->ijk', U_round, V_round, W_round)
                err_round = T2 - recon_round
                if np.linalg.norm(err_round) < 1e-9:
                    found_sub = (U_round, V_round, W_round)
                    break

    if found_sub:
        U_sub, V_sub, W_sub = found_sub
        rank_total = 4 * r_sub
        U_full = np.zeros((n*m, rank_total))
        V_full = np.zeros((m*p, rank_total))
        W_full = np.zeros((p*n, rank_total))
        
        for g in range(4):
            col_offset = g * r_sub
            for c in range(r_sub):
                # U is the same for all groups (indices i, j)
                U_full[:, col_offset + c] = U_sub[:, c]
                
                # V indices: j*p + k = j*8 + (g*2 + k_sub)
                for j in range(m2):
                    for k_sub in range(p2):
                        idx_v = j * p + (g * p2 + k_sub)
                        idx_v_sub = j * p2 + k_sub
                        V_full[idx_v, col_offset + c] = V_sub[idx_v_sub, c]
                        
                # W indices: k*n + i = (g*2 + k_sub)*2 + i
                for k_sub in range(p2):
                    for i in range(n2):
                        idx_w = (g * p2 + k_sub) * n + i
                        idx_w_sub = k_sub * n2 + i
                        W_full[idx_w, col_offset + c] = W_sub[idx_w_sub, c]
                        
        # Verify full solution
        T_full = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_full[i*m+j, j*p+k, k*n+i] = 1.0
                    
        recon_full = np.einsum('ic,jc,kc->ijk', U_full, V_full, W_full)
        err_full = T_full - recon_full
        
        if np.linalg.norm(err_full) < 1e-9:
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Fallback: Schoolbook rank 48
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
