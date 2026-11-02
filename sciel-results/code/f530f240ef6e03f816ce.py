import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 2, 4, 4
    
    # Build the rank-28 Strassen decomposition
    r0 = 28
    U0 = [[0] * r0 for _ in range(n * m)]
    V0 = [[0] * r0 for _ in range(m * p)]
    W0 = [[0] * r0 for _ in range(p * n)]

    def _add_strassen(off, ar0, ac0, br0, bc0, cr0, cc0):
        a00 = ar0 * m + (ac0 + 0)
        a01 = ar0 * m + (ac0 + 1)
        a10 = (ar0 + 1) * m + (ac0 + 0)
        a11 = (ar0 + 1) * m + (ac0 + 1)

        b00 = br0 * p + (bc0 + 0)
        b01 = br0 * p + (bc0 + 1)
        b10 = (br0 + 1) * p + (bc0 + 0)
        b11 = (br0 + 1) * p + (bc0 + 1)

        c00 = cc0 * n + cr0
        c01 = (cc0 + 1) * n + cr0
        c10 = cc0 * n + (cr0 + 1)
        c11 = (cc0 + 1) * n + (cr0 + 1)

        U0[a00][off] += 1
        U0[a11][off] += 1
        V0[b00][off] += 1
        V0[b11][off] += 1
        W0[c00][off] += 1
        W0[c11][off] += 1

        U0[a10][off + 1] += 1
        U0[a11][off + 1] += 1
        V0[b00][off + 1] += 1
        W0[c10][off + 1] += 1
        W0[c11][off + 1] -= 1

        U0[a00][off + 2] += 1
        V0[b01][off + 2] += 1
        V0[b11][off + 2] -= 1
        W0[c01][off + 2] += 1
        W0[c11][off + 2] += 1

        U0[a11][off + 3] += 1
        V0[b10][off + 3] += 1
        V0[b00][off + 3] -= 1
        W0[c00][off + 3] += 1
        W0[c10][off + 3] += 1

        U0[a00][off + 4] += 1
        U0[a01][off + 4] += 1
        V0[b11][off + 4] += 1
        W0[c00][off + 4] -= 1
        W0[c01][off + 4] += 1

        U0[a10][off + 5] += 1
        U0[a00][off + 5] -= 1
        V0[b00][off + 5] += 1
        V0[b01][off + 5] += 1
        W0[c11][off + 5] += 1

        U0[a01][off + 6] += 1
        U0[a11][off + 6] -= 1
        V0[b10][off + 6] += 1
        V0[b11][off + 6] += 1
        W0[c00][off + 6] += 1

    _add_strassen(0, 0, 0, 0, 0, 0, 0)
    _add_strassen(7, 0, 2, 2, 0, 0, 0)
    _add_strassen(14, 0, 0, 0, 2, 0, 2)
    _add_strassen(21, 0, 2, 2, 2, 0, 2)

    U_np = np.array(U0, dtype=float)
    V_np = np.array(V0, dtype=float)
    W_np = np.array(W0, dtype=float)

    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    start_time = time.time()
    time_limit = 130  # Leave 20s margin before 150s kill

    best_rank = 28
    best_U = U_np.copy()
    best_V = V_np.copy()
    best_W = W_np.copy()

    # Strategy: Try to find rank-27 by starting from rank-28 and optimizing
    # Key insight: start from a valid decomposition and try to reduce rank
    
    # First, try rank 27 with multiple restarts
    for target_rank in [27, 26, 25, 24]:
        if time.time() - start_time > time_limit:
            break
        
        max_restarts = 5 if target_rank == 27 else 3
        found = False
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            
            np.random.seed(42 + restart)
            
            if target_rank < 28:
                # Start from the rank-28 solution, try to merge terms
                # Take first target_rank columns and optimize
                U_r = U_np[:, :target_rank].copy()
                V_r = V_np[:, :target_rank].copy()
                W_r = W_np[:, :target_rank].copy()
                # Add some random perturbation
                U_r += np.random.randn(n*m, target_rank) * 0.01
                V_r += np.random.randn(m*p, target_rank) * 0.01
                W_r += np.random.randn(p*n, target_rank) * 0.01
            else:
                U_r = np.random.randn(n*m, target_rank) * 0.1
                V_r = np.random.randn(m*p, target_rank) * 0.1
                W_r = np.random.randn(p*n, target_rank) * 0.1
            
            def flatten(U, V, W, r):
                return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def unflatten(x, r):
                size_U = n*m * r
                size_V = m*p * r
                U_flat = x[:size_U]
                V_flat = x[size_U:size_U+size_V]
                W_flat = x[size_U+size_V:]
                return U_flat.reshape(n*m, r), V_flat.reshape(m*p, r), W_flat.reshape(p*n, r)
            
            def objective(x):
                U_r, V_r, W_r = unflatten(x, target_rank)
                T_recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                return np.sum((T - T_recon)**2)
            
            def gradient(x):
                U_r, V_r, W_r = unflatten(x, target_rank)
                T_recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                diff = T - T_recon
                
                dU = -2 * np.einsum('abc,br,cr->ar', diff, V_r, W_r)
                dV = -2 * np.einsum('abc,ar,cr->br', diff, U_r, W_r)
                dW = -2 * np.einsum('abc,ar,br->cr', diff, U_r, V_r)
                
                return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
            
            x0 = flatten(U_r, V_r, W_r, target_rank)
            
            try:
                result = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                                 options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 300})
                
                if time.time() - start_time > time_limit:
                    break
                
                final_obj = result.fun
                if final_obj < 1e-8:
                    U_r, V_r, W_r = unflatten(result.x, target_rank)
                    # Verify with higher precision
                    T_recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    if np.max(np.abs(T - T_recon)) < 1e-10:
                        best_rank = target_rank
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break
    
    # Try to round to half-integers
    if best_rank < 28:
        U_out = np.round(best_U * 2) / 2
        V_out = np.round(best_V * 2) / 2
        W_out = np.round(best_W * 2) / 2
        
        # Verify after rounding
        T_recon = np.einsum('ar,br,cr->abc', U_out, V_out, W_out)
        if np.max(np.abs(T - T_recon)) < 1e-9:
            U_list = U_out.tolist()
            V_list = V_out.tolist()
            W_list = W_out.tolist()
        else:
            # Try without rounding
            U_list = best_U.tolist()
            V_list = best_V.tolist()
            W_list = best_W.tolist()
    else:
        U_list = U0
        V_list = V0
        W_list = W0
    
    return {"U": U_list, "V": V_list, "W": W_list}
