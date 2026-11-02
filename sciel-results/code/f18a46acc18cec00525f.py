import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 2, 4, 4
    
    # Start with the known rank-28 Strassen-based decomposition
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

    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    # Rank profile analysis to verify if rank 28 is minimal
    # For a tensor of rank r, the mode-1 unfolding has rank <= r
    # The mode-1 unfolding of T is T.reshape(n*m, -1)
    U_np = np.array(U0, dtype=float)
    V_np = np.array(V0, dtype=float)
    W_np = np.array(W0, dtype=float)
    
    # Verify the rank-28 decomposition is exact
    T_recon = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    assert np.max(np.abs(T - T_recon)) < 1e-10, "Rank-28 decomposition is not exact"
    
    # Check mode unfoldings to establish lower bound on tensor rank
    # Mode-1 unfolding: shape (n*m, m*p*p*n) = (8, 128)
    T_unfold1 = T.reshape(n*m, -1)
    rank1 = np.linalg.matrix_rank(T_unfold1)
    
    # Mode-2 unfolding: shape (m*p, n*m*p*n) = (16, 64)
    T_unfold2 = T.transpose(1, 0, 2).reshape(m*p, -1)
    rank2 = np.linalg.matrix_rank(T_unfold2)
    
    # Mode-3 unfolding: shape (p*n, n*m*m*p) = (8, 64)
    T_unfold3 = T.transpose(2, 0, 1).reshape(p*n, -1)
    rank3 = np.linalg.matrix_rank(T_unfold3)
    
    # The tensor rank is at least max(rank1, rank2, rank3)
    lower_bound = max(rank1, rank2, rank3)
    
    # If the lower bound equals 28, we've certified that rank 28 is minimal
    if lower_bound >= 28:
        # Return the certified minimal rank-28 solution
        U_list = U0
        V_list = V0
        W_list = W0
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # If lower bound < 28, try to find a lower rank decomposition
    start_time = time.time()
    time_limit = 100  # Leave 20s margin
    
    best_rank = 28
    best_U = U_np
    best_V = V_np
    best_W = W_np
    
    # Try to find lower rank decompositions
    for target_rank in [24, 25, 26, 27]:
        if time.time() - start_time > time_limit:
            break
            
        np.random.seed(42)
        U_r = np.random.randn(n*m, target_rank) * 0.1
        V_r = np.random.randn(m*p, target_rank) * 0.1
        W_r = np.random.randn(p*n, target_rank) * 0.1
        
        def flatten(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def unflatten(x):
            size_U = n*m * target_rank
            size_V = m*p * target_rank
            U_flat = x[:size_U]
            V_flat = x[size_U:size_U+size_V]
            W_flat = x[size_U+size_V:]
            return U_flat.reshape(n*m, target_rank), V_flat.reshape(m*p, target_rank), W_flat.reshape(p*n, target_rank)
        
        def objective(x):
            U_r, V_r, W_r = unflatten(x)
            T_recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            return np.sum((T - T_recon)**2)
        
        def gradient(x):
            U_r, V_r, W_r = unflatten(x)
            T_recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            diff = T - T_recon
            
            dU = -2 * np.einsum('abc,br,cr->ar', diff, V_r, W_r)
            dV = -2 * np.einsum('abc,ar,cr->br', diff, U_r, W_r)
            dW = -2 * np.einsum('abc,ar,br->cr', diff, U_r, V_r)
            
            return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        
        x0 = flatten(U_r, V_r, W_r)
        
        try:
            result = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                             options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 500})
            
            if time.time() - start_time > time_limit:
                break
                
            final_obj = result.fun
            if final_obj < 1e-6:
                U_r, V_r, W_r = unflatten(result.x)
                best_rank = target_rank
                best_U = U_r
                best_V = V_r
                best_W = W_r
                break
        except Exception:
            pass
    
    if best_rank < 28:
        U_out = np.round(best_U * 2) / 2
        V_out = np.round(best_V * 2) / 2
        W_out = np.round(best_W * 2) / 2
        
        T_recon = np.einsum('ar,br,cr->abc', U_out, V_out, W_out)
        if np.max(np.abs(T - T_recon)) < 1e-9:
            U_list = U_out.tolist()
            V_list = V_out.tolist()
            W_list = W_out.tolist()
        else:
            U_list = U0
            V_list = V0
            W_list = W0
    else:
        U_list = U0
        V_list = V0
        W_list = W0
    
    return {"U": U_list, "V": V_list, "W": W_list}
