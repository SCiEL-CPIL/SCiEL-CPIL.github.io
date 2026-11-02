import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication.
    Attempts to find a rank-20 decomposition using structured optimization.
    Falls back to Strassen's algorithm (rank 21) if no rank-20 solution is found.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12

    # Build the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # First, build the rank-21 Strassen solution as baseline
    rank21 = _build_strassen_21(nm, mp, pn)
    
    def objective_and_grad(params, T, nm, mp, pn, rank):
        U = params[:nm*rank].reshape(nm, rank)
        V = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
        W = params[nm*rank+mp*rank:].reshape(pn, rank)
        
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        E = T_approx - T
        loss = 0.5 * np.sum(E**2)
        
        grad_U = -np.einsum('abc,br,cr->ar', E, V, W)
        grad_V = -np.einsum('abc,ar,cr->br', E, U, W)
        grad_W = -np.einsum('abc,ar,br->cr', E, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    def try_rank_20():
        """Try to find a rank-20 decomposition using structured initialization."""
        if time.time() > deadline - 15:
            return None
        
        # Strategy: Start from rank-21 Strassen and try to merge two terms
        # The idea is that if two terms are "close" in some sense, we can
        # potentially replace them with one term
        
        U21, V21, W21 = rank21
        U21 = np.array(U21, dtype=float)
        V21 = np.array(V21, dtype=float)
        W21 = np.array(W21, dtype=float)
        
        best_loss = np.inf
        best_params = None
        
        # Try all pairs of terms to merge (21 choose 2 = 210 pairs)
        # For each pair, try to find a single term that approximates both
        for i in range(21):
            for j in range(i + 1, 21):
                if time.time() > deadline - 10:
                    return best_params if best_loss < 1e-8 else None
                
                # Build a rank-20 initialization by merging terms i and j
                # Keep all other 19 terms as-is, and optimize the merged term
                U_init = np.zeros((nm, 20))
                V_init = np.zeros((mp, 20))
                W_init = np.zeros((pn, 20))
                
                # Copy non-merged terms
                merged_idx = 0
                for r in range(21):
                    if r == i or r == j:
                        continue
                    U_init[:, merged_idx] = U21[:, r]
                    V_init[:, merged_idx] = V21[:, r]
                    W_init[:, merged_idx] = W21[:, r]
                    merged_idx += 1
                
                # Initialize the merged term as average of the two
                U_init[:, 19] = 0.5 * (U21[:, i] + U21[:, j])
                V_init[:, 19] = 0.5 * (V21[:, i] + V21[:, j])
                W_init[:, 19] = 0.5 * (W21[:, i] + W21[:, j])
                
                params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                
                # Optimize only the last term (frozen first 19)
                # Use a reduced optimization: only optimize the 20th term
                def reduced_objective_and_grad(params_red, T, nm, mp, pn, U_fixed, V_fixed, W_fixed):
                    U_red = params_red[:nm].reshape(nm)
                    V_red = params_red[nm:nm+mp].reshape(mp)
                    W_red = params_red[nm+mp:nm+mp+pn].reshape(pn)
                    
                    U = np.hstack([U_fixed, U_red.reshape(-1, 1)])
                    V = np.hstack([V_fixed, V_red.reshape(-1, 1)])
                    W = np.hstack([W_fixed, W_red.reshape(-1, 1)])
                    
                    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                    E = T_approx - T
                    loss = 0.5 * np.sum(E**2)
                    
                    # Gradient w.r.t. U_red, V_red, W_red
                    grad_U_red = -np.einsum('abc,br,cr->a', E, V[:, 19], W[:, 19])
                    grad_V_red = -np.einsum('abc,ar,cr->b', E, U[:, 19], W[:, 19])
                    grad_W_red = -np.einsum('abc,ar,br->c', E, U[:, 19], V[:, 19])
                    
                    grad = np.concatenate([grad_U_red, grad_V_red, grad_W_red])
                    return loss, grad
                
                U_fixed = U_init[:, :19]
                V_fixed = V_init[:, :19]
                W_fixed = W_init[:, :19]
                
                params_red0 = np.concatenate([U_init[:, 19], V_init[:, 19], W_init[:, 19]])
                
                try:
                    from scipy.optimize import minimize
                    res = minimize(
                        reduced_objective_and_grad,
                        params_red0,
                        args=(T, nm, mp, pn, U_fixed, V_fixed, W_fixed),
                        method='L-BFGS-B',
                        jac=True,
                        options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    
                    if res.fun < best_loss:
                        best_loss = res.fun
                        best_params = np.concatenate([
                            U_fixed.flatten(), res.x[:nm],
                            V_fixed.flatten(), res.x[nm:nm+mp],
                            W_fixed.flatten(), res.x[nm+mp:nm+mp+pn]
                        ])
                        
                        if best_loss < 1e-12:
                            # Verify
                            U = best_params[:nm*20].reshape(nm, 20)
                            V = best_params[nm*20:nm*20+mp*20].reshape(mp, 20)
                            W = best_params[nm*20+mp*20:].reshape(pn, 20)
                            T_check = np.einsum('ar,br,cr->abc', U, V, W)
                            if np.allclose(T_check, T, atol=1e-8):
                                return best_params
                except Exception:
                    continue
        
        if best_params is not None and best_loss < 1e-8:
            U = best_params[:nm*20].reshape(nm, 20)
            V = best_params[nm*20:nm*20+mp*20].reshape(mp, 20)
            W = best_params[nm*20+mp*20:].reshape(pn, 20)
            T_check = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(T_check, T, atol=1e-6):
                return best_params
        
        return None

    # Attempt rank-20
    result = try_rank_20()
    
    if result is not None:
        U = result[:nm*20].reshape(nm, 20)
        V = result[nm*20:nm*20+mp*20].reshape(mp, 20)
        W = result[nm*20+mp*20:].reshape(pn, 20)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank-21
    U, V, W = rank21
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_strassen_21(nm, mp, pn):
    """Build the rank-21 Strassen-based decomposition."""
    rank = 21
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]
    
    def map_a(local_idx):
        return local_idx
        
    def map_b(local_idx, g):
        k0 = 2 * g
        k1 = 2 * g + 1
        if local_idx == 0:
            return 0 * 6 + k0
        elif local_idx == 1:
            return 0 * 6 + k1
        elif local_idx == 2:
            return 1 * 6 + k0
        else:
            return 1 * 6 + k1

    def map_c(local_idx, g):
        k0 = 2 * g
        k1 = 2 * g + 1
        if local_idx == 0:
            return k0 * 2 + 0
        elif local_idx == 1:
            return k1 * 2 + 0
        elif local_idx == 2:
            return k0 * 2 + 1
        else:
            return k1 * 2 + 1

    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            for local_idx in range(4):
                u_val = step['u'][local_idx]
                if u_val != 0:
                    U[map_a(local_idx), r] = u_val
                    
                v_val = step['v'][local_idx]
                if v_val != 0:
                    V[map_b(local_idx, g), r] = v_val
                    
                w_val = step['w'][local_idx]
                if w_val != 0:
                    W[map_c(local_idx, g), r] = w_val
    
    return U, V, W
