import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication.
    Attempts to find a rank-20 decomposition using multi-start L-BFGS optimization with
    a final high-precision polish pass. Falls back to Strassen's algorithm (rank 21) if
    no rank-20 solution is found.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

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

    def try_rank(rank, max_attempts=15):
        best_loss = np.inf
        best_params = None
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
                
            # Use different random seeds and scales for diversity
            seed = 1000 + attempt * 104729
            np.random.seed(seed)
            
            # Try different initialization scales
            scale = 1.0 + 0.1 * (attempt % 5)
            U0 = np.random.randn(nm, rank) * scale
            V0 = np.random.randn(mp, rank) * scale
            W0 = np.random.randn(pn, rank) * scale
            
            # Add some structured initialization for some attempts
            if attempt % 3 == 0:
                # Initialize based on Strassen-like structure
                for g in range(3):
                    for s in range(7):
                        r = 7 * g + s
                        if r < rank:
                            # Small perturbation around Strassen
                            U0[:, r] += 0.1 * np.random.randn(nm)
                            V0[:, r] += 0.1 * np.random.randn(mp)
                            W0[:, r] += 0.1 * np.random.randn(pn)
            
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                # First phase: coarse optimization
                res = minimize(
                    objective_and_grad, 
                    params0, 
                    args=(T, nm, mp, pn, rank),
                    method='L-BFGS-B', 
                    jac=True,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
                )
                
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_params = res.x.copy()
                    
                    # Check if converged to near zero
                    if best_loss < 1e-12:
                        # Final high-precision polish pass
                        res2 = minimize(
                            objective_and_grad, 
                            best_params, 
                            args=(T, nm, mp, pn, rank),
                            method='L-BFGS-B', 
                            jac=True,
                            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 200}
                        )
                        if res2.fun < best_loss:
                            best_loss = res2.fun
                            best_params = res2.x.copy()
                        break
                        
            except Exception:
                continue
                
        if best_params is not None and best_loss < 1e-8:
            U = best_params[:nm*rank].reshape(nm, rank)
            V = best_params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
            W = best_params[nm*rank+mp*rank:].reshape(pn, rank)
            
            # Final verification
            T_check = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(T_check, T, atol=1e-6):
                return U, V, W, rank
                
        return None

    # Attempt to find rank 20 decomposition
    result = try_rank(20, max_attempts=12)
    
    if result is not None:
        U, V, W, rank = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to Strassen rank 21
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

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
