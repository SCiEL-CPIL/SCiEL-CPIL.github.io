import time
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Attempts to find a rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
    Uses numerical optimization with a penalty for half-integer entries.
    """
    n, m, p = 2, 3, 4
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor T[a,b,c]
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Start with naive rank-24 decomposition
    rank_naive = nm * p  # 24
    U_naive = np.zeros((nm, rank_naive))
    V_naive = np.zeros((mp, rank_naive))
    W_naive = np.zeros((pn, rank_naive))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_naive[a, r] = 1.0
                V_naive[b, r] = 1.0
                W_naive[c, r] = 1.0
    
    best_result = {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
    best_rank = rank_naive
    
    deadline = time.time() + 140  # 10s buffer
    
    # We suspect the rank is around 20-23.
    # Try ranks from 23 down to 10.
    for target_rank in range(23, 9, -1):
        if time.time() > deadline:
            break
        
        n_restarts = 5 if target_rank >= 20 else 3
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(1000 + target_rank * 10 + restart)
            
            # Initialize with random values
            U_init = np.random.randn(nm, target_rank) * 0.5
            V_init = np.random.randn(mp, target_rank) * 0.5
            W_init = np.random.randn(pn, target_rank) * 0.5
            
            def flatten(U, V, W):
                return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def unflatten(x):
                U = x[:nm*target_rank].reshape(nm, target_rank)
                V = x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                W = x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                return U, V, W
            
            def half_integer_penalty(U, V, W):
                # Penalty for being far from half-integers
                # x is close to k/2 if 2x is close to integer k
                # (2x - round(2x))^2
                return np.sum((2*U - np.round(2*U))**2) + \
                       np.sum((2*V - np.round(2*V))**2) + \
                       np.sum((2*W - np.round(2*W))**2)
            
            def objective(x):
                U, V, W = unflatten(x)
                T_approx = np.zeros_like(T)
                for r in range(target_rank):
                    T_approx += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                error = np.sum((T - T_approx)**2)
                pen = half_integer_penalty(U, V, W)
                # We want error to be 0. Penalty is secondary but helps guide to valid domain.
                # If error is small, penalty should also be small if a solution exists.
                return error + 0.1 * pen
            
            def gradient(x):
                U, V, W = unflatten(x)
                T_approx = np.zeros_like(T)
                for r in range(target_rank):
                    T_approx += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                diff = T_approx - T
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for r in range(target_rank):
                    grad_U[:,r] = np.einsum('abc,bc->a', diff, V[:,r][:,None]*W[:,r][None,:])
                    grad_V[:,r] = np.einsum('abc,ac->b', diff, U[:,r][:,None]*W[:,r][None,:])
                    grad_W[:,r] = np.einsum('abc,ab->c', diff, U[:,r][:,None]*V[:,r][None,:])
                
                # Gradient of penalty
                # d/dx (2x - round(2x))^2 = 2(2x - round(2x)) * 2 = 4(2x - round(2x))
                pen_grad_U = 4 * (2*U - np.round(2*U))
                pen_grad_V = 4 * (2*V - np.round(2*V))
                pen_grad_W = 4 * (2*W - np.round(2*W))
                
                # Combine
                grad_U += 0.1 * pen_grad_U
                grad_V += 0.1 * pen_grad_V
                grad_W += 0.1 * pen_grad_W
                
                return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            x0 = flatten(U_init, V_init, W_init)
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B', jac=gradient,
                                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000})
                
                if result.fun < 1e-6: # Low error + low penalty
                    U_opt, V_opt, W_opt = unflatten(result.x)
                    
                    # Snap to half-integers
                    U_snap = np.round(2 * U_opt) / 2.0
                    V_snap = np.round(2 * V_opt) / 2.0
                    W_snap = np.round(2 * W_opt) / 2.0
                    
                    # Verify exactness
                    T_approx = np.zeros_like(T)
                    for r in range(target_rank):
                        T_approx += np.einsum('a,b,c->abc', U_snap[:,r], V_snap[:,r], W_snap[:,r])
                    
                    if np.max(np.abs(T - T_approx)) < 1e-9:
                        best_result = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                        best_rank = target_rank
                        # Found a valid one, can break out of restart loop
                        break  # Move to next target_rank (lower)
                        
            except Exception as e:
                pass
        
        # If we found a valid decomposition at this rank, we should continue to see if lower is possible
        # But if we are at 23 and found it, we should try 22.
        # The loop continues naturally.
        
    return best_result
