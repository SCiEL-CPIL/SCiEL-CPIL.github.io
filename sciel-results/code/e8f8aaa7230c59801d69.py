"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    nm = n * m  # 16
    mp = m * p  # 28
    pn = p * n  # 28
    
    # Build the target tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    T_flat = T.reshape(nm * mp * pn)
    
    # Schoolbook decomposition as baseline
    rank_school = n * m * p  # 112
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    def compute_error_and_grad(params, target_rank):
        """
        Compute error and gradient for L-BFGS-B.
        params: flattened array of [U, V, W]
        """
        size_U = nm * target_rank
        size_V = mp * target_rank
        size_W = pn * target_rank
        
        U = params[:size_U].reshape(nm, target_rank)
        V = params[size_U:size_U+size_V].reshape(mp, target_rank)
        W = params[size_U+size_V:].reshape(pn, target_rank)
        
        # Compute T_approx
        # T_approx[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Efficiently:
        # Step 1: Compute M = U @ V.T (nm x mp) ? No, that's for matrix mult.
        # For tensor:
        # T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        # This is slow for large R, but R is small (< 112).
        # Let's try to be more efficient.
        # T_approx = np.tensordot(U, V, axes=([1],[1])) -> (nm, mp, R)
        # T_approx = np.tensordot(T_approx, W, axes=([2],[1])) -> (nm, mp, pn)
        
        # Using einsum is often fastest for small arrays in Python if BLAS is used internally?
        # Actually, for R~100, nm=16, mp=28, pn=28.
        # Operations: 16*28*28*100 ~ 1.2M mults. Very fast.
        
        T_approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = T_approx - T
        error = np.sum(diff * diff)
        
        # Gradient
        # dE/dU[a,r] = 2 * sum_{b,c} diff[a,b,c] V[b,r] W[c,r]
        # dE/dV[b,r] = 2 * sum_{a,c} diff[a,b,c] U[a,r] W[c,r]
        # dE/dW[c,r] = 2 * sum_{a,b} diff[a,b,c] U[a,r] V[b,r]
        
        # grad_U = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
        # grad_V = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
        # grad_W = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
        
        grad_U = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
        grad_V = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
        grad_W = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return error, grad

    def try_rank(target_rank, n_restarts=10):
        nonlocal best_U, best_V, best_W, best_rank
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            # Scale is important. 1/sqrt(rank) is a good heuristic.
            scale = 1.0 / np.sqrt(target_rank)
            U = np.random.randn(nm, target_rank) * scale
            V = np.random.randn(mp, target_rank) * scale
            W = np.random.randn(pn, target_rank) * scale
            
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(
                    compute_error_and_grad, 
                    params0, 
                    method='L-BFGS-B', 
                    jac=True,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000}
                )
                
                if res.success:
                    final_params = res.x
                    size_U = nm * target_rank
                    size_V = mp * target_rank
                    U_opt = final_params[:size_U].reshape(nm, target_rank)
                    V_opt = final_params[size_U:size_U+size_V].reshape(mp, target_rank)
                    W_opt = final_params[size_U+size_V:].reshape(pn, target_rank)
                    
                    # Verify
                    T_approx = np.einsum('ar,br,cr->abc', U_opt, V_opt, W_opt, optimize=True)
                    max_err = np.max(np.abs(T_approx - T))
                    
                    if max_err < 1e-9:
                        if target_rank < best_rank:
                            best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
                            best_rank = target_rank
                        return True
                        
            except Exception:
                pass
                
        return False

    # Try ranks from 112 down
    # We know 112 works. We want to find if we can go lower.
    # Start from 111 downwards.
    # To save time, we can skip some ranks if we are confident, but let's try sequentially.
    # If 111 fails, 110 might also fail, but we don't know.
    
    # Optimization: Try 112 first to ensure we have a valid fallback if something goes wrong, 
    # but we already have the schoolbook one. 
    # Let's try 111, 110, ... down to 50.
    
    # If time is tight, we might not finish. 
    # Each minimize call takes ~0.1-0.5s. 
    # 10 restarts * 0.5s = 5s per rank.
    # 140s / 5s = 28 ranks.
    # So we can check ranks 111 down to 83.
    
    for target_rank in range(111, 49, -1):
        if time.time() > deadline - 15:
            break
        if try_rank(target_rank, n_restarts=5):
            # If we found a valid decomposition, we can try to go lower immediately
            # or just return it? 
            # The task asks for rank-minimal. 
            # If we find rank R, we should try R-1.
            # The loop continues, so it will try R-1 next.
            pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
