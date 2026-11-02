import time
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq
import random

DIMENSIONS = [2, 5, 5]
DEADLINE_SECONDS = 115.0  # Safety margin before 120s kill

def _build_tensor(n, m, p):
    """
    Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1
    """
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _tensor_error(U, V, W, T):
    """
    Compute ||U V^T W^T - T||_F^2
    T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    """
    reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    return np.sum((reconstructed - T) ** 2)

def _schoolbook_solution(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W

def _try_rank(n, m, p, rank, T, seed=0, time_limit=5.0):
    """
    Attempt to find a decomposition of the given rank.
    Returns (U, V, W, error) if successful, or None.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize randomly
    U = rng.randn(n*m, rank)
    V = rng.randn(m*p, rank)
    W = rng.randn(p*n, rank)
    
    # Normalize
    norms_U = np.linalg.norm(U, axis=0, keepdims=True)
    norms_V = np.linalg.norm(V, axis=0, keepdims=True)
    norms_W = np.linalg.norm(W, axis=0, keepdims=True)
    norms_U[norms_U == 0] = 1.0
    norms_V[norms_V == 0] = 1.0
    norms_W[norms_W == 0] = 1.0
    U /= norms_U
    V /= norms_V
    W /= norms_W

    best_err = _tensor_error(U, V, W, T)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    start_time = time.time()
    iterations = 0
    max_iterations = 2000
    
    while iterations < max_iterations:
        if time.time() - start_time > time_limit:
            break
            
        # Update U
        # T[i, b, c] = sum_r U[i, r] V[b, r] W[c, r]
        # Fix V, W. Solve for U.
        # T_flat[i, (b,c)] = sum_r U[i, r] (V[b, r] W[c, r])
        # Let M_U[(b,c), r] = V[b, r] W[c, r]
        # T_flat[i, :] = U[i, :] M_U
        # U[i, :] = T_flat[i, :] M_U^+
        
        V_rep = V[:, np.newaxis, :]  # (m*p, 1, rank)
        W_rep = W[np.newaxis, :, :]  # (1, p*n, rank)
        MW = V_rep * W_rep  # (m*p, p*n, rank)
        MW_flat = MW.reshape(m*p*p*n, rank)  # (m*p*p*n, rank)
        
        T_flat = T.reshape(n*m, m*p*p*n)
        # Solve U @ MW_flat = T_flat
        # U = T_flat @ pinv(MW_flat)
        U_new, _, _, _ = lstsq(MW_flat, T_flat.T, cond=None)
        U = U_new.T  # (n*m, rank)
        
        # Update V
        # T[i, b, c] = sum_r U[i, r] V[b, r] W[c, r]
        # Fix U, W. Solve for V.
        # T[:, b, :] flattened (n*m * p*n) = V[b, :] M_V
        # M_V[(i, c), r] = U[i, r] W[c, r]
        
        U_rep = U[:, np.newaxis, :]  # (n*m, 1, rank)
        W_rep = W[np.newaxis, :, :]  # (1, p*n, rank)
        MW = U_rep * W_rep  # (n*m, p*n, rank)
        MW_flat = MW.reshape(n*m*p*n, rank)
        
        T_flat_V = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
        V_new, _, _, _ = lstsq(MW_flat, T_flat_V.T, cond=None)
        V = V_new.T  # (m*p, rank)
        
        # Update W
        # T[i, b, c] = sum_r U[i, r] V[b, r] W[c, r]
        # Fix U, V. Solve for W.
        # T[:, :, c] flattened (n*m * m*p) = W[c, :] M_W
        # M_W[(i, b), r] = U[i, r] V[b, r]
        
        U_rep = U[:, np.newaxis, :]  # (n*m, 1, rank)
        V_rep = V[np.newaxis, :, :]  # (1, m*p, rank)
        MW = U_rep * V_rep  # (n*m, m*p, rank)
        MW_flat = MW.reshape(n*m*m*p, rank)
        
        T_flat_W = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p)
        W_new, _, _, _ = lstsq(MW_flat, T_flat_W.T, cond=None)
        W = W_new.T  # (p*n, rank)
        
        iterations += 1
        err = _tensor_error(U, V, W, T)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_err < 1e-10:
            break
            
    # Polish with scipy minimize if we got close
    if best_err < 1e-4 and time.time() - start_time < time_limit - 1:
        params = np.concatenate([best_U.flatten(), best_V.flatten(), best_W.flatten()])
        
        def loss(params):
            U_p = params[:n*m*rank].reshape(n*m, rank)
            V_p = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_p = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
            return _tensor_error(U_p, V_p, W_p, T)
        
        try:
            res = minimize(loss, params, method='BFGS', options={'maxiter': 500, 'gtol': 1e-12})
            if res.fun < best_err:
                params = res.x
                best_U = params[:n*m*rank].reshape(n*m, rank)
                best_V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                best_W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
                best_err = res.fun
        except Exception:
            pass

    return best_U, best_V, best_W, best_err

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook_solution(n, m, p)
    T = _build_tensor(n, m, p)
    
    # Try to reduce rank
    # Known lower bound is 25.
    # We try from 49 down to 25.
    # If we find one, we return it.
    
    candidate_ranks = list(range(49, 24, -1))
    
    for rank in candidate_ranks:
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
            
        success = False
        # Try multiple seeds
        # Use deterministic seeds for reproducibility in search
        for seed in range(10):
            if time.time() - start_time > DEADLINE_SECONDS - 2:
                break
                
            time_limit = 5.0
            try:
                U_new, V_new, W_new, err = _try_rank(n, m, p, rank, T, seed=seed, time_limit=time_limit)
                if err < 1e-8:
                    # Verify
                    if _tensor_error(U_new, V_new, W_new, T) < 1e-10:
                        U, V, W = U_new, V_new, W_new
                        success = True
                        break
            except Exception:
                continue
                
        if success:
            # Found a solution for this rank.
            # Since we are iterating downwards, this is the best so far.
            # We can continue to see if lower rank is possible, but usually
            # if 49 works, 48 is harder. But let's try a bit more if time permits.
            # Actually, if we found 49, we should check 48.
            # But if 49 fails, we should check 48?
            # The loop continues to the next rank (lower) automatically.
            # So if 49 succeeds, we set U,V,W and continue to 48.
            # If 48 fails, we keep 49.
            continue
        else:
            # Failed at this rank.
            # Continue to next lower rank?
            # Yes, it's possible.
            pass
            
    # If we didn't find anything better, return schoolbook
    # (U, V, W are already set to schoolbook if no success)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
