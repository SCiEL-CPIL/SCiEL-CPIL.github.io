import numpy as np
import time
import random
from fractions import Fraction
import scipy.optimize as opt

DIMENSIONS = [2, 3, 4]

def _build_tensor(n, m, p):
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    T = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-8):
    """Verify that sum_r U[:,r]*V[:,r]*W[:,r] == T"""
    dim_U = U.shape[0]
    dim_V = V.shape[0]
    dim_W = W.shape[0]
    reconstructed = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
    for r in range(U.shape[1]):
        reconstructed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(reconstructed, T, atol=tol, rtol=tol)

def _als_decompose(T, rank, n_iter=200, tol=1e-10, seed=42):
    """ALS decomposition with normalization and polishing"""
    rng = np.random.RandomState(seed)
    dim_U, dim_V, dim_W = T.shape
    
    # Initialize with random Gaussian
    U = rng.randn(dim_U, rank)
    V = rng.randn(dim_V, rank)
    W = rng.randn(dim_W, rank)
    
    # Normalize columns
    for r in range(rank):
        nU = np.linalg.norm(U[:, r])
        nV = np.linalg.norm(V[:, r])
        nW = np.linalg.norm(W[:, r])
        if nU > 1e-10: U[:, r] /= nU
        if nV > 1e-10: V[:, r] /= nV
        if nW > 1e-10: W[:, r] /= nW

    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(n_iter):
        # Update U
        M = np.zeros((rank, dim_V * dim_W))
        for r in range(rank):
            M[r, :] = np.outer(V[:, r], W[:, r]).flatten()
        T_reshaped = T.reshape(dim_U, dim_V * dim_W)
        U_new = np.linalg.lstsq(M.T, T_reshaped.T, rcond=None)[0].T
        
        # Update V
        M2 = np.zeros((rank, dim_U * dim_W))
        for r in range(rank):
            M2[r, :] = np.outer(U_new[:, r], W[:, r]).flatten()
        T_reshaped2 = T.transpose(1, 0, 2).reshape(dim_V, dim_U * dim_W)
        V_new = np.linalg.lstsq(M2.T, T_reshaped2.T, rcond=None)[0].T
        
        # Update W
        M3 = np.zeros((rank, dim_U * dim_V))
        for r in range(rank):
            M3[r, :] = np.outer(U_new[:, r], V_new[:, r]).flatten()
        T_reshaped3 = T.transpose(2, 0, 1).reshape(dim_W, dim_U * dim_V)
        W_new = np.linalg.lstsq(M3.T, T_reshaped3.T, rcond=None)[0].T
        

        error = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
        for r in range(rank):
            error += np.einsum('a,b,c->abc', U_new[:, r], V_new[:, r], W_new[:, r])
        error -= T
        err_val = np.linalg.norm(error)
        
        if err_val < best_error:
            best_error = err_val
            best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
        
        U, V, W = U_new, V_new, W_new
        
        if err_val < tol:
            break
    
    # Polish step with L-BFGS-B for better convergence
    if best_error < 1.0:
        # Flatten
        x0 = np.concatenate([best_U.flatten(), best_V.flatten(), best_W.flatten()])
        
        def objective(x):
            U_ = x[:dim_U*rank].reshape(dim_U, rank)
            V_ = x[dim_U*rank:dim_U*rank+dim_V*rank].reshape(dim_V, rank)
            W_ = x[dim_U*rank+dim_V*rank:].reshape(dim_W, rank)
            rec = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
            for r in range(rank):
                rec += np.einsum('a,b,c->abc', U_[:, r], V_[:, r], W_[:, r])
            diff = rec - T
            return np.sum(diff**2)
        
        def gradient(x):
            U_ = x[:dim_U*rank].reshape(dim_U, rank)
            V_ = x[dim_U*rank:dim_U*rank+dim_V*rank].reshape(dim_V, rank)
            W_ = x[dim_U*rank+dim_V*rank:].reshape(dim_W, rank)
            
            rec = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
            for r in range(rank):
                rec += np.einsum('a,b,c->abc', U_[:, r], V_[:, r], W_[:, r])
            diff = rec - T
            
            grad_U = np.zeros_like(U_)
            grad_V = np.zeros_like(V_)
            grad_W = np.zeros_like(W_)
            
            for r in range(rank):
                grad_U[:, r] = np.einsum('abc,bc->a', diff, np.outer(V_[:, r], W_[:, r]))
                grad_V[:, r] = np.einsum('abc,ac->b', diff, np.outer(U_[:, r], W_[:, r]))
                grad_W[:, r] = np.einsum('abc,ab->c', diff, np.outer(U_[:, r], V_[:, r]))
            
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        try:
            res = opt.minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                              options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-15})
            if res.fun < best_error:
                best_error = res.fun
                x = res.x
                best_U = x[:dim_U*rank].reshape(dim_U, rank)
                best_V = x[dim_U*rank:dim_U*rank+dim_V*rank].reshape(dim_V, rank)
                best_W = x[dim_U*rank+dim_V*rank:].reshape(dim_W, rank)
        except Exception:
            pass

    return best_U, best_V, best_W, best_error

def _try_rank(T, rank, n_trials=50, time_limit=140.0, start_time=None):
    """Try to find a valid rank-r decomposition"""
    if start_time is None:
        start_time = time.time()
    
    dim_U, dim_V, dim_W = T.shape
    
    for trial in range(n_trials):
        if time.time() - start_time > time_limit - 10.0:
            break
        
        seed = trial * 1000 + 42
        U, V, W, err = _als_decompose(T, rank, n_iter=100, tol=1e-12, seed=seed)
        
        if err < 1e-8:
            # Verify exactly
            if _verify(U, V, W, T, tol=1e-10):
                return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    T = _build_tensor(n, m, p)
    
    start_time = time.time()
    time_limit = 130.0 # Safety margin
    
    # Schoolbook fallback
    schoolbook_rank = n * m * p
    U_sb = np.zeros((dim_U, schoolbook_rank))
    V_sb = np.zeros((dim_V, schoolbook_rank))
    W_sb = np.zeros((dim_W, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Rank 19 is the target. 
    ranks_to_try = [19, 20, 21, 22, 23]
    
    for rank in ranks_to_try:
        if time.time() - start_time > time_limit - 20.0:
            break
        
        # Allocate more trials to rank 19
        trials = 80 if rank == 19 else 20
        result = _try_rank(T, rank, n_trials=trials, time_limit=time_limit, start_time=start_time)
        
        if result is not None:
            U, V, W = result
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = rank
            break
    
    return best_result
