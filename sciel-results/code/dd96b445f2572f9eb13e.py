"""Matrix multiplication 2x3x3 rank-minimal exact tensor decomposition search."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]
_DEADLINE = 140.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _DEADLINE - _elapsed()

def construct():
    n, m, p = DIMENSIONS
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    T_flat = T.reshape(-1)
    
    # Schoolbook rank = 18
    rank_sb = n * m * p
    U0 = np.zeros((n*m, rank_sb))
    V0 = np.zeros((m*p, rank_sb))
    W0 = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank_sb
    
    def _flatten_params(U, V, W, r):
        return np.concatenate([U.ravel(), V.ravel(), W.ravel()])
    
    def _unflatten(params, r):
        sz_u = n*m*r
        sz_v = m*p*r
        sz_w = p*n*r
        U = params[:sz_u].reshape(n*m, r)
        V = params[sz_u:sz_u+sz_v].reshape(m*p, r)
        W = params[sz_u+sz_v:sz_u+sz_v+sz_w].reshape(p*n, r)
        return U, V, W
    
    def _objective(params, r):
        U, V, W = _unflatten(params, r)
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_approx.reshape(-1) - T_flat
        return np.sum(diff*diff)
    
    def _objective_grad(params, r):
        U, V, W = _unflatten(params, r)
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_approx - T
        gU = np.einsum('abc,br,cr->ar', diff, V, W)
        gV = np.einsum('abc,ar,cr->br', diff, U, W)
        gW = np.einsum('abc,ar,br->cr', diff, U, V)
        return np.concatenate([gU.ravel(), gV.ravel(), gW.ravel()])
    
    def _verify_exact(U, V, W, tol=1e-9):
        """Check if the decomposition is exact and entries are integer/half-integer"""
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        if np.max(np.abs(T_approx - T)) > tol:
            return False
        for mat in [U, V, W]:
            for val in mat:
                nearest = round(val * 2) / 2
                if abs(val - nearest) > 1e-6:
                    return False
        return True
    
    def _round_to_half(U, V, W):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        return U_r, V_r, W_r
    
    def _polish(U, V, W, r, max_iter=200):
        """High-precision polish using L-BFGS-B with tight tolerances"""
        params = _flatten_params(U, V, W, r)
        try:
            res = minimize(
                _objective, params, args=(r,),
                jac=_objective_grad,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'gtol': 1e-12, 'ftol': 1e-15}
            )
            return _unflatten(res.x, r)
        except Exception:
            return U, V, W
    
    # Known lower bound for 2x3x3 matrix multiplication is 13
    # Try ranks from 17 down to 13
    rng = np.random.default_rng(42)
    
    # Pre-compute structured initializations based on tensor structure
    # The tensor has entries at T[i*m+j, j*p+k, k*n+i] = 1
    # We can exploit this by initializing with patterns that respect the structure
    
    def _structured_init(rank, rng):
        """Initialize with structure that respects the tensor's algebraic properties"""
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        for r in range(rank):
            # Each rank-1 component should "touch" a subset of (i,j,k) triples
            # Randomly choose which indices this component contributes to
            n_contrib = max(1, min(3, rng.integers(1, 4)))
            
            # Choose random (i,j,k) indices
            indices = []
            for _ in range(n_contrib):
                i = rng.integers(0, n)
                j = rng.integers(0, m)
                k = rng.integers(0, p)
                indices.append((i, j, k))
            
            # Assign values with half-integer constraint in mind
            for (i, j, k) in indices:
                sign_u = rng.choice([-1.0, 1.0])
                sign_v = rng.choice([-1.0, 1.0])
                sign_w = rng.choice([-1.0, 1.0])
                
                # Use values that are likely to be half-integers after rounding
                val_u = sign_u * rng.choice([0.5, 1.0, 1.5])
                val_v = sign_v * rng.choice([0.5, 1.0, 1.5])
                val_w = sign_w * rng.choice([0.5, 1.0, 1.5])
                
                U[i*m+j, r] += val_u
                V[j*p+k, r] += val_v
                W[k*n+i, r] += val_w
        
        return U, V, W
    
    def _perturbed_schoolbook_init(rank, rng):
        """Start from schoolbook and merge columns"""
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Start with first 'rank' schoolbook columns
        for r in range(min(rank, rank_sb)):
            U[:, r] = U0[:, r]
            V[:, r] = V0[:, r]
            W[:, r] = W0[:, r]
        
        # Add small perturbations to break symmetry
        for r in range(rank):
            U[:, r] += rng.standard_normal(n*m) * 0.1
            V[:, r] += rng.standard_normal(m*p) * 0.1
            W[:, r] += rng.standard_normal(p*n) * 0.1
        
        return U, V, W
    
    max_restarts_per_rank = 50
    
    for rank in range(rank_sb - 1, 12, -1):
        if _remaining() < 20:
            break
        
        found = False
        for restart in range(max_restarts_per_rank):
            if _remaining() < 15:
                break
            
            # Choose initialization strategy
            strategy = restart % 4
            
            if strategy == 0:
                # Structured initialization
                U, V, W = _structured_init(rank, rng)
            elif strategy == 1:
                # Perturbed schoolbook
                U, V, W = _perturbed_schoolbook_init(rank, rng)
            elif strategy == 2:
                # Random Gaussian with moderate scale
                U = rng.standard_normal((n*m, rank)) * 0.3
                V = rng.standard_normal((m*p, rank)) * 0.3
                W = rng.standard_normal((p*n, rank)) * 0.3
            else:
                # Sparse half-integer initialization
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                for r in range(rank):
                    n_nz_u = max(1, rng.integers(1, 4))
                    n_nz_v = max(1, rng.integers(1, 4))
                    n_nz_w = max(1, rng.integers(1, 4))
                    
                    idx_u = rng.choice(n*m, size=n_nz_u, replace=False)
                    idx_v = rng.choice(m*p, size=n_nz_v, replace=False)
                    idx_w = rng.choice(p*n, size=n_nz_w, replace=False)
                    
                    U[idx_u, r] = rng.choice([-1.5, -1, -0.5, 0.5, 1, 1.5])
                    V[idx_v, r] = rng.choice([-1.5, -1, -0.5, 0.5, 1, 1.5])
                    W[idx_w, r] = rng.choice([-1.5, -1, -0.5, 0.5, 1, 1.5])
            
            params = _flatten_params(U, V, W, rank)
            
            try:
                # Quick optimization with L-BFGS-B
                res = minimize(
                    _objective, params, args=(rank,),
                    jac=_objective_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'gtol': 1e-8, 'ftol': 1e-12}
                )
                
                if res.fun < 1e-6:
                    # Good candidate - polish it
                    U_p, V_p, W_p = _polish(
                        _unflatten(res.x, rank)[0],
                        _unflatten(res.x, rank)[1],
                        _unflatten(res.x, rank)[2],
                        rank, max_iter=300
                    )
                    
                    U_r, V_r, W_r = _round_to_half(U_p, V_p, W_p)
                    
                    if _verify_exact(U_r, V_r, W_r, tol=1e-7):
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = rank
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            continue
    
    # If we found a better rank, return it
    if best_rank < rank_sb:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        
        U_v = np.array(U_list)
        V_v = np.array(V_list)
        W_v = np.array(W_list)
        T_approx = np.einsum('ar,br,cr->abc', U_v, V_v, W_v)
        if np.max(np.abs(T_approx - T)) < 1e-7:
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    U_list = [[int(x) for x in row] for row in U0]
    V_list = [[int(x) for x in row] for row in V0]
    W_list = [[int(x) for x in row] for row in W0]
    return {"U": U_list, "V": V_list, "W": W_list}
