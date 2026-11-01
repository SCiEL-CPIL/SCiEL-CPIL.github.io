"""Matrix multiplication 2x3x3 rank-minimal exact tensor decomposition search."""
import time
import random
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

DIMENSIONS = [2, 3, 3]
_DEADLINE = 130.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _DEADLINE - _elapsed()

def _is_half_integer(val, tol=1e-5):
    """Check if val is close to a half-integer (k/2 for integer k)."""
    nearest = round(val * 2) / 2
    return abs(val - nearest) < tol

def _all_half_integer(mat, tol=1e-5):
    """Check if all entries in mat are close to half-integers."""
    for val in mat:
        if not _is_half_integer(val, tol):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Tensor T has shape (n*m, m*p, p*n) with T[i*m+j, j*p+k, k*n+i] = 1
    # We seek rank-r decomposition: T = sum_c U[:,c] * V[:,c] * W[:,c]
    # i.e., T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    
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
    
    def _verify_exact(U, V, W, tol=1e-7):
        """Check if the decomposition is exact and entries are integer/half-integer"""
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        if np.max(np.abs(T_approx - T)) > tol:
            return False
        # Check entries are close to integer or half-integer
        for mat in [U, V, W]:
            if not _all_half_integer(mat, tol=1e-5):
                return False
        return True
    
    def _polish(U, V, W, r, max_iter=100):
        """High-precision polish pass with tighter tolerance"""
        params = _flatten_params(U, V, W, r)
        best_obj = _objective(params, r)
        best_params = params.copy()
        
        # Try L-BFGS-B
        try:
            res = minimize(
                _objective, params, args=(r,),
                jac=_objective_grad,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'gtol': 1e-12, 'ftol': 1e-14}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        except Exception:
            pass
        
        # Try Nelder-Mead for small problems as a backup
        if r <= 18 and _remaining() > 5:
            try:
                res2 = minimize(
                    _objective, best_params, args=(r,),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res2.fun < best_obj:
                    best_obj = res2.fun
                    best_params = res2.x
            except Exception:
                pass
        
        if best_obj < 1e-10:
            U_p, V_p, W_p = _unflatten(best_params, r)
            return U_p, V_p, W_p
        return U, V, W
    
    def _round_to_half(U, V, W):
        """Round entries to nearest half-integer"""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        return U_r, V_r, W_r
    
    def _fix_zeros(U, V, W, r):
        """Try to fix near-zero entries to exactly zero and re-verify"""
        U_f = U.copy()
        V_f = V.copy()
        W_f = W.copy()
        
        # Set entries with |val| < 0.01 to 0
        U_f[np.abs(U_f) < 0.01] = 0
        V_f[np.abs(V_f) < 0.01] = 0
        W_f[np.abs(W_f) < 0.01] = 0
        
        # Round to half-integers
        U_f = np.round(U_f * 2) / 2
        V_f = np.round(V_f * 2) / 2
        W_f = np.round(W_f * 2) / 2
        
        return U_f, V_f, W_f
    
    def _structured_init(rank, rng):
        """Generate structured initializations exploiting tensor symmetry."""
        inits = []
        
        # Strategy 1: Based on the known algebraic structure of matmul
        # The tensor T[i*m+j, j*p+k, k*n+i] = 1 has a specific pattern
        # Try to initialize with patterns that respect this
        
        for variant in range(4):
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            if variant == 0:
                # Diagonal-like structure
                for r in range(rank):
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n*m)) % p
                    U[i*m+j, r] = 1.0
                    V[j*p+k, r] = 1.0
                    W[k*n+i, r] = 1.0
                    # Add some random perturbation
                    if r < rank:
                        U[i*m+j, r] += rng.normal(0, 0.1)
                        V[j*p+k, r] += rng.normal(0, 0.1)
                        W[k*n+i, r] += rng.normal(0, 0.1)
            elif variant == 1:
                # Sparse with half-integer values
                for r in range(rank):
                    # Each rank component touches a few entries
                    nz_u = rng.integers(1, 4)
                    nz_v = rng.integers(1, 4)
                    nz_w = rng.integers(1, 4)
                    
                    idx_u = rng.choice(n*m, size=min(nz_u, n*m), replace=False)
                    idx_v = rng.choice(m*p, size=min(nz_v, m*p), replace=False)
                    idx_w = rng.choice(p*n, size=min(nz_w, p*n), replace=False)
                    
                    for idx in idx_u:
                        U[idx, r] = rng.choice([-1.5, -1, -0.5, 0.5, 1, 1.5])
                    for idx in idx_v:
                        V[idx, r] = rng.choice([-1.5, -1, -0.5, 0.5, 1, 1.5])
                    for idx in idx_w:
                        W[idx, r] = rng.choice([-1.5, -1, -0.5, 0.5, 1, 1.5])
            elif variant == 2:
                # Block structure
                for r in range(rank):
                    # Assign blocks
                    i_block = r % n
                    j_block = (r // n) % m
                    k_block = (r // (n*m)) % p
                    
                    U[i_block*m:(i_block+1)*m, r] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
                    V[j_block*p:(j_block+1)*p, r] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
                    W[k_block*n:(k_block+1)*n, r] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
            else:
                # Random with emphasis on the tensor's support pattern
                for r in range(rank):
                    # Focus on entries that are "active" in the tensor
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if rng.random() < 0.2:
                                    U[i*m+j, r] = rng.choice([-1.5, -1, -0.5, 0.5, 1, 1.5])
                                if rng.random() < 0.2:
                                    V[j*p+k, r] = rng.choice([-1.5, -1, -0.5, 0.5, 1, 1.5])
                                if rng.random() < 0.2:
                                    W[k*n+i, r] = rng.choice([-1.5, -1, -0.5, 0.5, 1, 1.5])
            
            inits.append((U, V, W))
        
        return inits
    
    # Try ranks from 17 down to 13
    # Known lower bound for 2x3x3 is around 13
    rng = np.random.default_rng(42)
    
    # Strategy: For each rank, try multiple initialization strategies
    # Focus on structured inits that exploit the tensor's algebraic properties
    
    max_restarts_per_rank = 30
    
    for rank in range(rank_sb - 1, 12, -1):
        if _remaining() < 15:
            break
        
        found = False
        
        # Get structured initializations
        structured_inits = _structured_init(rank, rng)
        
        for restart in range(max_restarts_per_rank):
            if _remaining() < 10:
                break
            
            # Choose initialization strategy
            if restart < len(structured_inits):
                # Use structured initialization
                U, V, W = structured_inits[restart]
            elif restart % 2 == 0:
                # Random Gaussian with moderate scale
                U = rng.standard_normal((n*m, rank)) * 0.3
                V = rng.standard_normal((m*p, rank)) * 0.3
                W = rng.standard_normal((p*n, rank)) * 0.3
            else:
                # Sparse random with half-integer values
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                for r in range(rank):
                    nz_u = rng.integers(1, 4)
                    nz_v = rng.integers(1, 4)
                    nz_w = rng.integers(1, 4)
                    
                    idx_u = rng.choice(n*m, size=min(nz_u, n*m), replace=False)
                    idx_v = rng.choice(m*p, size=min(nz_v, m*p), replace=False)
                    idx_w = rng.choice(p*n, size=min(nz_w, p*n), replace=False)
                    
                    for idx in idx_u:
                        U[idx, r] = rng.choice([-1.5, -1, -0.5, 0.5, 1, 1.5])
                    for idx in idx_v:
                        V[idx, r] = rng.choice([-1.5, -1, -0.5, 0.5, 1, 1.5])
                    for idx in idx_w:
                        W[idx, r] = rng.choice([-1.5, -1, -0.5, 0.5, 1, 1.5])
            
            params = _flatten_params(U, V, W, rank)
            
            try:
                # Try L-BFGS-B
                res = minimize(
                    _objective, params, args=(rank,),
                    jac=_objective_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'gtol': 1e-8, 'ftol': 1e-10}
                )
                
                if res.fun < 1e-6:
                    U_opt, V_opt, W_opt = _unflatten(res.x, rank)
                    
                    # Polish pass
                    U_p, V_p, W_p = _polish(U_opt, V_opt, W_opt, rank, max_iter=100)
                    
                    # Try rounding to half-integers
                    U_r, V_r, W_r = _round_to_half(U_p, V_p, W_p)
                    
                    if _verify_exact(U_r, V_r, W_r, tol=1e-7):
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = rank
                        found = True
                        break
                    
                    # Also try fixing zeros
                    U_f, V_f, W_f = _fix_zeros(U_p, V_p, W_opt, rank)
                    if _verify_exact(U_f, V_f, W_f, tol=1e-7):
                        best_U, best_V, best_W = U_f, V_f, W_f
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
        
        # Final verification
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
