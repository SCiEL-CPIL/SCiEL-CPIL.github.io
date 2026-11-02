"""Tensor decomposition search for 2x3x10 matrix multiplication."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

def _build_tensor(n, m, p):
    """Build the n x m x p matrix multiplication tensor (flattened)."""
    tensor = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i*m+j, j*p+k, k*n+i] = 1.0
    return tensor

def _verify_decomposition_exact(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition using the tensor structure directly."""
    rank = len(U[0])
    nm, mp, pn = n*m, m*p, p*n
    
    # Full reconstruction check
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Check against target
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = T_recon[i*m+j, j*p+k, k*n+i]
                if abs(actual - expected) > tol:
                    return False
    
    # Check that all other entries are approximately 0
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                i = a // m
                j = a % m
                jj = b // p
                k = b % p
                kk = c // n
                ii = c % n
                # The only non-zero entry should be when i==ii, j==jj, k==kk
                if not (i == ii and j == jj and k == kk):
                    if abs(T_recon[a, b, c]) > tol:
                        return False
    return True

def _rank_of_matrix(M):
    """Compute numerical rank of a matrix."""
    try:
        s = np.linalg.svd(M, compute_uv=False)
        return int(np.sum(s > 1e-10))
    except:
        return M.shape[0]

def _als_decomposition(n, m, p, target_rank, T, max_iters=50, tol=1e-10, deadline=None):
    """Run ALS decomposition with multiple random initializations."""
    nm, mp, pn = n*m, m*p, p*n
    
    # Precompute flattened tensors
    T_flat_u = T.reshape(nm, mp * pn)
    T_flat_v = T.reshape(mp, nm * pn)
    T_flat_w = T.reshape(pn, nm * mp)
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for seed in range(30):
        if deadline is not None and time.time() > deadline:
            break
        
        np.random.seed(seed)
        
        # Random initialization
        U = np.random.randn(nm, target_rank)
        V = np.random.randn(mp, target_rank)
        W = np.random.randn(pn, target_rank)
        
        # Normalize columns
        for r in range(target_rank):
            nu = np.linalg.norm(U[:, r])
            nv = np.linalg.norm(V[:, r])
            nw = np.linalg.norm(W[:, r])
            if nu > 1e-15:
                U[:, r] /= nu
            if nv > 1e-15:
                V[:, r] /= nv
            if nw > 1e-15:
                W[:, r] /= nw
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            # Update U
            M_vw = np.einsum('jr,kr->jkr', V, W).reshape(mp * pn, target_rank)
            try:
                U_new = T_flat_u @ np.linalg.pinv(M_vw.T)
            except:
                break
            
            # Update V
            N_uw = np.einsum('ir,kr->ikr', U_new, W).reshape(nm * pn, target_rank)
            try:
                V_new = T_flat_v @ np.linalg.pinv(N_uw.T)
            except:
                break
            
            # Update W
            P_uv = np.einsum('ir,jr->ijr', U_new, V_new).reshape(nm * mp, target_rank)
            try:
                W_new = T_flat_w @ np.linalg.pinv(P_uv.T)
            except:
                break
            
            # Check convergence
            T_check = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
            error = np.max(np.abs(T_check - T))
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
            
            U, V, W = U_new, V_new, W_new
            
            if error < tol:
                break
    
    return best_U, best_V, best_W, best_error

def _structured_init(n, m, p, rank):
    """Create structured initialization based on the tensor's block structure."""
    nm, mp, pn = n*m, m*p, p*n
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col >= rank:
                    break
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
            if col >= rank:
                break
        if col >= rank:
            break
    
    return U, V, W

def _try_round(U, V, W, n, m, p, T, deadline=None):
    """Try rounding to integers or half-integers and verify."""
    nm, mp, pn = n*m, m*p, p*n
    
    # Try integer rounding
    U_int = np.round(U)
    V_int = np.round(V)
    W_int = np.round(W)
    T_check = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)
    if np.max(np.abs(T_check - T)) < 1e-10:
        return U_int, V_int, W_int
    
    # Try half-integer rounding
    U_half = np.round(U * 2) / 2
    V_half = np.round(V * 2) / 2
    W_half = np.round(W * 2) / 2
    T_check = np.einsum('ir,jr,kr->ijk', U_half, V_half, W_half)
    if np.max(np.abs(T_check - T)) < 1e-10:
        return U_half, V_half, W_half
    
    # Try small perturbations around rounded values
    for scale in [1.0, 0.5]:
        U_r = np.round(U * (1/scale)) * scale
        V_r = np.round(V * (1/scale)) * scale
        W_r = np.round(W * (1/scale)) * scale
        T_check = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
        if np.max(np.abs(T_check - T)) < 1e-10:
            return U_r, V_r, W_r
    
    return None, None, None

def _polish(U, V, W, n, m, p, T, max_iters=100, tol=1e-12, deadline=None):
    """High-precision polish pass with tighter tolerance."""
    nm, mp, pn = n*m, m*p, p*n
    rank = U.shape[1]
    
    T_flat_u = T.reshape(nm, mp * pn)
    T_flat_v = T.reshape(mp, nm * pn)
    T_flat_w = T.reshape(pn, nm * mp)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U
        M_vw = np.einsum('jr,kr->jkr', V, W).reshape(mp * pn, rank)
        try:
            U_new = T_flat_u @ np.linalg.pinv(M_vw.T)
        except:
            break
        
        # Update V
        N_uw = np.einsum('ir,kr->ikr', U_new, W).reshape(nm * pn, rank)
        try:
            V_new = T_flat_v @ np.linalg.pinv(N_uw.T)
        except:
            break
        
        # Update W
        P_uv = np.einsum('ir,jr->ijr', U_new, V_new).reshape(nm * mp, rank)
        try:
            W_new = T_flat_w @ np.linalg.pinv(P_uv.T)
        except:
            break
        
        T_check = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
        error = np.max(np.abs(T_check - T))
        
        U, V, W = U_new, V_new, W_new
        
        if error < tol:
            break
    
    return U, V, W, error

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    rank_school = n * m * p  # 60
    
    T = _build_tensor(n, m, p)
    
    # Start with schoolbook decomposition as fallback
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Deadline: 150s limit, use 135s for safety
    deadline = time.time() + 135
    
    # Compute lower bounds from matrix unfoldings
    T_unfold1 = T.reshape(nm, mp * pn)
    T_unfold2 = T.reshape(mp, nm * pn)
    T_unfold3 = T.reshape(pn, nm * mp)
    
    rank_bound1 = _rank_of_matrix(T_unfold1)
    rank_bound2 = _rank_of_matrix(T_unfold2)
    rank_bound3 = _rank_of_matrix(T_unfold3)
    
    min_rank_bound = max(rank_bound1, rank_bound2, rank_bound3)
    
    # Try ranks from min_rank_bound up to schoolbook rank
    for r in range(min_rank_bound, rank_school):
        if time.time() > deadline:
            break
        
        # Try structured initialization first
        U_init, V_init, W_init = _structured_init(n, m, p, r)
        
        # Run ALS with structured initialization
        U_als = U_init.copy()
        V_als = V_init.copy()
        W_als = W_init.copy()
        
        T_flat_u = T.reshape(nm, mp * pn)
        T_flat_v = T.reshape(mp, nm * pn)
        T_flat_w = T.reshape(pn, nm * mp)
        
        found_exact = False
        for it in range(30):
            if time.time() > deadline:
                break
            
            # Update U
            M_vw = np.einsum('jr,kr->jkr', V_als, W_als).reshape(mp * pn, r)
            try:
                U_new = T_flat_u @ np.linalg.pinv(M_vw.T)
            except:
                break
            
            # Update V
            N_uw = np.einsum('ir,kr->ikr', U_new, W_als).reshape(nm * pn, r)
            try:
                V_new = T_flat_v @ np.linalg.pinv(N_uw.T)
            except:
                break
            
            # Update W
            P_uv = np.einsum('ir,jr->ijr', U_new, V_new).reshape(nm * mp, r)
            try:
                W_new = T_flat_w @ np.linalg.pinv(P_uv.T)
            except:
                break
            
            T_check = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
            error = np.max(np.abs(T_check - T))
            
            U_als, V_als, W_als = U_new, V_new, W_new
            
            if error < 1e-10:
                # Try to round
                U_r, V_r, W_r = _try_round(U_als, V_als, W_als, n, m, p, T, deadline)
                if U_r is not None:
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = r
                    found_exact = True
                    break
        
        if found_exact:
            break
        
        # Also try random initializations for this rank
        if time.time() > deadline:
            break
        
        U_rand, V_rand, W_rand, err_rand = _als_decomposition(
            n, m, p, r, T, max_iters=30, tol=1e-10, deadline=deadline
        )
        
        if U_rand is not None and err_rand < 1e-10:
            U_r, V_r, W_r = _try_round(U_rand, V_rand, W_rand, n, m, p, T, deadline)
            if U_r is not None:
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = r
                break
    
    # Final polish if we found a candidate
    if best_rank < rank_school and time.time() < deadline:
        U_pol, V_pol, W_pol, err_pol = _polish(
            best_U, best_V, best_W, n, m, p, T,
            max_iters=50, tol=1e-12, deadline=deadline
        )
        if err_pol < 1e-10:
            U_r, V_r, W_r = _try_round(U_pol, V_pol, W_pol, n, m, p, T, deadline)
            if U_r is not None:
                best_U, best_V, best_W = U_r, V_r, W_r
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
