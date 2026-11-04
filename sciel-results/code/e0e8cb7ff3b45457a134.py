import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28
    
    # Build target tensor
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    schoolbook_rank = n * m * p  # 112
    best_result = None
    best_rank = schoolbook_rank
    
    # Strategy 1: Strassen-based 4x4x4 decomposition (rank 49)
    U_str, V_str, W_str = _strassen_4x4x4()
    
    # Verify Strassen decomposition
    recon_str = np.einsum('ak,bk,ck->abc', U_str, V_str, W_str)
    T_444 = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = i * 4 + j
                b = j * 7 + k
                c = k * 4 + i
                T_444[a, b, c] = 1
    
    strassen_valid = np.allclose(recon_str, T_444, atol=1e-10)
    
    if strassen_valid:
        # Residual is the 4x4x3 part (k=4,5,6)
        R = T - recon_str
        # R should only have nonzero entries where k >= 4
        
        # Try to find a low-rank decomposition of the residual
        residual_result = _find_residual_decomposition(R, dim_u, dim_v, dim_w, deadline - 10.0)
        if residual_result is not None:
            r_res = len(residual_result["U"][0])
            U_res = np.array(residual_result["U"]).T
            V_res = np.array(residual_result["V"]).T
            W_res = np.array(residual_result["W"]).T
            
            U_combined = np.hstack([U_str, U_res])
            V_combined = np.hstack([V_str, V_res])
            W_combined = np.hstack([W_str, W_res])
            
            recon = np.einsum('ak,bk,ck->abc', U_combined, V_combined, W_combined)
            if np.allclose(recon, T, atol=1e-10):
                total_rank = 49 + r_res
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_result = {
                        "U": U_combined.tolist(),
                        "V": V_combined.tolist(),
                        "W": W_combined.tolist()
                    }
    
    # Strategy 2: Direct ALS peeling on full tensor
    if time.time() < deadline - 15.0:
        als_result = _als_peel_full(T, dim_u, dim_v, dim_w, deadline - 5.0, 100)
        if als_result is not None:
            r_als = len(als_result["U"][0])
            if r_als < best_rank:
                best_rank = r_als
                best_result = als_result
    
    # Fallback: schoolbook
    if best_result is None or best_rank >= schoolbook_rank:
        return _schoolbook(n, m, p)
    
    return best_result


def _strassen_4x4x4():
    """Return Strassen-based decomposition of 4x4x4 matrix multiplication.
    
    Uses Kronecker product of 2x2x2 Strassen decomposition.
    Returns U, V, W each of shape (16, 49) such that
    sum_k U[a,k]*V[b,k]*W[c,k] = T[a,b,c] for the 4x4x4 part.
    """
    U_2, V_2, W_2 = _strassen_2x2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            idx = r1 * 7 + r2
            U[:, idx] = np.kron(U_2[:, r1], U_2[:, r2])
            V[:, idx] = np.kron(V_2[:, r1], V_2[:, r2])
            W[:, idx] = np.kron(W_2[:, r1], W_2[:, r2])
    
    # Now we need to map from 4x4x4 standard layout to our 4x4x7 layout
    # U is already correct: index a = i*4+j where i,j in 0..3
    # V needs remapping: standard V index is j*4+k, but we need j*7+k
    # W needs remapping: standard W index is k*4+i, but we need k*4+i (same for 4x4 part)
    
    V_full = np.zeros((28, 49))
    for j in range(4):
        for k in range(4):
            V_full[j*7+k, :] = V[j*4+k, :]
    
    W_full = np.zeros((28, 49))
    for k in range(4):
        for i in range(4):
            W_full[k*4+i, :] = W[k*4+i, :]
    
    return U, V_full, W_full


def _strassen_2x2x2():
    """Return Strassen decomposition of 2x2x2 matrix multiplication.
    
    Returns U, V, W each of shape (4, 7) such that
    sum_k U[a,k]*V[b,k]*W[c,k] = T[a,b,c]
    where T[i*2+j, j*2+k, k*2+i] = 1
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (a21+a22)b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a11(b12-b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = a22(b21-b11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a11+a12)b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (a21-a11)(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (a12-a22)(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _find_residual_decomposition(R, dim_u, dim_v, dim_w, deadline):
    """Find a low-rank decomposition of the residual tensor R.
    
    The residual has a special structure: it's the 4x4x3 part of the 4x4x7 problem.
    We use structured random initialization with rational values and iterative refinement.
    """
    # The residual tensor R[a,b,c] is nonzero only when:
    # a = i*4+j (i,j in 0..3), b = j*7+k (j in 0..3, k in 4..6), c = k*4+i (k in 4..6, i in 0..3)
    # So R has a very sparse structure.
    
    # Try ranks from 12 up to 24 (4*3*4 = 48 schoolbook, but we hope for much less)
    for target_rank in range(12, 25):
        if time.time() > deadline:
            break
        
        result = _try_rank_decomposition(R, dim_u, dim_v, dim_w, target_rank, deadline)
        if result is not None:
            return result
    
    return None


def _try_rank_decomposition(T, dim_u, dim_v, dim_w, rank, deadline):
    """Try to find a rank decomposition of T with the given rank using iterative refinement."""
    
    # Multiple restarts with different initializations
    n_restarts = 5
    best_err = np.inf
    best_U, best_V, best_W = None, None, None
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Initialize with structured random values from {-1, 0, 1}
        U = np.random.choice([-1, 0, 1], size=(dim_u, rank)).astype(np.float64)
        V = np.random.choice([-1, 0, 1], size=(dim_v, rank)).astype(np.float64)
        W = np.random.choice([-1, 0, 1], size=(dim_w, rank)).astype(np.float64)
        
        # Ensure no column is all zeros
        for col in range(rank):
            if np.all(U[:, col] == 0):
                U[np.random.randint(dim_u), col] = 1
            if np.all(V[:, col] == 0):
                V[np.random.randint(dim_v), col] = 1
            if np.all(W[:, col] == 0):
                W[np.random.randint(dim_w), col] = 1
        
        # Iterative refinement using ALS-style updates
        n_iters = 30
        for it in range(n_iters):
            if time.time() > deadline:
                break
            
            # Update U: for each column k, solve least squares
            # T[a,b,c] = sum_k U[a,k]*V[b,k]*W[c,k]
            # For fixed V and W, update U:
            # For each k: U[:,k] = argmin sum_{a,b,c} (T[a,b,c] - sum_{k'} U[a,k']*V[b,k']*W[c,k'])^2
            # This is a linear system: for each k, U[:,k] minimizes ||T - U*V*W||_F
            # Using matrix form: reshape T to (dim_u, dim_v*dim_w)
            
            T_mat = T.reshape(dim_u, dim_v * dim_w)
            VW_mat = np.zeros((dim_v * dim_w, rank))
            for k in range(rank):
                VW_mat[:, k] = np.outer(V[:, k], W[:, k]).reshape(-1)
            
            # Solve U * VW_mat ≈ T_mat
            # U ≈ T_mat * VW_mat^+ (pseudoinverse)
            try:
                U_new, _, _, _ = np.linalg.lstsq(VW_mat, T_mat.T, rcond=None)
                U = U_new.T
            except:
                break
            
            # Update V
            UW_mat = np.zeros((dim_u * dim_w, rank))
            for k in range(rank):
                UW_mat[:, k] = np.outer(U[:, k], W[:, k]).reshape(-1)
            
            T_mat_v = T.reshape(dim_v, dim_u * dim_w)
            try:
                V_new, _, _, _ = np.linalg.lstsq(UW_mat, T_mat_v.T, rcond=None)
                V = V_new.T
            except:
                break
            
            # Update W
            UV_mat = np.zeros((dim_u * dim_v, rank))
            for k in range(rank):
                UV_mat[:, k] = np.outer(U[:, k], V[:, k]).reshape(-1)
            
            T_mat_w = T.reshape(dim_w, dim_u * dim_v)
            try:
                W_new, _, _, _ = np.linalg.lstsq(UV_mat, T_mat_w.T, rcond=None)
                W = W_new.T
            except:
                break
            
            # Check convergence
            recon = np.einsum('ak,bk,ck->abc', U, V, W)
            err = np.linalg.norm(T - recon)
            if err < 1e-8:
                break
        
        # Final check
        recon = np.einsum('ak,bk,ck->abc', U, V, W)
        err = np.linalg.norm(T - recon)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-8:
            break
    
    if best_err < 1e-8 and best_U is not None:
        # Verify exact decomposition
        recon = np.einsum('ak,bk,ck->abc', best_U, best_V, best_W)
        if np.allclose(recon, T, atol=1e-10):
            return {
                "U": best_U.tolist(),
                "V": best_V.tolist(),
                "W": best_W.tolist()
            }
    
    return None


def _als_peel_full(T, dim_u, dim_v, dim_w, deadline, max_terms):
    """ALS-based rank-1 peeling on the full tensor."""
    R = T.copy()
    U_list = []
    V_list = []
    W_list = []
    
    for term_idx in range(max_terms):
        if time.time() > deadline:
            break
        if not np.any(R):
            break
        
        best_err = np.inf
        best_u, best_v, best_w = None, None, None
        
        n_restarts = 4
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            if restart == 0:
                cu = np.sum(R, axis=(1, 2))
                cv = np.sum(R, axis=(0, 2))
                cw = np.sum(R, axis=(0, 1))
            else:
                cu = np.random.randn(dim_u)
                cv = np.random.randn(dim_v)
                cw = np.random.randn(dim_w)
            
            n_iters = 15
            for it in range(n_iters):
                if time.time() > deadline:
                    break
                
                VW = np.outer(cv, cw)
                denom_u = np.sum(VW**2)
                if denom_u > 1e-10:
                    cu = np.einsum('abc,bc->a', R, VW) / denom_u
                else:
                    break
                
                UW = np.outer(cu, cw)
                denom_v = np.sum(UW**2)
                if denom_v > 1e-10:
                    cv = np.einsum('abc,ac->b', R, UW) / denom_v
                else:
                    break
                
                UV = np.outer(cu, cv)
                denom_w = np.sum(UV**2)
                if denom_w > 1e-10:
                    cw = np.einsum('abc,ab->c', R, UV) / denom_w
                else:
                    break
            
            recon = np.einsum('a,b,c->abc', cu, cv, cw)
            err = np.linalg.norm(R - recon)
            if err < best_err:
                best_err = err
                best_u, best_v, best_w = cu.copy(), cv.copy(), cw.copy()
        
        if best_u is None or not np.isfinite(best_err):
            break
        
        R -= np.einsum('a,b,c->abc', best_u, best_v, best_w)
        U_list.append(best_u)
        V_list.append(best_v)
        W_list.append(best_w)
        
        if np.allclose(R, 0, atol=1e-10):
            break
    
    if np.allclose(R, 0, atol=1e-10) and len(U_list) > 0:
        rank = len(U_list)
        U = np.array(U_list).T
        V = np.array(V_list).T
        W = np.array(W_list).T
        
        reconstructed = np.einsum('ak,bk,ck->abc', U, V, W)
        if np.allclose(reconstructed, T, atol=1e-10):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    return None


def _schoolbook(n, m, p):
    """Return schoolbook decomposition of n x m times m x p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
