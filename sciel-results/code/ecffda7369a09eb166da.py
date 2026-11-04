import numpy as np
import time
import sys

_START = time.time()
_DEADLINE = _START + 145  # Leave 5s margin for cleanup/return

def _verify(U, V, W, M, tol=1e-9):
    """
    Vectorized verification of the tensor decomposition.
    U: (9, R), V: (9, R), W: (9, R)
    """
    # Compute T
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check difference
    diff = T - M
    max_diff = np.max(np.abs(diff))
    
    if max_diff > tol:
        return False, max_diff
        
    # Check if entries are effectively half-integers
    if not np.allclose(U * 2, np.round(U * 2), atol=1e-6): return False, max_diff
    if not np.allclose(V * 2, np.round(V * 2), atol=1e-6): return False, max_diff
    if not np.allclose(W * 2, np.round(W * 2), atol=1e-6): return False, max_diff

    return True, max_diff

def _polish(U, V, W, M, max_iters=5000, tol=1e-9):
    """
    Local search on the half-integer lattice to minimize ||T - M||.
    Uses incremental error updates for speed.
    """
    def compute_T():
        return np.einsum('ar,br,cr->abc', U, V, W)
    
    T = compute_T()
    current_err = np.linalg.norm(T - M)
    if current_err < tol:
        return U, V, W, True
        
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_T = T.copy()
    
    for _ in range(max_iters):
        if time.time() > _DEADLINE - 2:
            break
            
        improved = False
        
        # Try U changes
        for a in range(9):
            for r in range(U.shape[1]):
                orig = U[a, r]
                # Try orig - 0.5
                delta = -0.5
                # Delta T[a,b,c] = delta * V[b,r] * W[c,r]
                delta_T = np.outer(V[:, r], W[:, r]) * delta
                new_T = best_T.copy()
                new_T[a] += delta_T
                e = np.linalg.norm(new_T - M)
                if e < current_err - 1e-12:
                    U[a, r] = orig + delta
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    best_T = new_T
                    current_err = e
                    improved = True
                    break
                # Try orig + 0.5
                delta = 0.5
                delta_T = np.outer(V[:, r], W[:, r]) * delta
                new_T = best_T.copy()
                new_T[a] += delta_T
                e = np.linalg.norm(new_T - M)
                if e < current_err - 1e-12:
                    U[a, r] = orig + delta
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    best_T = new_T
                    current_err = e
                    improved = True
                    break
            
            if improved:
                U = best_U; V = best_V; W = best_W
                if current_err < tol: return U, V, W, True
                continue
                
        # Try V changes
        for b in range(9):
            for r in range(V.shape[1]):
                orig = V[b, r]
                # Try orig - 0.5
                delta = -0.5
                delta_T = np.outer(U[:, r], W[:, r]) * delta
                new_T = best_T.copy()
                new_T[:, b] += delta_T
                e = np.linalg.norm(new_T - M)
                if e < current_err - 1e-12:
                    V[b, r] = orig + delta
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    best_T = new_T
                    current_err = e
                    improved = True
                    break
                # Try orig + 0.5
                delta = 0.5
                delta_T = np.outer(U[:, r], W[:, r]) * delta
                new_T = best_T.copy()
                new_T[:, b] += delta_T
                e = np.linalg.norm(new_T - M)
                if e < current_err - 1e-12:
                    V[b, r] = orig + delta
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    best_T = new_T
                    current_err = e
                    improved = True
                    break
            
            if improved:
                U = best_U; V = best_V; W = best_W
                if current_err < tol: return U, V, W, True
                continue
                
        # Try W changes
        for c in range(9):
            for r in range(W.shape[1]):
                orig = W[c, r]
                # Try orig - 0.5
                delta = -0.5
                delta_T = np.outer(U[:, r], V[:, r]) * delta
                new_T = best_T.copy()
                new_T[:, :, c] += delta_T
                e = np.linalg.norm(new_T - M)
                if e < current_err - 1e-12:
                    W[c, r] = orig + delta
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    best_T = new_T
                    current_err = e
                    improved = True
                    break
                # Try orig + 0.5
                delta = 0.5
                delta_T = np.outer(U[:, r], V[:, r]) * delta
                new_T = best_T.copy()
                new_T[:, :, c] += delta_T
                e = np.linalg.norm(new_T - M)
                if e < current_err - 1e-12:
                    W[c, r] = orig + delta
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    best_T = new_T
                    current_err = e
                    improved = True
                    break
            
            if improved:
                U = best_U; V = best_V; W = best_W
                if current_err < tol: return U, V, W, True
                continue
                
        if not improved:
            break
            
    return best_U, best_V, best_W, current_err < tol

def _als_decompose(rank, M, max_iters=100, tol=1e-8, init_scale=0.1):
    # Initialize U, V, W with structured random
    U = np.random.randn(9, rank) * init_scale
    V = np.random.randn(9, rank) * init_scale
    W = np.random.randn(9, rank) * init_scale
    
    for _ in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
            
        # Update U
        B_mat = np.zeros((81, rank))
        for r in range(rank):
            B_mat[:, r] = np.outer(V[:, r], W[:, r]).flatten()
        M_mat = M.reshape(9, 81)
        try:
            U_new, _, _, _ = np.linalg.lstsq(B_mat.T, M_mat.T, rcond=None)
            U = U_new.T
        except:
            pass
            
        # Update V
        M_mat2 = M.transpose(1, 0, 2).reshape(9, 81)
        B_mat2 = np.zeros((81, rank))
        for r in range(rank):
            B_mat2[:, r] = np.outer(U[:, r], W[:, r]).flatten()
        try:
            V_new, _, _, _ = np.linalg.lstsq(B_mat2.T, M_mat2.T, rcond=None)
            V = V_new.T
        except:
            pass
            
        # Update W
        M_mat3 = M.transpose(2, 0, 1).reshape(9, 81)
        B_mat3 = np.zeros((81, rank))
        for r in range(rank):
            B_mat3[:, r] = np.outer(U[:, r], V[:, r]).flatten()
        try:
            W_new, _, _, _ = np.linalg.lstsq(B_mat3.T, M_mat3.T, rcond=None)
            W = W_new.T
        except:
            pass
        
        # Check convergence
        T = np.einsum('ar,br,cr->abc', U, V, W)
        if np.linalg.norm(T - M) < tol:
            break
            
    return U, V, W

def construct():
    # Construct Target Tensor M
    M = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                M[a, b, c] = 1.0

    # Try ranks 23, 24, 25
    best_result = None
    
    for rank in [23, 24, 25]:
        if time.time() > _DEADLINE - 5:
            break
            
        found = False
        # More restarts with different initialization scales
        init_scales = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        
        for init_scale in init_scales:
            if time.time() > _DEADLINE - 5:
                break
                
            U, V, W = _als_decompose(rank, M, max_iters=150, init_scale=init_scale)
            
            # Round to nearest 0.5
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            valid, err = _verify(U_r, V_r, W_r, M)
            if valid:
                best_result = (U_r, V_r, W_r, rank)
                found = True
                break
            
            # If not valid, try polishing
            if not found:
                U_p, V_p, W_p, valid_p = _polish(U_r, V_r, W_r, M)
                if valid_p:
                    best_result = (U_p, V_p, W_p, rank)
                    found = True
                    break

        if found:
            break

    if best_result:
        U, V, W, r = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to 27
    U_fb = np.zeros((9, 27), dtype=float)
    V_fb = np.zeros((9, 27), dtype=float)
    W_fb = np.zeros((9, 27), dtype=float)
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_fb[i*3+j, col] = 1.0
                V_fb[j*3+k, col] = 1.0
                W_fb[k*3+i, col] = 1.0
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
