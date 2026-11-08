import numpy as np
import time
import random

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n=3, m=3, p=6):
    """Verify that U, V, W form a valid decomposition."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the target tensor
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
    
    # Compute the decomposed tensor
    T_decomp = np.zeros_like(T)
    for r in range(rank):
        T_decomp += np.einsum('a,b,c->abc', W[:, r], U[:, r], V[:, r])
    
    return np.allclose(T_decomp, T, atol=1e-6)

def _als_decomposition(T, rank, n=3, m=3, p=6, max_iters=100, tol=1e-8):
    """
    Alternating Least Squares for Tensor Decomposition.
    T: (p*n, n*m, m*p)
    Returns U, V, W or None if not converged.
    """
    dim_w, dim_u, dim_v = T.shape
    
    # Initialize using SVD of unfolded matrices
    # Unfold T along axis 0: (dim_w, dim_u * dim_v)
    T_u0 = T.reshape(dim_w, dim_u * dim_v)
    _, _, Vt0 = np.linalg.svd(T_u0, full_matrices=False)
    W = Vt0[:rank, :].T
    
    # Unfold T along axis 1: (dim_u, dim_w * dim_v)
    T_u1 = T.transpose(1, 0, 2).reshape(dim_u, dim_w * dim_v)
    _, _, Vt1 = np.linalg.svd(T_u1, full_matrices=False)
    U = Vt1[:rank, :].T
    
    # Unfold T along axis 2: (dim_v, dim_w * dim_u)
    T_u2 = T.transpose(2, 0, 1).reshape(dim_v, dim_w * dim_u)
    _, _, Vt2 = np.linalg.svd(T_u2, full_matrices=False)
    V = Vt2[:rank, :].T
    
    for it in range(max_iters):
        # Update U
        # T_decomp[b, a, c] = sum_r U[b, r] V[c, r] W[a, r]
        # For fixed r, U[:, r] = (W[:, r]^T V[:, r])^+ T_{:, :, r}^T ?
        # Let's solve for U[:, r] given V, W.
        # T[:, :, :] = sum_r U[:, r] (V[:, r] * W[:, r])^T ? No.
        # T[a, b, c] = sum_r U[b, r] V[c, r] W[a, r]
        # Fix r. We want U[:, r].
        # T[:, b, :] = sum_r' U[b, r'] V[:, r'] W[:, r']^T ?
        # Let's use matrix form.
        # T_{(1)} [b, a*c] = U[:, r] * (V[:, r] * W[:, r]) ?
        # Actually, T_{(1)} = U * (V \circ W)^T ? No.
        # T_{(1)}[b, (a,c)] = sum_r U[b, r] V[c, r] W[a, r]
        # Let M = V * W (element-wise product of V and W transposed? No)
        # M[r, (a,c)] = V[c, r] * W[a, r]
        # Then T_{(1)} = U * M
        # So U = T_{(1)} * M^+
        
        # Construct M for U update
        # M shape: (rank, dim_w * dim_v)
        # M[r, a * dim_v + c] = W[a, r] * V[c, r]
        M = np.zeros((rank, dim_w * dim_v))
        for r in range(rank):
            M[r, :] = (W[:, r] * V[:, r]).reshape(-1) # Wait, W is dim_w x rank, V is dim_v x rank
            # We need to align indices.
            # T[a, b, c]. a is W, b is U, c is V.
            # T_{(1)} is b x (a*c).
            # M[r, idx(a,c)] = W[a, r] * V[c, r]
            # We can compute this by:
            # W_expanded = W[:, :, None] * V[None, :, :] ? No.
            # W is (dim_w, rank). V is (dim_v, rank).
            # We want (dim_w * dim_v, rank) matrix N where N[a, c, r] = W[a, r] V[c, r].
            # N = np.einsum('ar,cr->acr', W, V)
            N = np.einsum('ar,cr->acr', W, V)
            M = N.reshape(dim_w * dim_v, rank).T
            
        # T_u1 is (dim_u, dim_w * dim_v)
        # U = T_u1 @ M^+
        try:
            U_new = T_u1 @ np.linalg.pinv(M)
        except:
            break
            
        # Update V
        # T_{(2)} is c x (a*b).
        # T[c, a, b] = sum_r V[c, r] U[b, r] W[a, r]
        # M_v[r, idx(a,b)] = W[a, r] * U[b, r]
        N = np.einsum('ar,br->abr', W, U)
        M = N.reshape(dim_w * dim_u, rank).T
        T_u2 = T.transpose(2, 0, 1).reshape(dim_v, dim_w * dim_u)
        try:
            V_new = T_u2 @ np.linalg.pinv(M)
        except:
            break
            
        # Update W
        # T_{(0)} is a x (b*c).
        # T[a, b, c] = sum_r W[a, r] U[b, r] V[c, r]
        # M_w[r, idx(b,c)] = U[b, r] * V[c, r]
        N = np.einsum('br,cr->bcr', U, V)
        M = N.reshape(dim_u * dim_v, rank).T
        T_u0 = T.reshape(dim_w, dim_u * dim_v)
        try:
            W_new = T_u0 @ np.linalg.pinv(M)
        except:
            break
            
        U, V, W = U_new, V_new, W_new
        
        # Check error
        T_decomp = np.einsum('ar,br,cr->abc', W, U, V)
        err = np.linalg.norm(T_decomp - T)
        if err < tol:
            return U, V, W
            
    return None

def _schoolbook(n=3, m=3, p=6):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as fallback
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    best_result = (U_fb, V_fb, W_fb, schoolbook_rank)
    
    # Build target tensor once
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0

    # Try to find lower rank decompositions using ALS
    # We try ranks from schoolbook-1 down to a lower bound
    # The lower bound is max(n*m, m*p, p*n) = 18
    # But we know it's likely around 46 (2 * 23)
    # Let's try from 53 down to 40
    
    for rank in range(schoolbook_rank - 1, 39, -1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for each rank
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Add small noise to SVD initialization to escape local minima
            # We'll just call ALS, which uses SVD init internally
            # To add randomness, we can perturb T slightly? No, T is fixed.
            # We can perturb the initial U, V, W.
            # Let's modify _als_decomposition to accept initial values?
            # Or just rely on SVD which is deterministic.
            # To get different results, we can shuffle the columns of T? No.
            # Let's just run ALS. If it fails, try next rank.
            
            res = _als_decomposition(T, rank, n, m, p, max_iters=200)
            if res is not None:
                U, V, W = res
                if _verify(U, V, W, n, m, p):
                    best_result = (U, V, W, rank)
                    found = True
                    break
        
        if found:
            # If we found a rank, we can stop or try to go lower?
            # The loop goes down, so it will try lower ranks automatically.
            # But if we found 53, we try 52, etc.
            pass
        else:
            # If we failed to find rank r, it's likely we can't find r-1 either
            # But ALS might be stuck. Let's break to save time.
            # However, sometimes rank r-1 is easier? Unlikely.
            # Let's break if we fail for a few consecutive ranks?
            # For now, break to save time for other attempts if needed.
            break
            
    U, V, W, rank = best_result
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
