import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 110

def _verify(U, V, W, tol=1e-10):
    # Same as before
    T = np.einsum('ar,br,cr->abc', U, V, W)
    # Check valid entries
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                if abs(T[a, b, c] - 1.0) > tol:
                    return False
    # Check invalid entries
    valid_mask = np.zeros((9, 9, 9), dtype=bool)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                valid_mask[a, b, c] = True
    invalid = T[~valid_mask]
    if np.max(np.abs(invalid)) > tol:
        return False
    return True

def _als_decompose(rank, max_iters=100, tol=1e-8):
    # Initialize U, V, W randomly
    U = np.random.randn(9, rank)
    V = np.random.randn(9, rank)
    W = np.random.randn(9, rank)
    
    # Target tensor M
    M = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                M[a, b, c] = 1.0
                
    for _ in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
            
        # Update U: M = sum_r U[:,r] outer(V[:,r], W[:,r])
        # M_abc = sum_r U_ar V_br W_cr
        # For fixed r, U_ar is the variable.
        # We can reshape M to (9, 9*9) where the second dim is (b,c)
        # M_reshaped[a, b*9+c] = M[a,b,c]
        # We want to solve M_reshaped = U @ (V * W)_reshaped
        # Let X = V * W (element-wise product, then reshaped)
        # X has shape (9*9, rank) ? No.
        # V[:,r] shape (9,), W[:,r] shape (9,).
        # Outer product V[:,r] outer W[:,r] has shape (9,9).
        # We need to form a matrix B of shape (81, rank) where B[(b*9+c), r] = V[b,r] * W[c,r].
        B_mat = np.zeros((81, rank))
        for r in range(rank):
            outer_vw = np.outer(V[:, r], W[:, r])
            B_mat[:, r] = outer_vw.flatten()
            
        M_mat = M.reshape(9, 81).T # shape (81, 9)? No.
        # M_abc. We want to solve for U (9, rank).
        # M_reshaped: index a is row. index (b,c) is col.
        # M_reshaped[a, b*9+c] = M[a,b,c]
        M_mat = M.transpose(0, 1, 2).reshape(9, 81) # shape (9, 81)
        # We want M_mat = U @ B_mat.T ?
        # M_mat[a, idx] = sum_r U[a,r] * B_mat[idx, r]
        # Yes.
        # Solve U @ B_mat.T = M_mat
        # U = M_mat @ pinv(B_mat.T)
        try:
            U_new, _, _, _ = np.linalg.lstsq(B_mat.T, M_mat.T, rcond=None)
            U = U_new.T
        except:
            pass
            
        # Update V
        # M_abc = sum_r U_ar V_br W_cr
        # Fix U, W. Solve for V.
        # M_reshaped2: index b is row. index (a,c) is col.
        # M_reshaped2[b, a*9+c] = M[a,b,c]
        M_mat2 = M.transpose(1, 0, 2).reshape(9, 81)
        B_mat2 = np.zeros((81, rank))
        for r in range(rank):
            outer_uw = np.outer(U[:, r], W[:, r])
            B_mat2[:, r] = outer_uw.flatten()
        try:
            V_new, _, _, _ = np.linalg.lstsq(B_mat2.T, M_mat2.T, rcond=None)
            V = V_new.T
        except:
            pass
            
        # Update W
        # M_reshaped3: index c is row. index (a,b) is col.
        # M_reshaped3[c, a*9+b] = M[a,b,c]
        M_mat3 = M.transpose(2, 0, 1).reshape(9, 81)
        B_mat3 = np.zeros((81, rank))
        for r in range(rank):
            outer_uv = np.outer(U[:, r], V[:, r])
            B_mat3[:, r] = outer_uv.flatten()
        try:
            W_new, _, _, _ = np.linalg.lstsq(B_mat3.T, M_mat3.T, rcond=None)
            W = W_new.T
        except:
            pass
            
        # Check convergence
        T = np.einsum('ar,br,cr->abc', U, V, W)
        if np.linalg.norm(T - M) < tol:
            break
            
    # Round to nearest 0.5
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    if _verify(U_r, V_r, W_r):
        return U_r, V_r, W_r
    return None

def construct():
    # Try ranks from 23 to 26
    for rank in [23, 24, 25, 26]:
        if time.time() > _DEADLINE - 5:
            break
        result = _als_decompose(rank)
        if result:
            U, V, W = result
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
