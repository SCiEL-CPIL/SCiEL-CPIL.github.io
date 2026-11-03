"""Matrix multiplication 3x5x5 tensor decomposition."""
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 5
    
    # Build the full multiplication tensor as a 3D array
    # T[a_idx, b_idx, c_idx] = 1 if valid, 0 otherwise
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1
    
    # Try to find a decomposition of rank R
    # We'll use a greedy approach: start with schoolbook and try to merge
    
    def try_rank(R, max_iters=200):
        """Try to find a rank-R decomposition using iterative refinement."""
        if R < 25:  # trivial lower bound is max(n*p, m*p, n*m) = 25
            return None
        
        # Initialize with schoolbook (rank 75) and randomly select R columns
        # Actually, let's start with a smarter initialization
        # For each (i,k) pair, we need contributions from all j
        # Let's try a structured initialization
        
        # Strategy: use R random rank-1 terms and solve for U, V, W
        # We have 75 equations and R*(n*m + m*p + p*n) = R*55 unknowns
        # For R >= 2, this is underdetermined, so we can use least squares
        
        np.random.seed(42)
        
        for attempt in range(max_iters):
            # Initialize U, V, W with random values
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Scale to avoid numerical issues
            U /= np.linalg.norm(U) * 10
            V /= np.linalg.norm(V) * 10
            W /= np.linalg.norm(W) * 10
            
            # Iteratively refine: fix two matrices, solve for the third
            for inner_iter in range(100):
                # Fix V and W, solve for U
                # For each (a_idx), we need: sum_r U[a_idx, r] * V[b_idx, r] * W[c_idx, r] = T[a_idx, b_idx, c_idx]
                # This is: for each a_idx, the vector U[a_idx, :] should satisfy
                # sum_{b_idx, c_idx} T[a_idx, b_idx, c_idx] * V[b_idx, r] * W[c_idx, r] * U[a_idx, r] = T[a_idx, b_idx, c_idx]
                # which simplifies to: for each a_idx, U[a_idx, :] * diag(V*W) * 1 = T[a_idx, :, :] flattened... 
                
                # Actually, let's use the standard approach:
                # The constraint is: T = sum_r U_r ⊗ V_r ⊗ W_r
                # where U_r is the r-th column of U, etc.
                
                # Fix V and W, solve for U:
                # For each r, U_r should be orthogonal to the "residual" in a certain sense
                # Better: use least squares on the unfolded tensor
                
                # Unfold T as a matrix: T_mat[a_idx, (b_idx, c_idx)] = T[a_idx, b_idx, c_idx]
                # Then T_mat = U * (V ⊙ W)^T where ⊙ is Khatri-Rao product
                # V ⊙ W has shape (m*p, R) ⊗ (p*n, R) -> Khatri-Rao is (m*p*p*n, R) but that's too big
                
                # Alternative: use the fact that for fixed r, the contribution to T from term r is
                # outer(U_r, outer(V_r, W_r))
                # So T = sum_r U_r @ (V_r ⊗ W_r)^T reshaped... this is getting complicated.
                
                # Let's use a simpler approach: alternating least squares on the tensor
                # Fix V and W. For each a_idx, we want:
                # sum_r U[a_idx, r] * (V * W)[a_idx, r] = T[a_idx, :, :]
                # where (V*W)[a_idx, r] = sum_{b_idx, c_idx} V[b_idx, r] * W[c_idx, r] * T[a_idx, b_idx, c_idx]
                # Wait, that's not quite right either.
                
                # Let me think more carefully. The tensor T has entries T[a, b, c].
                # We want: T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
                
                # Fix V and W. Then for each a, the equation is:
                # T[a, :, :] = sum_r U[a, r] * (V[:, r] ⊗ W[:, r])
                # where V[:, r] ⊗ W[:, r] is the outer product, a vector of length m*p*p*n
                # But T[a, :, :] is also of length m*p*p*n (flattened)
                
                # So: T[a, :]_flat = sum_r U[a, r] * (V_r ⊗ W_r)_flat
                # This is a linear system: T_flat[a, :] = U[a, :] @ M_r
                # where M_r[b_flat, c_flat] = V[b, r] * W[c, r]
                
                # So M = Khatri-Rao product of V and W, shape (m*p*p*n, R)
                # And T_flat has shape (n*m, m*p*p*n)
                # Then U = T_flat @ pinv(M)
                
                # But m*p*p*n = 5*5*5*3 = 375, so M is 375 x R, T_flat is 15 x 375
                # This is feasible!
                
                # Build M = V ⊙ W (Khatri-Rao)
                # V has shape (m*p, R) = (25, R)
                # W has shape (p*n, R) = (15, R)
                # Khatri-Rao: M has shape (25*15, R) = (375, R)
                M = np.kron(V.flatten(), W.flatten())  # This is wrong, let me do it properly
                
                # Actually, Khatri-Rao product: M[i*m_p + j, r] = V[i, r] * W[j, r]
                # where i is in [0, m*p), j is in [0, p*n)
                M = np.zeros((m*p * p*n, R))
                for r in range(R):
                    M[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                
                # T_flat: shape (n*m, m*p*p*n)
                T_flat = T.reshape(n*m, m*p*p*n)
                
                # Solve: T_flat = U @ M
                # U = T_flat @ M^+ (pseudo-inverse)
                if R <= m*p*p*n:
                    U_new = T_flat @ np.linalg.pinv(M)
                else:
                    U_new = T_flat @ np.linalg.pinv(M)
                
                # Check residual
                T_recon = np.einsum('ar,bcr,acr->abc', U_new, V, W)
                residual = np.linalg.norm(T - T_recon)
                
                if residual < 1e-6:
                    # Convert to exact fractions
                    return _to_exact(U_new, V, W)
                
                # Update U
                U = U_new
                
                # Now fix U and W, solve for V
                M2 = np.zeros((n*m * p*n, R))
                for r in range(R):
                    M2[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                
                # T_flat2: shape (m*p, n*m*p*n)
                T_flat2 = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
                
                V_new = T_flat2 @ np.linalg.pinv(M2)
                
                T_recon = np.einsum('ar,bcr,acr->abc', U, V_new, W)
                residual = np.linalg.norm(T - T_recon)
                
                if residual < 1e-6:
                    return _to_exact(U, V_new, W)
                
                V = V_new
                
                # Fix U and V, solve for W
                M3 = np.zeros((n*m * m*p, R))
                for r in range(R):
                    M3[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                
                T_flat3 = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p)
                
                W_new = T_flat3 @ np.linalg.pinv(M3)
                
                T_recon = np.einsum('ar,bcr,acr->abc', U, V, W_new)
                residual = np.linalg.norm(T - T_recon)
                
                if residual < 1e-6:
                    return _to_exact(U, V, W_new)
                
                W = W_new
            
            # After 100 inner iterations, check final residual
            T_recon = np.einsum('ar,bcr,acr->abc', U, V, W)
            residual = np.linalg.norm(T - T_recon)
            if residual < 1e-6:
                return _to_exact(U, V, W)
        
        return None
    
    def _to_exact(U_float, V_float, W_float):
        """Convert floating point solution to exact fractions."""
        R = U_float.shape[1]
        U_frac = [[Fraction(U_float[i, r]).limit_denominator(1000) for r in range(R)] for i in range(U_float.shape[0])]
        V_frac = [[Fraction(V_float[i, r]).limit_denominator(1000) for r in range(R)] for i in range(V_float.shape[0])]
        W_frac = [[Fraction(W_float[i, r]).limit_denominator(1000) for r in range(R)] for i in range(W_float.shape[0])]
        
        # Verify exactness
        n, m, p = 3, 5, 5
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    total = sum(U_frac[a_idx][r] * V_frac[b_idx][r] * W_frac[c_idx][r] for r in range(R))
                    if total != 1:
                        return None
        
        return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    # Try ranks from 25 upward
    for R in range(25, 50):
        result = try_rank(R, max_iters=50)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
