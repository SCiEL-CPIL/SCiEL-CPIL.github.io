"""Matrix multiplication T_{2,3,5} tensor decomposition using ALS and rounding."""
import numpy as np

def construct():
    n, m, p = 2, 3, 5
    dim_a, dim_b, dim_c = n * m, m * p, p * n
    
    # Construct the target tensor T as a vector
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # Flatten index: (a * dim_b + b) * dim_c + c
    T_flat = np.zeros(dim_a * dim_b * dim_c)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                idx = (a * dim_b + b) * dim_c + c
                T_flat[idx] = 1.0
    
    # Try to find a decomposition of rank R
    # Lower bound for R(2,3,5) is not trivially known to be very small, 
    # but schoolbook is 30. Strassen-like methods might reduce it.
    # Let's try ranks from 20 to 30.
    
    for R in range(20, 31):
        # Initialize factors randomly
        np.random.seed(42) # Deterministic for reproducibility in this context
        U = np.random.randn(dim_a, R)
        V = np.random.randn(dim_b, R)
        W = np.random.randn(dim_c, R)
        
        # ALS iterations
        converged = False
        for it in range(100):
            # Solve for U
            # Residual R = T - sum u_r v_r w_r
            # Flatten: vec(R) = sum (w_r \otimes v_r) \otimes u_r
            # We want to minimize || T_flat - U * (W \otimes V).T ||
            # Let M_U be a matrix of shape (dim_b * dim_c, R) where column r is v_r \otimes w_r
            # Then T_flat is approximated by U @ M_U.T ? No.
            # vec(u \otimes v \otimes w) = (w \otimes v) \otimes u
            # So T_flat = sum_r (w_r \otimes v_r) \otimes u_r
            # Let A_r = w_r \otimes v_r (size dim_b * dim_c)
            # T_flat (size dim_a * dim_b * dim_c) can be viewed as a matrix of size (dim_a, dim_b * dim_c)
            # Let T_mat be T reshaped to (dim_a, dim_b * dim_c)
            # T_mat = sum_r u_r (A_r).T = U @ (A_matrix).T ?
            # A_matrix has rows A_r.T?
            # Let's form a matrix M of shape (dim_b * dim_c, R) where column r is vec(v_r \otimes w_r)
            # Then T_mat (dim_a x dim_b*dim_c) = U @ M.T
            # So U = T_mat @ (M.T @ M)^{-1} @ M.T
            
            # Build M for U update
            M_U = np.zeros((dim_b * dim_c, R))
            for r in range(R):
                # vec(v_r \otimes w_r)
                M_U[:, r] = np.kron(V[:, r], W[:, r])
            
            T_mat_U = T_flat.reshape(dim_a, dim_b * dim_c)
            try:
                U = np.linalg.lstsq(M_U.T, T_mat_U.T, rcond=None)[0].T
            except:
                continue
            
            # Solve for V
            # T_flat = sum_r (u_r \otimes w_r) \otimes v_r
            # Reshape T to (dim_b, dim_a * dim_c)
            # Column r of M_V is vec(u_r \otimes w_r)
            M_V = np.zeros((dim_a * dim_c, R))
            for r in range(R):
                M_V[:, r] = np.kron(U[:, r], W[:, r])
            
            T_mat_V = T_flat.reshape(dim_b, dim_a * dim_c)
            try:
                V = np.linalg.lstsq(M_V.T, T_mat_V.T, rcond=None)[0].T
            except:
                continue
            
            # Solve for W
            # T_flat = sum_r (u_r \otimes v_r) \otimes w_r
            # Reshape T to (dim_c, dim_a * dim_b)
            M_W = np.zeros((dim_a * dim_b, R))
            for r in range(R):
                M_W[:, r] = np.kron(U[:, r], V[:, r])
            
            T_mat_W = T_flat.reshape(dim_c, dim_a * dim_b)
            try:
                W = np.linalg.lstsq(M_W.T, T_mat_W.T, rcond=None)[0].T
            except:
                continue
            
            # Check convergence
            # Reconstruct
            recon = np.zeros_like(T_flat)
            for r in range(R):
                recon += np.kron(np.kron(U[:, r], V[:, r]), W[:, r])
            error = np.linalg.norm(T_flat - recon)
            if error < 1e-6:
                converged = True
                break
        
        if converged:
            # Try to round to integers or half-integers
            # Check if entries are close to 0.5 * integer
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify exactness with rounded values
            recon_rounded = np.zeros_like(T_flat)
            for r in range(R):
                recon_rounded += np.kron(np.kron(U_rounded[:, r], V_rounded[:, r]), W_rounded[:, r])
            
            if np.allclose(T_flat, recon_rounded, atol=1e-9):
                # Return the decomposition
                # Convert to lists of floats (or ints if whole)
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook if ALS fails
    rank = n * m * p
    U = [[0] * rank for _ in range(dim_a)]
    V = [[0] * rank for _ in range(dim_b)]
    W = [[0] * rank for _ in range(dim_c)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][column] = 1
                V[b][column] = 1
                W[c][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
