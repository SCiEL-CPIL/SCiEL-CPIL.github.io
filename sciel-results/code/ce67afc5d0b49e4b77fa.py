import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_school, "V": V_school, "W": W_school}
    
    # We look for ranks starting from the theoretical lower bound
    # For 2x3x3, the lower bound is 13.
    target_ranks = [13, 14, 15, 16, 17]
    
    np.random.seed(42)
    
    for R in target_ranks:
        for restart in range(100):
            # Initialize with random Gaussian noise
            # Scaling is important. If we scale too small, we converge slowly.
            # If too large, we might overshoot.
            # The tensor entries are 0 or 1.
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Normalize initial vectors to prevent numerical instability
            U /= np.linalg.norm(U) * np.sqrt(R)
            V /= np.linalg.norm(V) * np.sqrt(R)
            W /= np.linalg.norm(W) * np.sqrt(R)
            
            converged = False
            # ALS Iterations
            # We want to minimize || T - sum_r u_r v_r^T w_r^T ||_F
            # This is equivalent to finding U, V, W such that
            # sum_r U[:, r] V[:, r]^T W[:, r]^T approx T
            # Note: The tensor T has indices (i, j, k) where T[i, j, k] = 1 if i<m, j<m, k<p? 
            # No, Matrix Mult (A 2x3, B 3x3) -> C 2x3.
            # C[i, k] = sum_j A[i, j] B[j, k].
            # Tensor T[i, j, k] is 1 if the term A[i, j] B[j, k] contributes to C[i, k].
            # So T[i, j, k] = 1 for all valid i, j, k.
            # Wait, the task says "output factor follows official trace convention W[k*n+i, r]".
            # The standard matrix multiplication tensor has T[i, j, k] = 1 for all i in [0, n), j in [0, m), k in [0, p).
            
            for it in range(100):
                # Update U: Fix V, W.
                # For each i, j, we want sum_r U[i, j, r] * (sum_k V[j, k, r] * W[k, i, r]) = 1
                # Let M_U[i, j, r] = sum_k V[j, k, r] * W[k, i, r]
                # We want U[i, j, :] to be the least squares solution to U[i, j, :] dot M_U[i, j, :] = 1
                # This is U[i, j, :] = M_U[i, j, :] / ||M_U[i, j, :]||^2
                
                # Vectorized computation of M_U
                # V is (m*p, R), W is (p*n, R)
                # Reshape V to (m, p, R), W to (p, n, R)
                V_reshaped = V.reshape(m, p, R)
                W_reshaped = W.reshape(p, n, R)
                
                # M_U[i, j, r] = sum_k V[j, k, r] * W[k, i, r]
                # This is a contraction over k.
                # M_U shape: (n, m, R)
                # M_U[i, j, :] = sum_k V_reshaped[j, k, :] * W_reshaped[k, i, :]
                # We can use einsum: 'jkr,ki->ijr'
                M_U = np.einsum('jkr,ki->ijr', V_reshaped, W_reshaped)
                
                # Norm squared of each vector M_U[i, j, :]
                norms_sq_U = np.sum(M_U * M_U, axis=2)
                # Avoid division by zero
                norms_sq_U[norms_sq_U < 1e-12] = 1e-12
                
                U_new = M_U / norms_sq_U[:, :, np.newaxis]
                U = U_new.reshape(n * m, R)
                
                # Update V: Fix U, W.
                # For each j, k, we want sum_r V[j, k, r] * (sum_i U[i, j, r] * W[k, i, r]) = 1
                # M_V[j, k, r] = sum_i U[i, j, r] * W[k, i, r]
                # M_V shape: (m, p, R)
                # U_reshaped: (n, m, R), W_reshaped: (p, n, R)
                # M_V[j, k, :] = sum_i U[i, j, :] * W[k, i, :]
                # einsum: 'ijr,ki->jkr'
                U_reshaped = U.reshape(n, m, R)
                M_V = np.einsum('ijr,ki->jkr', U_reshaped, W_reshaped)
                
                norms_sq_V = np.sum(M_V * M_V, axis=2)
                norms_sq_V[norms_sq_V < 1e-12] = 1e-12
                
                V_new = M_V / norms_sq_V[:, :, np.newaxis]
                V = V_new.reshape(m * p, R)
                
                # Update W: Fix U, V.
                # For each k, i, we want sum_r W[k, i, r] * (sum_j U[i, j, r] * V[j, k, r]) = 1
                # M_W[k, i, r] = sum_j U[i, j, r] * V[j, k, r]
                # M_W shape: (p, n, R)
                # U_reshaped: (n, m, R), V_reshaped: (m, p, R)
                # M_W[k, i, :] = sum_j U[i, j, :] * V[j, k, :]
                # einsum: 'ijr,jk->kir'
                M_W = np.einsum('ijr,jk->kir', U_reshaped, V_reshaped)
                
                norms_sq_W = np.sum(M_W * M_W, axis=2)
                norms_sq_W[norms_sq_W < 1e-12] = 1e-12
                
                W_new = M_W / norms_sq_W[:, :, np.newaxis]
                W = W_new.reshape(p * n, R)
                
                # Check for convergence and exactness
                # Compute the error tensor
                # T_approx[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
                # T_approx shape: (n, m, p)
                T_approx = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
                
                # The target tensor T_target is all 1s of shape (n, m, p)
                # Error = T_approx - 1
                err = np.linalg.norm(T_approx - 1.0)
                
                if err < 1e-9:
                    # Try to snap to half-integers
                    U_rounded = np.round(U * 2) / 2.0
                    V_rounded = np.round(V * 2) / 2.0
                    W_rounded = np.round(W * 2) / 2.0
                    
                    # Verify exactness with rounded values
                    U_r_reshaped = U_rounded.reshape(n, m, R)
                    V_r_reshaped = V_rounded.reshape(m, p, R)
                    W_r_reshaped = W_rounded.reshape(p, n, R)
                    
                    T_approx_r = np.einsum('ijr,jkr,kir->ijk', U_r_reshaped, V_r_reshaped, W_r_reshaped)
                    
                    if np.allclose(T_approx_r, 1.0, atol=1e-9):
                        # Exact solution found
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                    else:
                        # If rounded version doesn't work, we might need to adjust
                        # or accept the float solution if the task allowed floats, 
                        # but it asks for integer/half-integer.
                        # If the float solution is extremely close, maybe the rounding 
                        # was slightly off. We can try to refine the rounding.
                        # For now, we continue searching.
                        pass
                        
            # If we didn't converge to an exact solution in this restart, try next
            pass
            
    return fallback
# EVOLVE-BLOCK-END
