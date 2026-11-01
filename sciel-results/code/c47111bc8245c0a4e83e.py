import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school
    
    # Define the target tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    # ALS function to find a decomposition of given rank
    def als_decompose(rank, max_iters=200, tol=1e-8, seed=0):
        np.random.seed(seed)
        # Random initialization
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Normalize initial factors to prevent divergence
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)

        for it in range(max_iters):
            if time.time() > deadline - 10:
                return None
            
            # Update U: 
            # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # Residual R[a, b, c] = T[a, b, c] - sum_{r != current} ...
            # Actually, we solve for U[a, r] by fixing V, W.
            # For each a, U[a, :] should minimize sum_{b,c} (T[a,b,c] - sum_r' U[a,r'] V[b,r'] W[c,r'])^2
            # This is a linear system for each a? No, it's coupled across r.
            # Let's use the standard ALS approach:
            # Fix V, W. Compute the "effective" target for U.
            # For a fixed 'a', we want to find U[a, :] such that:
            # U[a, :] @ (V.T @ W_reshaped?) ... this is tricky with 3-mode.
            
            # Better approach: 
            # T = U @ V.T @ W.T (in some sense)
            # Actually, T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # Let's reshape T to 2D for each mode.
            
            # Update U:
            # For each a, let T_a be the slice T[a, :, :] of shape (m*p, p*n)
            # We want U[a, :] such that U[a, :] approximates the projection of T_a onto the space spanned by outer products of columns of V and W?
            # No, simpler:
            # Let M_VW be a matrix of shape (m*p, p*n) where M_VW[b, c] = sum_r V[b, r] W[c, r] = (V @ W.T)[b, c]
            # This is not quite right because the sum is over r for the *product*.
            # T[a, b, c] = sum_r U[a, r] (V[b, r] W[c, r])
            # Let Z[b, c, r] = V[b, r] W[c, r].
            # Then T[a, :, :] = sum_r U[a, r] Z[:, :, r]
            # This is a linear combination of rank-1 matrices? No, Z[:, :, r] is a matrix.
            # So T[a, :, :] is in the span of {Z[:, :, r] for r=1..rank}.
            # We can solve for U[a, :] by least squares.
            
            # Flatten Z to shape (m*p*p*n, rank)
            Z_flat = (V[:, :, None] * W[None, :, :]).reshape(m*p*p*n, rank)
            
            # T_flat is shape (n*m, m*p*p*n)
            T_flat = T.reshape(n*m, m*p*p*n)
            
            # Solve T_flat ≈ U @ Z_flat.T
            # U (n*m, rank), Z_flat.T (rank, m*p*p*n)
            # U = T_flat @ Z_flat @ (Z_flat.T @ Z_flat)^{-1}
            
            try:
                ZtZ = Z_flat.T @ Z_flat
                # Add small regularization for stability
                ZtZ += 1e-6 * np.eye(rank)
                U = T_flat @ Z_flat @ np.linalg.inv(ZtZ)
            except np.linalg.LinAlgError:
                return None
            
            # Update V:
            # T[b, a, c] = sum_r V[b, r] U[a, r] W[c, r]
            # Fix U, W.
            # Z2_flat[a, c, r] = U[a, r] W[c, r]
            Z2 = (U[:, :, None] * W[None, :, :]).reshape(m*p*p*n, rank) # Note: U is n*m, W is p*n
            # Wait, indices:
            # T[a, b, c]. 
            # For V[b, r], we fix a, c.
            # T[:, b, :] shape (n*m, p*n).
            # Z2 should be such that T[:, b, :] = sum_r V[b, r] (U[:, r] W[:, r]^T?)
            # Let's check indices.
            # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # Fix b. T[:, b, :] is a matrix of shape (n*m, p*n).
            # Entry (a, c) is sum_r U[a, r] V[b, r] W[c, r].
            # Let Y[a, c] = sum_r U[a, r] W[c, r]. This is (U @ W.T)[a, c].
            # Then T[:, b, :] is NOT simply Y * V[b, :].
            # It is sum_r V[b, r] * (U[:, r] * W[:, r]^T) -- no, U[a,r] and W[c,r] are scalars.
            # The term for a fixed r is U[:, r] (column of U) outer W[:, r] (column of W) ?
            # Let M_r = U[:, r] @ W[:, r].T. This is a matrix of shape (n*m, p*n).
            # Then T[:, b, :] = sum_r V[b, r] M_r.
            # So we form a matrix Z_V of shape (n*m*p*n, rank) where column r is M_r.flatten().
            
            M_r_flat = (U[:, :, None] * W[None, :, :]).reshape(n*m*p*n, rank)
            T_V_flat = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n) # Shape (m*p, n*m*p*n)
            
            try:
                ZtZ_V = M_r_flat.T @ M_r_flat
                ZtZ_V += 1e-6 * np.eye(rank)
                V = T_V_flat @ M_r_flat @ np.linalg.inv(ZtZ_V)
            except np.linalg.LinAlgError:
                return None

            # Update W:
            # T[c, a, b] = sum_r W[c, r] U[a, r] V[b, r]
            # Fix U, V.
            # T[:, :, c] shape (n*m, m*p).
            # Entry (a, b) is sum_r U[a, r] V[b, r] W[c, r].
            # Let N_r = U[:, r] @ V[:, r].T. Shape (n*m, m*p).
            # Then T[:, :, c] = sum_r W[c, r] N_r.
            
            N_r_flat = (U[:, :, None] * V[None, :, :]).reshape(n*m*m*p, rank)
            T_W_flat = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p) # Shape (p*n, n*m*m*p)
            
            try:
                ZtZ_W = N_r_flat.T @ N_r_flat
                ZtZ_W += 1e-6 * np.eye(rank)
                W = T_W_flat @ N_r_flat @ np.linalg.inv(ZtZ_W)
            except np.linalg.LinAlgError:
                return None

            # Check convergence
            # Reconstruct T_recon
            # T_recon[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # Use einsum for speed
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_recon - T)
            
            if err < tol:
                return U, V, W

        return None

    # Try ranks from low to high
    # Lower bound is max(n*m, m*p, p*n) = 9
    # Known good rank is 18
    for target_rank in range(10, rank_school + 1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random seeds
        found = False
        for seed in range(50):
            if time.time() > deadline - 20:
                break
            
            result = als_decompose(target_rank, max_iters=100, seed=seed)
            if result is not None:
                U, V, W = result
                # Final verification
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                if np.max(np.abs(T_recon - T)) < 1e-8:
                    best_rank = target_rank
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    found = True
                    break
        
        if found:
            break

    return best_result
