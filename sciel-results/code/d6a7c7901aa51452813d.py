import time
import numpy as np

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 2, 2, 5
    
    # Build target tensor T
    # T[i, j, k] = 1 if i=a*m+b, j=b*p+d, k=a*p+d
    T = np.zeros((n*m, m*p, p*n))
    for a in range(n):
        for b in range(m):
            for d in range(p):
                i = a * m + b
                j = b * p + d
                k = a * p + d
                T[i, j, k] = 1.0

    # Initialize with Rank 18 solution
    rank_init = 18
    U = np.zeros((n*m, rank_init))
    V = np.zeros((m*p, rank_init))
    W = np.zeros((p*n, rank_init))
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
        
        r = base + 2
        U[0][r] = 1
        V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
        W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 3
        U[3][r] = 1
        V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
        W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
        
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1*p+col1][r] = 1
        W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
        
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
        W[col1*n+1][r] = 1
        
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1

    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    # Schoolbook for column 4
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1

    best_rank = rank_init
    best_U, best_V, best_W = U, V, W

    deadline = time.time() + 140
    
    # Try to reduce rank to 17 by zeroing out one term and re-optimizing
    for drop_idx in range(rank_init):
        if time.time() > deadline:
            break
            
        # Create a copy with one column zeroed
        U_try = U.copy()
        V_try = V.copy()
        W_try = W.copy()
        
        U_try[:, drop_idx] = 0
        V_try[:, drop_idx] = 0
        W_try[:, drop_idx] = 0
        
        # Remove the dropped column to get rank 17 matrices
        mask = [i for i in range(rank_init) if i != drop_idx]
        U_r = U_try[:, mask]
        V_r = V_try[:, mask]
        W_r = W_try[:, mask]
        
        # ALS Optimization
        # We want to minimize || T - sum U_r[:,r] V_r[:,r] W_r[:,r] ||_F
        # Reshape T to (n*m, m*p*p*n)? No, better to use matmul tricks.
        # T is (I, J, K).
        # U is (I, R), V is (J, R), W is (K, R).
        # Tensor product sum: T_approx[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        
        # Fix V, W. Solve for U.
        # U[i,r] * (V[:,r] * W[:,r])_flattened? No.
        # The term for U[i,r] is U[i,r] * (V[:,r] \otimes W[:,r])
        # We can flatten T to (I, J*K).
        # T_flat[i, j*K+k] = T[i,j,k]
        # Then U[i,r] * (V[:,r] \otimes W[:,r])_flat
        # Let M_r = V[:,r] \otimes W[:,r] flattened to (J*K,).
        # Then T_flat[i, :] = sum_r U[i,r] M_r
        # This is a linear system for each i? No, for each r?
        # Actually, for fixed V, W, the equation is:
        # T_flat = U @ M^T  where M is (R, J*K) with rows M_r.
        # So U = T_flat @ pinv(M^T) ?
        # M is (R, J*K). M^T is (J*K, R).
        # We want U (I, R) such that U @ M^T = T_flat (I, J*K).
        # U = T_flat @ (M^T)^+ = T_flat @ pinv(M^T).
        
        T_flat = T.reshape(n*m, m*p * p*n)
        
        # Precompute M for current V, W
        # M[r, j*K+k] = V[j,r] * W[k,r]
        # This is outer product of V[:,r] and W[:,r]
        
        def als_step(U, V, W, T_flat, max_iter=20):
            I, R = U.shape
            J, _ = V.shape
            K, _ = W.shape
            JK = J * K
            
            for it in range(max_iter):
                # Solve for U
                # M[r, :] = outer(V[:,r], W[:,r])
                # M is (R, JK)
                M = np.einsum('jr,kr->rjk', V, W).reshape(R, JK)
                # U = T_flat @ pinv(M.T)
                # M.T is (JK, R)
                try:
                    U_new = T_flat @ np.linalg.pinv(M.T)
                except np.linalg.LinAlgError:
                    break
                # Normalize? Not strictly necessary for ALS, but helps.
                # Actually, we can just use U_new.
                U = U_new
                
                # Solve for V
                # T_flat reshaped to (I*J, K)?
                # T[i,j,k]. Term V[j,r] * (U[i,r] * W[k,r])
                # Let N[j, r] = V[j,r].
                # Equation: T_flat[i, j*K+k] = sum_r V[j,r] * (U[i,r] * W[k,r])
                # Group by j:
                # For each j, T_sub[j, i*K+k] = sum_r V[j,r] * (U[i,r] * W[k,r])
                # Let P[j, i*K+k] = T[i,j,k]
                # Let Q[j, r] = V[j,r]
                # Let A_r[i*K+k] = U[i,r] * W[k,r]
                # Then P = Q @ A^T ?
                # A is (R, I*K). A[r, i*K+k] = U[i,r] * W[k,r]
                # P is (J, I*K). P[j, i*K+k] = T[i,j,k]
                # So Q = P @ pinv(A.T)
                
                A = np.einsum('ir,kr->rik', U, W).reshape(R, I*K)
                P = T.transpose(1, 0, 2).reshape(J, I*K)
                try:
                    V_new = P @ np.linalg.pinv(A.T)
                except np.linalg.LinAlgError:
                    break
                V = V_new
                
                # Solve for W
                # T[i,j,k]. Term W[k,r] * (U[i,r] * V[j,r])
                # Group by k:
                # For each k, T_sub[k, i*J+j] = sum_r W[k,r] * (U[i,r] * V[j,r])
                # Let B[k, r] = W[k,r]
                # Let C_r[i*J+j] = U[i,r] * V[j,r]
                # C is (R, I*J). C[r, i*J+j] = U[i,r] * V[j,r]
                # S is (K, I*J). S[k, i*J+j] = T[i,j,k]
                # B = S @ pinv(C.T)
                
                C = np.einsum('ir,jr->rij', U, V).reshape(R, I*J)
                S = T.transpose(2, 0, 1).reshape(K, I*J)
                try:
                    W_new = S @ np.linalg.pinv(C.T)
                except np.linalg.LinAlgError:
                    break
                W = W_new
                
            return U, V, W

        U_opt, V_opt, W_opt = als_step(U_r, V_r, W_r, T_flat, max_iter=30)
        
        # Calculate error
        # Reconstruct T_approx
        T_approx = np.einsum('ir,jr,kr->ijk', U_opt, V_opt, W_opt)
        error = np.linalg.norm(T - T_approx)
        
        if error < 1e-6:
            best_rank = 17
            best_U = U_opt
            best_V = V_opt
            best_W = W_opt
            break # Found a valid rank 17 decomposition
            
    # Return the best result
    # Convert to lists of lists for the interface
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END
