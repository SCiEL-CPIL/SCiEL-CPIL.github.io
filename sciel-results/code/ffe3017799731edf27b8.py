import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    # Tensor T[a,b,c] where a=i*m+j, b=j'*p+k, c=k'*n+i'
    # T[a,b,c] = 1 iff i=i', j=j', k=k'
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find rank-R decomposition
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Try ranks from low to high
    best_result = None
    
    # First, try a structured approach
    # For 2x4x4, try rank 20
    
    for R in [20, 22, 24, 26, 28, 30, 32]:
        if time.time() > deadline - 20:
            break
        
        # Try to find decomposition with rank R
        # Use alternating minimization
        np.random.seed(42)
        
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(p*n, R)
        
        # Normalize to avoid issues
        for r in range(R):
            norm = np.linalg.norm(W[:, r])
            if norm > 1e-10:
                W[:, r] /= norm
        
        for iteration in range(200):
            if time.time() > deadline - 10:
                break
            
            # Solve for U given V, W
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For each a: T[a,:,:] = sum_r U[a,r] * (V[:,r] * W[:,r])
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # This is: T[a] = U[a,:] @ (V * W) where (V*W)[b,c,r] = V[b,r]*W[c,r]
            # Reshape: for each a, T[a,:,:] should equal sum_r U[a,r] * outer(V[:,r], W[:,r])
            
            # Build the linear system for U
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # Let M[a,b,c,r] = V[b,r] * W[c,r]
            # Then T[a,b,c] = sum_r M[a,b,c,r] * U[a,r]... no
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed a: T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
            # Let X[a,r] = U[a,r], and for each (b,c): T[a,b,c] = sum_r X[a,r] * V[b,r] * W[c,r]
            # This is a linear system: for each a, T[a,:,:] (flattened) = X[a,:] @ A where A[r, (b,c)] = V[b,r]*W[c,r]
            
            A = np.zeros((R, m*p * p*n))
            for r in range(R):
                for b in range(m*p):
                    for c in range(p*n):
                        A[r, b * p*n + c] = V[b, r] * W[c, r]
            
            # For each a: T[a,:,:].flatten() = U[a,:] @ A
            # So U[a,:] = T[a,:,:].flatten() @ A^+ (pseudoinverse)
            # But this is overdetermined, use least squares
            T_flat = T.reshape(n*m, m*p * p*n)
            # U = T_flat @ pinv(A) but A is R x (m*p*p*n), T_flat is (n*m) x (m*p*p*n)
            # We want U such that U @ A ≈ T_flat
            # U = T_flat @ A.T @ inv(A @ A.T) if A @ A.T is invertible
            
            try:
                ATA = A @ A.T
                # Add regularization
                ATA += 1e-10 * np.eye(R)
                U_new = T_flat @ A.T @ np.linalg.inv(ATA)
            except np.linalg.LinAlgError:
                break
            
            # Solve for V given U, W
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed b: T[:,b,c] = sum_r V[b,r] * (U[:,r] * W[c,r])
            B_mat = np.zeros((R, n*m * p*n))
            for r in range(R):
                for a in range(n*m):
                    for c in range(p*n):
                        B_mat[r, a * p*n + c] = U[a, r] * W[c, r]
            
            T_flat2 = T.reshape(m*p, n*m * p*n)
            try:
                BTA = B_mat @ B_mat.T
                BTA += 1e-10 * np.eye(R)
                V_new = T_flat2 @ B_mat.T @ np.linalg.inv(BTA)
            except np.linalg.LinAlgError:
                break
            
            # Solve for W given U, V
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed c: T[a,b,c] = sum_r W[c,r] * (U[a,r] * V[b,r])
            C_mat = np.zeros((R, n*m * m*p))
            for r in range(R):
                for a in range(n*m):
                    for b in range(m*p):
                        C_mat[r, a * m*p + b] = U[a, r] * V[b, r]
            
            T_flat3 = T.reshape(p*n, n*m * m*p)
            try:
                CTA = C_mat @ C_mat.T
                CTA += 1e-10 * np.eye(R)
                W_new = T_flat3 @ C_mat.T @ np.linalg.inv(CTA)
            except np.linalg.LinAlgError:
                break
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_recon - T) / np.linalg.norm(T)
            
            if err < 1e-10:
                # Found a valid decomposition
                best_result = (R, U, V, W)
                break
        
        if best_result is not None:
            break
        if time.time() > deadline - 5:
            break
    
    if best_result is None:
        # Fall back to schoolbook
        R = n * m * p
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        R, U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
