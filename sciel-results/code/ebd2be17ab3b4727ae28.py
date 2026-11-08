import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j' and k==k' and i==i'
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def try_rank(r, max_iters=500, tol=1e-10):
        """Try to find rank-r decomposition using iterative methods."""
        if time.time() > deadline - 5:
            return None
        
        # Random initialization
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        best_residual = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Alternate minimization
            # Fix V, W; update U: for each row a, U[a,:] = (T[a,:,:] * (V ⊙ W)) / ||...||
            # T[a,:,:] is (m*p, p*n), V⊙W is (m*p, p*n) element-wise product
            VW = V * W  # broadcast: (m*p, r) * (p*n, r) -> need outer product per r
            # Actually: T[a,:,:] = sum_r U[a,r] * V[:,r] * W[:,r]
            # So for fixed V,W: U[a,r] should satisfy T[a,:,:] ≈ sum_r U[a,r] * (V[:,r]*W[:,r])
            # This is a linear system: for each a, T[a,:,:] = U[a,:] @ diag(V*W per element)
            # More precisely: T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # So T[a] = sum_r U[a,r] * (V[:,r] * W[:,r]) where V[:,r]*W[:,r] is element-wise
            # Let M_r = V[:,r] * W[:,r] (shape m*p * p*n)
            # Then T[a] = sum_r U[a,r] * M_r
            # This is: T[a] = U[a] @ M where M has columns M_r... 
            # Actually T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # Reshape: let's define for each r, the matrix M_r[b,c] = V[b,r]*W[c,r]
            # Then T[a] = sum_r U[a,r] * M_r (element-wise, summing over r)
            # So T = sum_r U[:,r] * V[:,r] * W[:,r] (broadcast)
            
            # To update U: fix V, W. For each a:
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # This is a linear system in U[a,:]: 
            # Let A_mat be (m*p*p*n) x r, where A_mat[(b,c), r] = V[b,r]*W[c,r]
            # Then T[a].flatten() = A_mat @ U[a]
            # So U[a] = A_mat^+ @ T[a].flatten()
            
            A_mat = (V * W[:, np.newaxis, :]).reshape(m*p*p*n, r)  # (b,c,r) -> flat
            # Wait, V is (m*p, r), W is (p*n, r). V[:, r] * W[:, r] doesn't make sense directly.
            # V[b, r] * W[c, r] for b in range(m*p), c in range(p*n)
            # So M_r = np.outer(V[:,r], W[:,r]) for each r
            # A_mat[:, r] = M_r.flatten()
            
            A_mat = np.zeros((m*p*p*n, r))
            for r_idx in range(r):
                A_mat[:, r_idx] = np.outer(V[:, r_idx], W[:, r_idx]).flatten()
            
            # Solve: for each a, T[a].flatten() = A_mat @ U[a]
            # U = lstsq(A_mat, T.flatten(1).T).T
            T_flat = T.reshape(n*m, -1)  # (n*m, m*p*p*n)
            try:
                U_new, _, _, _ = np.linalg.lstsq(A_mat.T, T_flat.T, rcond=None)
                U = U_new.T  # (n*m, r)
            except:
                break
            
            # Update V similarly
            B_mat = np.zeros((n*m*p*n, r))
            for r_idx in range(r):
                B_mat[:, r_idx] = np.outer(U[:, r_idx], W[:, r_idx]).flatten()
            T_flat2 = T.reshape(-1, p*n).T.reshape(n*m*p*n, 1)  # hmm, this is getting complex
            
            # Let me use a simpler approach: alternating least squares
            # Update U: fix V, W
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed b,c: T[a,b,c] = sum_r U[a,r] (V[b,r] W[c,r])
            # So for each (b,c), we have a linear system in U[:,r]
            # Let's do: for each r, the contribution is U[:,r] * (V[:,r] * W[:,r])
            # T = sum_r np.einsum('ar,br,cr->abc', U, V, W)
            
            # ALS: update U by fixing V,W
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # Reshape T as (n*m, m*p*p*n)
            # For each a: T[a,:] = sum_r U[a,r] * (V[:,r] * W[:,r]).flatten()
            # Let M = V * W broadcast: M[b,c,r] = V[b,r]*W[c,r], shape (m*p, p*n, r)
            # M_flat = M.reshape(m*p*p*n, r)
            # T[a] = M_flat.T @ U[a]  -> U[a] = lstsq(M_flat.T, T[a])
            
            M = np.einsum('br,cr->bcr', V, W)  # (m*p, p*n, r)
            M_flat = M.reshape(m*p*p*n, r)
            T_2d = T.reshape(n*m, m*p*p*n)
            # T_2d[a, bc] = sum_r U[a,r] * M_flat[bc, r]
            # So T_2d = U @ M_flat.T
            # U = lstsq(M_flat.T, T_2d.T).T
            try:
                U_sol, _, _, _ = np.linalg.lstsq(M_flat.T, T_2d.T, rcond=None)
                U = U_sol.T
            except:
                pass
            
            # Update V: fix U, W
            # T[a,b,c] = sum_r V[b,r] * U[a,r] * W[c,r]
            # For fixed a,c: T[a,b,c] = sum_r V[b,r] * (U[a,r] * W[c,r])
            M2 = np.einsum('ar,cr->acr', U, W)  # (n*m, p*n, r)
            M2_flat = M2.reshape(n*m*p*n, r)
            T_2d2 = T.reshape(m*p, n*m*p*n)  # T[b, a*c]
            # T_2d2[b, ac] = sum_r V[b,r] * M2_flat[ac, r]
            try:
                V_sol, _, _, _ = np.linalg.lstsq(M2_flat.T, T_2d2.T, rcond=None)
                V = V_sol.T
            except:
                pass
            
            # Update W: fix U, V
            # T[a,b,c] = sum_r W[c,r] * U[a,r] * V[b,r]
            M3 = np.einsum('ar,br->abr', U, V)  # (n*m, m*p, r)
            M3_flat = M3.reshape(n*m*m*p, r)
            T_2d3 = T.reshape(p*n, n*m*m*p)  # T[c, a*b]
            try:
                W_sol, _, _, _ = np.linalg.lstsq(M3_flat.T, T_2d3.T, rcond=None)
                W = W_sol.T
            except:
                pass
            
            # Compute residual
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(T - T_recon)
            if residual < best_residual:
                best_residual = residual
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if residual < tol:
                return best_U, best_V, best_W
        
        if best_residual < tol:
            return best_U, best_V, best_W
        return None
    
    # Try ranks from low to high
    for r in range(19, 28):  # Known lower bound for 3x3 is 19
        if time.time() > deadline - 10:
            break
        result = try_rank(r, max_iters=300)
        if result is not None:
            U, V, W = result
            # Convert to lists with integer/half-integer if possible
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
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
