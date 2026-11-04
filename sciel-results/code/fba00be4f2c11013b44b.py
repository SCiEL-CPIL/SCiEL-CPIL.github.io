"""Matrix multiplication 4,4,5 - rank-minimal tensor decomposition via NMTF."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the matrix multiplication tensor T[i,j,k] = delta[i*m+j, j*p+k, k*n+i]
    # Actually T is indexed as T[a,b,c] where a in [0,nm), b in [0,mp), c in [0,pn)
    # T[a,b,c] = 1 iff there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # NMTF algorithm
    def nmtf_single(rank, seed, max_iter=200, tol=1e-10):
        rng = np.random.RandomState(seed)
        
        # Initialize U, V, W randomly
        U = rng.randn(nm, rank)
        V = rng.randn(mp, rank)
        W = rng.randn(pn, rank)
        
        prev_err = np.inf
        for it in range(max_iter):
            if time.time() > DEADLINE - 5:
                break
            
            # Alternating least squares: update one factor at a time
            # Update U: for each a, minimize ||T[a,:,:] - sum_r U[a,r]*V[:,r]⊗W[:,r]||^2
            # This is equivalent to: for each a, solve for U[a,:] given V,W
            # T[a,:,:] = sum_r U[a,r] * (V[:,r] ⊗ W[:,r])
            # So U[a,:] = (V⊗W)^+ T[a,:,:]
            
            # More efficient: reshape and use matrix operations
            # For fixed a: T[a,:,:].reshape(mp*pn) = sum_r U[a,r] * (V[:,r]*W[:,r]).reshape(mp*pn)
            # Let M = V * W (element-wise), shape (mp, pn, rank)
            # Then for each a: T[a].reshape(-1) = M.reshape(mp*pn, rank).T @ U[a,:]
            
            # Build design matrix: D[a, b, c, r] = V[b,r]*W[c,r]
            # This is huge. Instead use the structure.
            
            # Update U:
            # For each a in [0, nm):
            #   residual[a] = T[a,:,:] - sum_r U[a,r]*V[:,r][:,None]*W[:,r][None,:]
            #   We want to minimize ||residual[a]||^2 over U[a,:]
            #   This is a linear least squares: sum_r U[a,r] * (V[:,r]⊗W[:,r]) = T[a,:,:]
            #   Let G[a,r] = sum_{b,c} T[a,b,c] * V[b,r]*W[c,r]  (cross-correlation)
            #   Let H[r,s] = sum_{a,b,c} V[b,r]*W[c,r]*V[b,s]*W[c,s]  (gram matrix)
            #   Then H @ U[:,r] ≈ G[:,r]... no, this is per-a.
            
            # Actually the standard NMTF update:
            # Fix V, W. For each a, solve:
            #   min_{u} ||T[a] - sum_r u_r * (V[:,r] ⊗ W[:,r])||_F^2
            #   => sum_r u_r * (V[:,r] ⊗ W[:,r]) = T[a]
            #   This is: (V⊗W)^T (V⊗W) u = (V⊗W)^T T[a]
            #   where (V⊗W) is mp*pn x rank matrix with entry (b*p*n+c, r) = V[b,r]*W[c,r]
            
            # Build the mp*pn x rank matrix
            # M_mat[b*p*n+c, r] = V[b,r] * W[c,r]
            # This is mp*pn x rank = 20*5 x rank = 100 x rank
            # For rank up to ~70, this is manageable.
            
            # M_mat has shape (mp*pn, rank)
            M_mat = np.einsum('br,cr->bcr', V, W).reshape(mp * pn, rank)
            
            # Gram matrix: G = M_mat.T @ M_mat, shape (rank, rank)
            G = M_mat.T @ M_mat
            # Regularize
            G += 1e-8 * np.eye(rank)
            
            # For each a: rhs[a,r] = sum_{b,c} T[a,b,c] * M_mat[b*p*n+c, r]
            # rhs = T.reshape(nm, mp*pn) @ M_mat
            rhs = T.reshape(nm, mp * pn) @ M_mat  # shape (nm, rank)
            
            # Solve G @ U[a,:]^T = rhs[a,:]^T for all a
            # U = rhs @ inv(G)  (each row of rhs solved with G)
            try:
                U_new = rhs @ np.linalg.inv(G)
            except np.linalg.LinAlgError:
                U_new = rhs @ np.linalg.pinv(G)
            
            # Update V
            # Fix U, W. For each b:
            #   min_v ||T[:,b,:] - sum_r v_r * (U[:,r] ⊗ W[:,r])||_F^2
            #   M_mat2[c*p*n+a, r] = U[a,r] * W[c,r]  shape (pn*nm, rank)
            M_mat2 = np.einsum('ar,cr->acr', U, W).reshape(pn * nm, rank)
            G2 = M_mat2.T @ M_mat2 + 1e-8 * np.eye(rank)
            rhs2 = T.reshape(nm, mp, pn).transpose(1, 2, 0).reshape(mp, pn * nm) @ M_mat2
            try:
                V_new = rhs2 @ np.linalg.inv(G2)
            except np.linalg.LinAlgError:
                V_new = rhs2 @ np.linalg.pinv(G2)
            
            # Update W
            # Fix U, V. For each c:
            #   min_w ||T[:,:,c] - sum_r w_r * (U[:,r] ⊗ V[:,r])||_F^2
            #   M_mat3[a*mp+b, r] = U[a,r] * V[b,r]  shape (nm*mp, rank)
            M_mat3 = np.einsum('ar,br->abr', U, V).reshape(nm * mp, rank)
            G3 = M_mat3.T @ M_mat3 + 1e-8 * np.eye(rank)
            rhs3 = T.reshape(nm, mp, pn).transpose(2, 0, 1).reshape(pn, nm * mp) @ M_mat3
            try:
                W_new = rhs3 @ np.linalg.inv(G3)
            except np.linalg.LinAlgError:
                W_new = rhs3 @ np.linalg.pinv(G3)
            
            U, V, W = U_new, V_new, W_new
            
            # Compute error
            err = 0.0
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        val = np.dot(U[a], np.einsum('r,r->', V[b], W[c]))
                        err += (T[a, b, c] - val) ** 2
            err = np.sqrt(err)
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        return U, V, W, err
    
    def check_exact(U, V, W, tol=1e-7):
        """Check if decomposition is exact (within tolerance)."""
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    val = np.dot(U[a], np.einsum('r,r->', V[b], W[c]))
                    if abs(T[a, b, c] - val) > tol:
                        return False
        return True
    
    def try_rank(rank, n_restarts=50):
        best_err = np.inf
        best_factors = None
        for s in range(n_restarts):
            if time.time() > DEADLINE - 10:
                break
            U, V, W, err = nmtf_single(rank, seed=s * 1000 + 42, max_iter=150)
            if err < best_err:
                best_err = err
                best_factors = (U, V, W)
            if best_err < 1e-7:
                break
        return best_factors, best_err
    
    # Try ranks from low to high
    # Known: for (4,4,4) rank is 49 (Strassen-like), for (4,4,5) likely lower than 80
    # Let's try a range of ranks
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Start from a reasonable lower rank and go up
    # For 4x4x5, let's try ranks from 60 upward
    for rank in range(55, n * m * p + 1):
        if time.time() > DEADLINE - 15:
            break
        factors, err = try_rank(rank, n_restarts=30)
        if factors is not None and err < 1e-7:
            U, V, W = factors
            # Round to nearest integer or half-integer if possible
            # First check if it's already close to integers
            best_result = (U, V, W, rank)
            break
    
    # If NMTF didn't find exact, fall back to schoolbook
    if best_result is None:
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        best_result = (U, V, W, rank)
    
    U, V, W, rank = best_result
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
