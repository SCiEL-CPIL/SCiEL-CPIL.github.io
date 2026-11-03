"""Matrix multiplication 3x4x7 rank minimization."""
DIMENSIONS = [3, 4, 7]

def construct():
    """Construct a rank-minimal exact tensor decomposition
    for multiplying a 3x4 matrix by a 4x7 matrix.
    
    Strategy: Use higher-order orthogonalization (tensor train / MPO style)
    to find a low-rank decomposition of the matmul tensor. The matmul tensor
    T[i,j,k] = 1 for all i in [0,3), j in [0,4), k in [0,7).
    
    We use alternating minimization: fix U and V, solve for W via least squares
    (exact since the system is determined), then fix V and W, solve for U, etc.
    Start from schoolbook and iteratively reduce rank.
    """
    import time
    import numpy as np
    
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n  # 12, 28, 21
    
    # The matmul tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k)
    # In our index convention:
    #   U index: a = i*m + j  (i in [0,n), j in [0,m))
    #   V index: b = j*p + k  (j in [0,m), k in [0,p))
    #   W index: c = k*n + i  (k in [0,p), i in [0,n))
    # T[a,b,c] = 1 iff there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
    # i.e., j == a%m == b//p and i == a//m == c%n and k == b%p == c//n
    
    # Build the tensor as a 3D array T[a,b,c]
    T = np.zeros((N, M, P), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose T."""
        r = U.shape[1]
        # Compute sum_c U[:,c] * V[:,c] * W[:,c] as tensor
        # T_recon[a,b,c] = sum_c U[a,c]*V[b,c]*W[c,c]
        # Use einsum for efficiency
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.allclose(recon, T, atol=1e-9)
    
    def make_schoolbook():
        """Create schoolbook decomposition (rank = n*m*p)."""
        rank = n * m * p
        U = np.zeros((N, rank), dtype=np.float64)
        V = np.zeros((M, rank), dtype=np.float64)
        W = np.zeros((P, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return U, V, W
    
    def try_rank(r, U_init=None, V_init=None, W_init=None, max_iter=200):
        """Try to find a rank-r decomposition via alternating minimization.
        
        Returns (U, V, W) if found, else None.
        """
        if U_init is None:
            U = np.random.randn(N, r)
            V = np.random.randn(M, r)
            W = np.random.randn(P, r)
        else:
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                return None
            
            # Solve for W: given U, V, find W such that
            # sum_c U[a,c]*V[b,c]*W[c,c] ≈ T[a,b,c]
            # For each c, the contribution is U[:,c] * V[:,c] * W[c,c]
            # This is: for each (a,b), sum_c (U[a,c]*V[b,c]) * W[c,c] = T[a,b,:]
            # Let's matricize: T_ab[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]
            # Fix a,b: T[a,b,:] = (U[a,:] * V[b,:]) @ W.T
            # So T_ab = (U * V) @ W.T where U*V is elementwise... no.
            # T[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
            # Let M_ab[d] = U[a,d] * V[b,d], then T[a,b,c] = sum_d M_ab[d] * W[c,d]
            # So T_ab = M_ab @ W.T
            # For each (a,b): T[a,b,:] = M_ab @ W.T
            # This is underdetermined in general, but we can solve via least squares
            # over all (a,b) pairs.
            
            # Alternative: fix U, V. The residual for W is:
            # R_W = T - sum_c U[:,c]*V[:,c]*W[c,:]  (but W is P x r, so W[c,:] is row c)
            # Actually T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]
            # For fixed U,V, this is linear in W.
            # Let's stack: for each c, T[:,:,c] = sum_d U[:,d]*V[:,d]*W[c,d]
            # T[:,:,c] = W[c,:] @ (U.T * V.T) ... no
            # T[:,:,c] = sum_d W[c,d] * (U[:,d] * V[:,d])
            # So T[:,:,c] is a linear combination of the vectors U[:,d]*V[:,d] with coefficients W[c,d]
            # Let's form: for each c, we need W[c,:] such that sum_d W[c,d] * (U[:,d]*V[:,d]) ≈ T[:,:,c]
            # This is: W[c,:] @ (U * V).T ≈ T[:,:,c].T  (where U*V is elementwise, shape N x r)
            # Let A = U * V (elementwise), shape (N, r)
            # Then for each c: W[c,:] @ A.T ≈ T[:,:,c].T
            # i.e., A @ W[c,:].T ≈ T[:,:,c].T
            # Stack over c: A @ W.T ≈ T.reshape(N, M*P).T ... need to be careful
            
            # Let me use a different approach. Matricize T along the first mode:
            # T_mode1[a, (b,c)] = T[a,b,c]
            # T_mode1 = sum_d U[:,d] * (V[:,d] * W[:,d]).T ... 
            # T[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
            # T_mode1[a, b*M+c] = sum_d U[a,d] * (V[b,d] * W[c,d])
            # Let B_d = V[:,d] * W[:,d] (elementwise, shape M*P if we flatten)
            # Actually let's flatten (b,c) into a single index: bc = b*P + c
            # T_mode1[a, bc] = sum_d U[a,d] * V[b,d] * W[c,d] where bc = b*P+c
            # This isn't simply U @ something because V and W depend on b and c separately.
            
            # Let's just do it directly. For fixed U, V:
            # We want to find W (P x r) minimizing ||T - tensor(U,V,W)||_F
            # This is linear in W. The gradient with respect to W[c,d] is:
            # -2 * sum_{a,b} T[a,b,c] * U[a,d] * V[b,d] + 2 * sum_{a,b} recon[a,b,c] * U[a,d] * V[b,d]
            # At optimum: sum_{a,b} recon[a,b,c] * U[a,d] * V[b,d] = sum_{a,b} T[a,b,c] * U[a,d] * V[b,d]
            # recon[a,b,c] = sum_e U[a,e]*V[b,e]*W[c,e]
            # So: sum_e W[c,e] * (sum_{a,b} U[a,e]*V[b,e]*U[a,d]*V[b,d]) = sum_{a,b} T[a,b,c]*U[a,d]*V[b,d]
            # Let G[e,d] = sum_{a,b} U[a,e]*V[b,e]*U[a,d]*V[b,d] = (U.T @ U) * (V.T @ V) elementwise? 
            # No: sum_{a,b} U[a,e]*U[a,d]*V[b,e]*V[b,d] = (sum_a U[a,e]*U[a,d]) * (sum_b V[b,e]*V[b,d])
            # = (U.T @ U)[e,d] * (V.T @ V)[e,d]
            # So G = (U.T @ U) * (V.T @ V) (elementwise product)
            # And the RHS for each c: RHS[c,d] = sum_{a,b} T[a,b,c]*U[a,d]*V[b,d]
            # = sum_a U[a,d] * (sum_b T[a,b,c]*V[b,d])
            # Let's compute: for each c, RHS_c = U.T @ (V.T @ T[:, :, c].T) ... 
            # T[:, :, c] is (N, M). V.T @ T[:,:,c].T is (r, N). U.T @ that is (r, r)? No.
            # sum_{a,b} T[a,b,c]*U[a,d]*V[b,d] = sum_d' ... let me just compute it.
            
            # Compute G = (U.T @ U) * (V.T @ V), shape (r, r)
            GU = U.T @ U  # (r, r)
            GV = V.T @ V  # (r, r)
            G = GU * GV  # elementwise, (r, r)
            
            # Compute RHS: RHS[c, d] = sum_{a,b} T[a,b,c] * U[a,d] * V[b,d]
            # T is (N, M, P). For each c, T[:,:,c] is (N, M).
            # RHS[:, d] = sum_c ... no, RHS is (P, r)
            # RHS[c, d] = sum_{a=0}^{N-1} sum_{b=0}^{M-1} T[a,b,c] * U[a,d] * V[b,d]
            # = sum_a U[a,d] * (sum_b T[a,b,c] * V[b,d])
            # Let's compute: for each d, U[:,d] is (N,). For each c, we need sum_a U[a,d] * (T[:,:,c] @ V[:,d])
            # T[:,:,c] @ V[:,d] is (N,) vector with entry a = sum_b T[a,b,c]*V[b,d]
            # So RHS[c,d] = U[a,d] dot (T[:,:,c] @ V[:,d]) = sum_a U[a,d] * sum_b T[a,b,c]*V[b,d]
            
            # Efficient: Let's form for each d: V[:,d] (M,). 
            # T @ V[:,d] gives (N, P) matrix with entry (a,c) = sum_b T[a,b,c]*V[b,d]
            # Then RHS[:,d] = U[:,d].T @ (T @ V[:,d]) = sum_a U[a,d] * (T @ V[:,d])[a,:]
            # RHS[:,d] is (P,) vector.
            
            RHS = np.zeros((P, r), dtype=np.float64)
            for d in range(r):
                # T @ V[:,d] is (N, P)
                TV = T @ V[:, d]  # (N, P)
                RHS[:, d] = U[:, d] @ TV  # (P,)
            
            # Solve G @ W[c,:].T = RHS[c,:] for each c
            # i.e., for each c: G @ W[c,:].T = RHS[c,:]
            # W[c,:].T = G^{-1} @ RHS[c,:]
            # Stack: W.T = G^{-1} @ RHS.T, so W = (G^{-1} @ RHS.T).T = RHS @ (G^{-1}).T
            # Since G is symmetric, G^{-1} is symmetric, so W = RHS @ G^{-1}
            
            try:
                G_inv = np.linalg.inv(G)
            except np.linalg.LinAlgError:
                G_inv = np.linalg.pinv(G)
            
            W_new = RHS @ G_inv  # (P, r)
            
            # Check convergence
            if iteration > 0:
                diff = np.linalg.norm(W_new - W) / max(1.0, np.linalg.norm(W))
                if diff < 1e-10:
                    W = W_new
                    break
            W = W_new
            
            # Solve for U: symmetric to W
            # G_U = (V.T @ V) * (W.T @ W)
            GV2 = V.T @ V
            GW = W.T @ W
            G_U = GV2 * GW
            # RHS_U[a, d] = sum_{b,c} T[a,b,c] * V[b,d] * W[c,d]
            RHS_U = np.zeros((N, r), dtype=np.float64)
            for d in range(r):
                # For each d: need sum_{b,c} T[a,b,c]*V[b,d]*W[c,d] for all a
                # T @ (V[:,d] * W[:,d]) ... T is (N,M,P), V[:,d] is (M,), W[:,d] is (P,)
                # sum_{b,c} T[a,b,c]*V[b,d]*W[c,d] = (T * (V[:,d] * W[:,d]))[a,:,:].sum(axis=(1,2))
                # More efficiently: T reshaped or use einsum
                # Let's compute: for each a, sum_{b,c} T[a,b,c]*V[b,d]*W[c,d]
                # = sum_b V[b,d] * (sum_c T[a,b,c]*W[c,d])
                # T @ W[:,d] along last axis: (N, M) with entry (a,b) = sum_c T[a,b,c]*W[c,d]
                TW = np.einsum('abc,c->ab', T, W[:, d])  # (N, M)
                RHS_U[:, d] = TW @ V[:, d]  # (N,)
            
            try:
                G_U_inv = np.linalg.inv(G_U)
            except np.linalg.LinAlgError:
                G_U_inv = np.linalg.pinv(G_U)
            
            U_new = RHS_U @ G_U_inv  # (N, r)
            
            if iteration > 0:
                diff = np.linalg.norm(U_new - U) / max(1.0, np.linalg.norm(U))
                if diff < 1e-10:
                    U = U_new
                    break
            U = U_new
            
            # Solve for V
            G_V = (U.T @ U) * (W.T @ W)
            RHS_V = np.zeros((M, r), dtype=np.float64)
            for d in range(r):
                # sum_{a,c} T[a,b,c]*U[a,d]*W[c,d] for all b
                TU = np.einsum('abc,a->bc', T, U[:, d])  # (M, P)
                RHS_V[:, d] = TU @ W[:, d]  # (M,)
            
            try:
                G_V_inv = np.linalg.inv(G_V)
            except np.linalg.LinAlgError:
                G_V_inv = np.linalg.pinv(G_V)
            
            V_new = RHS_V @ G_V_inv  # (M, r)
            
            if iteration > 0:
                diff = np.linalg.norm(V_new - V) / max(1.0, np.linalg.norm(V))
                if diff < 1e-10:
                    V = V_new
                    break
            V = V_new
        
        # Final verification
        if verify(U, V, W):
            return U, V, W
        return None
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = make_schoolbook()
    
    # Try to reduce rank from 84 downward
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Try various starting ranks
    for target_rank in range(best_rank - 1, max(20, best_rank - 40), -1):
        if time.time() > deadline - 10:
            break
        
        # Multiple random restarts for each target rank
        found = False
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Initialize by truncating SVD of the tensor
            # Matricize T along first mode: T_mode1 (N, M*P)
            T_mode1 = T.reshape(N, M * P)
            # SVD
            try:
                U_svd, s, Vt = np.linalg.svd(T_mode1, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            
            # Truncate to target_rank
            U_trunc = U_svd[:, :target_rank] * s[:target_rank]
            Vt_trunc = Vt[:target_rank, :]
            
            # Vt_trunc is (target_rank, M*P), reshape to (target_rank, M, P)
            Vt_reshaped = Vt_trunc.reshape(target_rank, M, P)
            
            # Now U is (N, target_rank), and we need V (M, target_rank) and W (P, target_rank)
            # T[a,b,c] ≈ sum_d U[a,d] * V[b,d] * W[c,d]
            # From SVD: T[a, bc] = sum_d U_svd[a,d] * s[d] * Vt[d, bc]
            # = sum_d (U_svd[a,d]*s[d]) * Vt[d, b, c]
            # So U_init = U_trunc, and V[b,d]*W[c,d] = Vt[d,b,c]
            # This is a rank-1 decomposition of Vt[d,:,:] for each d, which isn't generally possible
            # So we need to factor Vt_reshaped[d] into V[:,d] * W[:,d]
            # Use SVD of each Vt_reshaped[d]: Vt_reshaped[d] = sum_e sv[d,e] * u[d,e,:] * v[d,e,:]
            # But we need rank-1 per d, so take the top singular vector
            V_init = np.zeros((M, target_rank), dtype=np.float64)
            W_init = np.zeros((P, target_rank), dtype=np.float64)
            for d in range(target_rank):
                mat = Vt_reshaped[d]  # (M, P)
                try:
                    u_d, s_d, vt_d = np.linalg.svd(mat, full_matrices=False)
                    if len(s_d) > 0:
                        V_init[:, d] = u_d[0] * np.sqrt(s_d[0])
                        W_init[:, d] = vt_d[0] * np.sqrt(s_d[0])
                except np.linalg.LinAlgError:
                    V_init[:, d] = 1.0 / np.sqrt(M)
                    W_init[:, d] = 1.0 / np.sqrt(P)
            
            result = try_rank(target_rank, U_trunc, V_init, W_init, max_iter=100)
            if result is not None:
                U, V, W = result
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                found = True
                break
        
        if found:
            # Try to go even lower
            continue
        else:
            # If we couldn't achieve this rank, try lower
            continue
    
    # Convert to lists
    U_list = [[U_list_val for U_list_val in row] for row in best_U.tolist()]
    V_list = [[V_list_val for V_list_val in row] for row in best_V.tolist()]
    W_list = [[W_list_val for W_list_val in row] for row in best_W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
