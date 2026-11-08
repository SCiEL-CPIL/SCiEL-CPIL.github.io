# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    U_mat = np.array(U, dtype=np.float64)
    V_mat = np.array(V, dtype=np.float64)
    W_mat = np.array(W, dtype=np.float64)
    
    # Vectorized verification
    # Reshape to 3D tensors
    # U: (n, m, rank)
    # V: (m, p, rank)
    # W: (p, n, rank)
    
    U_3d = U_mat.reshape(n, m, rank)
    V_3d = V_mat.reshape(m, p, rank)
    W_3d = W_mat.reshape(p, n, rank)
    
    # We need sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1 for all i,j,k
    
    # This is a batched tensor contraction.
    # Result[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Loop over j to keep memory manageable or use einsum
    # einsum path: 'ijr,jkr,kir->ijk'
    # This might be memory heavy for large rank, but rank is <= 120, n=4, m=5, p=6.
    # 4*5*6*120 = 14400 ops. Very fast.
    
    try:
        # Ensure contiguous memory for speed
        U_3d = np.ascontiguousarray(U_3d)
        V_3d = np.ascontiguousarray(V_3d)
        W_3d = np.ascontiguousarray(W_3d)
        
        # W needs to be permuted to align indices for einsum?
        # W is indexed by k, n, r. In W_3d, it is [k, i, r].
        # We want W[k, i, r].
        # U[i, j, r]
        # V[j, k, r]
        
        # Result[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        
        # einsum: 'ijr,jkr,kir->ijk'
        result = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d, optimize=True)
        
        return np.all(np.abs(result - 1.0) < tol)
    except Exception:
        return False

def _round_to_halfint(mat):
    """Round a matrix to nearest half-integer values."""
    return np.round(mat * 2.0) / 2.0

def _try_rank(rank, n, m, p, max_iters=100, n_restarts=10, deadline=None):
    """Try to find an exact decomposition of given rank using corrected ALS with rounding."""
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Random initialization
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Normalize columns to prevent divergence
        for c in range(rank):
            norm_u = np.linalg.norm(U[:, c])
            norm_v = np.linalg.norm(V[:, c])
            norm_w = np.linalg.norm(W[:, c])
            max_norm = max(norm_u, norm_v, norm_w, 1e-10)
            U[:, c] /= max_norm
            V[:, c] /= max_norm
            W[:, c] /= max_norm
            
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline - 5:
                break
                
            # --- Update U ---
            # For each i, j, solve: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all k
            # Let Z[i,j,k,r] = V[j,k,r] W[k,i,r]
            # We want U[i,j,:] such that dot(U[i,j,:], Z[i,j,k,:]) = 1 for all k.
            # This is Z[i,j] @ U[i,j] = ones(p) where Z[i,j] is p x rank.
            # If rank >= p, min norm solution: U[i,j] = Z[i,j].T @ inv(Z[i,j] @ Z[i,j].T) @ ones(p)
            # To avoid inversion, use lstsq or solve.
            
            # Precompute W_3d for reuse? 
            # W is (p, n, rank).
            # V is (m, p, rank).
            
            W_3d = W.reshape(p, n, rank)
            V_3d = V.reshape(m, p, rank)
            U_3d = U.reshape(n, m, rank)
            
            for i in range(n):
                for j in range(m):
                    # Z[i,j,k,r] = V[j,k,r] * W[k,i,r]
                    # V_3d[j] is (p, rank)
                    # W_3d[:, i, :] is (p, rank)
                    Z = V_3d[j] * W_3d[:, i, :] # (p, rank)
                    
                    # Solve Z @ u = ones(p)
                    # Z is p x rank.
                    if rank >= p:
                        # Min norm solution
                        # u = Z.T @ (Z @ Z.T)^-1 @ ones
                        # Use solve for stability
                        try:
                            # Z @ Z.T is p x p
                            ZZT = Z @ Z.T
                            # Add small regularization for stability
                            ZZT += 1e-10 * np.eye(p)
                            u = np.linalg.solve(ZZT, np.ones(p))
                            u = Z.T @ u
                        except np.linalg.LinAlgError:
                            u = np.linalg.lstsq(Z, np.ones(p), rcond=None)[0]
                    else:
                        # LSQ solution
                        u = np.linalg.lstsq(Z, np.ones(p), rcond=None)[0]
                    
                    U_3d[i, j, :] = u
            
            U = U_3d.reshape(n * m, rank)
            
            # --- Update V ---
            # For each j, k, solve: sum_r V[j,k,r] U[i,j,r] W[k,i,r] = 1 for all i
            # Let Z[i,j,k,r] = U[i,j,r] W[k,i,r]
            # We want V[j,k,:] such that dot(V[j,k,:], Z[i,j,k,:]) = 1 for all i.
            # Z[i,j,k] is n x rank.
            # V[j,k] is 1 x rank.
            # Z @ v = ones(n)
            
            U_3d = U.reshape(n, m, rank)
            W_3d = W.reshape(p, n, rank)
            V_3d = V.reshape(m, p, rank)
            
            for j in range(m):
                for k in range(p):
                    # Z[i,j,k,r] = U[i,j,r] * W[k,i,r]
                    # U_3d[:, j, :] is (n, rank)
                    # W_3d[k, :, :] is (n, rank)
                    Z = U_3d[:, j, :] * W_3d[k, :, :] # (n, rank)
                    
                    # Solve Z @ v = ones(n)
                    if rank >= n:
                        try:
                            ZZT = Z @ Z.T
                            ZZT += 1e-10 * np.eye(n)
                            v = np.linalg.solve(ZZT, np.ones(n))
                            v = Z.T @ v
                        except np.linalg.LinAlgError:
                            v = np.linalg.lstsq(Z, np.ones(n), rcond=None)[0]
                    else:
                        v = np.linalg.lstsq(Z, np.ones(n), rcond=None)[0]
                    
                    V_3d[j, k, :] = v
            
            V = V_3d.reshape(m * p, rank)
            
            # --- Update W ---
            # For each k, i, solve: sum_r W[k,i,r] U[i,j,r] V[j,k,r] = 1 for all j
            # Let Z[i,j,k,r] = U[i,j,r] V[j,k,r]
            # We want W[k,i,:] such that dot(W[k,i,:], Z[i,j,k,:]) = 1 for all j.
            # Z[i,j,k] is m x rank.
            # W[k,i] is 1 x rank.
            # Z @ w = ones(m)
            
            U_3d = U.reshape(n, m, rank)
            V_3d = V.reshape(m, p, rank)
            W_3d = W.reshape(p, n, rank)
            
            for k in range(p):
                for i in range(n):
                    # Z[i,j,k,r] = U[i,j,r] * V[j,k,r]
                    # U_3d[i, :, :] is (m, rank)
                    # V_3d[:, k, :] is (m, rank)
                    Z = U_3d[i, :, :] * V_3d[:, k, :] # (m, rank)
                    
                    # Solve Z @ w = ones(m)
                    if rank >= m:
                        try:
                            ZZT = Z @ Z.T
                            ZZT += 1e-10 * np.eye(m)
                            w = np.linalg.solve(ZZT, np.ones(m))
                            w = Z.T @ w
                        except np.linalg.LinAlgError:
                            w = np.linalg.lstsq(Z, np.ones(m), rcond=None)[0]
                    else:
                        w = np.linalg.lstsq(Z, np.ones(m), rcond=None)[0]
                    
                    W_3d[k, i, :] = w
            
            W = W_3d.reshape(p * n, rank)
            
            # Check convergence and exactness periodically
            if it % 5 == 0:
                # Round to half-integers and verify
                U_r = _round_to_halfint(U)
                V_r = _round_to_halfint(V)
                W_r = _round_to_halfint(W)
                if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                    return (U_r, V_r, W_r)
                
                # Also try without rounding (if exact float solution found)
                if _verify_decomposition(U, V, W, n, m, p):
                    return (U, V, W)
        
        # Final attempt with rounding
        U_r = _round_to_halfint(U)
        V_r = _round_to_halfint(V)
        W_r = _round_to_halfint(W)
        if _verify_decomposition(U_r, V_r, W_r, n, m, p):
            return (U_r, V_r, W_r)
    
    return None

def _construct_schoolbook(n, m, p):
    """Construct the schoolbook rank decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    return U, V, W

def construct():
    """Constructs a rank-minimal exact tensor decomposition for 4x5 x 5x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s safety margin
    
    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Try ranks in decreasing order of likelihood?
    # Or start from a reasonable heuristic.
    # Let's try a few specific ranks first, then a broader search.
    # Ranks to try: 100, 90, 80, 70, 60, 50, 40, 30
    
    ranks_to_try = [100, 90, 80, 70, 60, 50, 40, 35, 30]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        if rank >= rank_schoolbook:
            continue
            
        # Allocate time based on rank
        # Lower ranks are harder, give them more time?
        # Or just fixed time per rank.
        # Let's give each rank attempt 10 seconds.
        rank_deadline = time.time() + 10
        
        result = _try_rank(rank, n, m, p, max_iters=200, n_restarts=20, deadline=rank_deadline)
        if result is not None:
            U, V, W = result
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            # Verify once more
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U, V, W = _construct_schoolbook(n, m, p)
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
