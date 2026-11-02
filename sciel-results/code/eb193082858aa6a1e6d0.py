import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    u_rows = n * m  # 9
    v_rows = m * p  # 15
    w_rows = p * n  # 15
    
    # Build the target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    T = np.zeros((u_rows, v_rows, w_rows))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Precompute constraint indices for quick verification
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                constraints.append((u_idx, v_idx, w_idx))
    
    def verify_exact(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the tensor."""
        for u_idx, v_idx, w_idx in constraints:
            val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
            if abs(val - 1.0) > tol:
                return False
        return True
    
    def round_and_check(U, V, W):
        """Round to half-integers and verify exactness."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if verify_exact(U_r, V_r, W_r):
            return U_r, V_r, W_r
        
        # Also try integer rounding
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if verify_exact(U_i, V_i, W_i):
            return U_i, V_i, W_i
        
        return None
    
    def als_solve(U, V, W, T, max_iters=200, tol=1e-10):
        """Alternating Least Squares for tensor decomposition."""
        for it in range(max_iters):
            # Update U: for each a, solve G_U @ U[a,:] = h_U[a,:]
            # where h_U[a,d] = sum_{b,c} T[a,b,c] * V[b,d] * W[c,d]
            # and G_U[d1,d2] = sum_{b,c} V[b,d1]*W[c,d1]*V[b,d2]*W[c,d2]
            
            # Vectorized computation
            # TW[b,c,d] = V[b,d] * W[c,d]
            TW = V[:, None, :] * W[None, :, :]  # (v_rows, w_rows, r)
            
            # h_U[a,d] = sum_{b,c} T[a,b,c] * TW[b,c,d]
            h_U = np.einsum('abc,bcd->ad', T, TW)  # (u_rows, r)
            
            # G_U[d1,d2] = sum_{b,c} TW[b,c,d1] * TW[b,c,d2]
            G_U = np.einsum('bcd,bce->de', TW, TW)  # (r, r)
            
            try:
                U_new = np.linalg.solve(G_U, h_U.T).T
            except np.linalg.LinAlgError:
                U_new = h_U
            
            # Update V
            UW = U[:, None, :] * W[None, :, :]  # (u_rows, w_rows, r)
            h_V = np.einsum('abc,acd->bd', T, UW)  # (v_rows, r)
            G_V = np.einsum('acd,ace->de', UW, UW)  # (r, r)
            
            try:
                V_new = np.linalg.solve(G_V, h_V.T).T
            except np.linalg.LinAlgError:
                V_new = h_V
            
            # Update W
            UV = U[:, None, :] * V[None, :, :]  # (u_rows, v_rows, r)
            h_W = np.einsum('abc,abd->cd', T, UV)  # (w_rows, r)
            G_W = np.einsum('abd,abe->de', UV, UV)  # (r, r)
            
            try:
                W_new = np.linalg.solve(G_W, h_W.T).T
            except np.linalg.LinAlgError:
                W_new = h_W
            
            # Check convergence
            diff = max(
                np.max(np.abs(U_new - U)),
                np.max(np.abs(V_new - V)),
                np.max(np.abs(W_new - W))
            )
            U, V, W = U_new, V_new, W_new
            
            if diff < tol:
                break
        
        return U, V, W
    
    # Schoolbook solution as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((u_rows, rank_sb))
    V_sb = np.zeros((v_rows, rank_sb))
    W_sb = np.zeros((w_rows, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Try ranks from 15 (lower bound) up to 44
    # Focus on promising ranges: 20-35 likely contains the minimum
    for target_rank in range(15, rank_sb):
        if time.time() > deadline:
            break
        
        if target_rank >= best_rank:
            break
        
        # Try multiple random initializations
        for seed in range(20):
            if time.time() > deadline:
                break
            
            np.random.seed(seed + target_rank * 100)
            
            # Initialize with small random values
            scale = 0.1
            U = np.random.randn(u_rows, target_rank) * scale
            V = np.random.randn(v_rows, target_rank) * scale
            W = np.random.randn(w_rows, target_rank) * scale
            
            # Run ALS
            U, V, W = als_solve(U, V, W, T, max_iters=100)
            
            # Try to round and verify
            result = round_and_check(U, V, W)
            if result is not None:
                U_r, V_r, W_r = result
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    break  # Found solution for this rank, try lower
        
        # Early exit if we found a very good rank
        if best_rank <= 20:
            break
    
    # Final verification
    if not verify_exact(best_U, best_V, best_W):
        # Fallback to schoolbook
        best_U = U_sb
        best_V = V_sb
        best_W = W_sb
    
    # Convert to lists
    U_list = [[float(v) for v in row] for row in best_U.tolist()]
    V_list = [[float(v) for v in row] for row in best_V.tolist()]
    W_list = [[float(v) for v in row] for row in best_W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
