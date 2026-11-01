import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110  # safety margin before 120s limit
    
    # First, build the known rank-56 Strassen solution as fallback
    rank_56 = 56
    U56 = [[0] * rank_56 for _ in range(n * m)]
    V56 = [[0] * rank_56 for _ in range(m * p)]
    W56 = [[0] * rank_56 for _ in range(p * n)]
    
    u_vecs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_vecs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_vecs = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for r in range(8):
        base_col = 7 * r
        for s in range(7):
            col = base_col + s
            for li in range(4):
                if u_vecs[s][li] != 0:
                    U56[li][col] = u_vecs[s][li]
            v_indices = [2*r, 2*r+1, 16+2*r, 16+2*r+1]
            for li in range(4):
                if v_vecs[s][li] != 0:
                    V56[v_indices[li]][col] = v_vecs[s][li]
            w_indices = [4*r, 4*r+2, 4*r+1, 4*r+3]
            for li in range(4):
                if w_vecs[s][li] != 0:
                    W56[w_indices[li]][col] = w_vecs[s][li]
    
    # Verify rank-56 solution
    def _verify(U, V, W):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        T_recon = np.einsum('as,bs,cs->abc', Ua, Va, Wa)
        # Build target tensor
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a = i * m + k
                    b = k * p + j
                    c = j * n + i
                    T[a, b, c] = 1.0
        return np.allclose(T_recon, T, atol=1e-8)
    
    if not _verify(U56, V56, W56):
        # Should not happen, but return anyway
        return {"U": U56, "V": V56, "W": W56}
    
    # Try to find a lower rank solution using iterative approach
    # Strategy: try rank 55 by solving a least-squares problem
    # Initialize from rank-56 by zeroing one column and solving
    
    def _try_rank(r, U_init=None, V_init=None, W_init=None, max_iter=200):
        """Try to find a rank-r decomposition using ALS (alternating least squares)"""
        if U_init is not None:
            Ua = np.array(U_init, dtype=float).copy()
            Va = np.array(V_init, dtype=float).copy()
            Wa = np.array(W_init, dtype=float).copy()
        else:
            np.random.seed(42)
            Ua = np.random.randn(n*m, r) * 0.1
            Va = np.random.randn(m*p, r) * 0.1
            Wa = np.random.randn(p*n, r) * 0.1
        
        # Build target tensor
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a = i * m + k
                    b = k * p + j
                    c = j * n + i
                    T[a, b, c] = 1.0
        
        T_flat = T.reshape(-1)
        
        for it in range(max_iter):
            if time.time() > deadline:
                return None, None, None
            
            # Update U: for each a, solve for U[a,:] given V and W
            # T[a,b,c] = sum_s U[a,s] * V[b,s] * W[c,s]
            # For fixed a: T[a,:,:].reshape(-1) = (V^T ⊗ W^T) @ U[a,:]
            # Actually: sum_s U[a,s] * V[b,s] * W[c,s] = U[a,:] @ (V[b,:] * W[c,:])
            # Reshape: T[a] (size m*p*p*n) = (V ⊙ W) @ U[a,:] where ⊙ is Khatri-Rao
            # V ⊙ W: each column s is V[:,s] ⊗ W[:,s], size m*p*p*n
            # So T[a] = KR(V,W) @ U[a,:]
            
            # Build Khatri-Rao product of V and W
            # KR(V,W) has shape (m*p * p*n, r)
            # Column s = V[:,s] ⊗ W[:,s]
            KR_VW = np.einsum('bs,cs->bc s', Va, Wa).reshape(m*p*p*n, r)
            
            # Solve for each row of U
            for a in range(n*m):
                T_a = T[a].reshape(-1)
                # T_a = KR_VW @ U[a,:]
                # U[a,:] = (KR_VW^T KR_VW)^{-1} KR_VW^T T_a
                # Use lstsq for stability
                Ua[a, :], residuals, rank_s, sv = np.linalg.lstsq(KR_VW, T_a, rcond=None)
            
            # Update V
            KR_UW = np.einsum('as,cs->ac s', Ua, Wa).reshape(n*m*p*n, r)
            for b in range(m*p):
                T_b = T[:, b, :].reshape(-1)
                Va[b, :], residuals, rank_s, sv = np.linalg.lstsq(KR_UW, T_b, rcond=None)
            
            # Update W
            KR_UV = np.einsum('as,bs->ab s', Ua, Va).reshape(n*m*m*p, r)
            for c in range(p*n):
                T_c = T[:, :, c].reshape(-1)
                Wa[c, :], residuals, rank_s, sv = np.linalg.lstsq(KR_UV, T_c, rcond=None)
            
            # Check convergence
            T_recon = np.einsum('as,bs,cs->abc', Ua, Va, Wa)
            err = np.linalg.norm(T_recon - T)
            if err < 1e-10:
                return Ua, Va, Wa
        
        # Check final error
        T_recon = np.einsum('as,bs,cs->abc', Ua, Va, Wa)
        err = np.linalg.norm(T_recon - T)
        if err < 1e-8:
            return Ua, Va, Wa
        return None, None, None
    
    best_result = {"U": U56, "V": V56, "W": W56}
    
    # Try rank 55
    if time.time() < deadline - 5:
        # Initialize by taking rank-56 and zeroing the last column
        U55_init = [row[:-1] for row in U56]
        V55_init = [row[:-1] for row in V56]
        W55_init = [row[:-1] for row in W56]
        
        U55, V55, W55 = _try_rank(55, U55_init, V55_init, W55_init, max_iter=100)
        if U55 is not None:
            # Convert to list of lists with integer/half-integer rounding
            U55_list = [[round(float(x)) if abs(float(x) - round(float(x))) < 1e-6 else float(x) for x in row] for row in U55]
            V55_list = [[round(float(x)) if abs(float(x) - round(float(x))) < 1e-6 else float(x) for x in row] for row in V55]
            W55_list = [[round(float(x)) if abs(float(x) - round(float(x))) < 1e-6 else float(x) for x in row] for row in W55]
            
            # Verify the rounded version
            if _verify(U55_list, V55_list, W55_list):
                best_result = {"U": U55_list, "V": V55_list, "W": W55_list}
            else:
                # Try without rounding
                U55_f = [[float(x) for x in row] for row in U55]
                V55_f = [[float(x) for x in row] for row in V55]
                W55_f = [[float(x) for x in row] for row in W55]
                if _verify(U55_f, V55_f, W55_f):
                    best_result = {"U": U55_f, "V": V55_f, "W": W55_f}
    
    # Try rank 54 if time permits
    if time.time() < deadline - 10:
        U54_init = [row[:-2] for row in U56]
        V54_init = [row[:-2] for row in V56]
        W54_init = [row[:-2] for row in W56]
        
        U54, V54, W54 = _try_rank(54, U54_init, V54_init, W54_init, max_iter=50)
        if U54 is not None:
            U54_list = [[round(float(x)) if abs(float(x) - round(float(x))) < 1e-6 else float(x) for x in row] for row in U54]
            V54_list = [[round(float(x)) if abs(float(x) - round(float(x))) < 1e-6 else float(x) for x in row] for row in V54]
            W54_list = [[round(float(x)) if abs(float(x) - round(float(x))) < 1e-6 else float(x) for x in row] for row in W54]
            
            if _verify(U54_list, V54_list, W54_list):
                best_result = {"U": U54_list, "V": V54_list, "W": W54_list}
            else:
                U54_f = [[float(x) for x in row] for row in U54]
                V54_f = [[float(x) for x in row] for row in V54]
                W54_f = [[float(x) for x in row] for row in W54]
                if _verify(U54_f, V54_f, W54_f):
                    best_result = {"U": U54_f, "V": V54_f, "W": W54_f}
    
    return best_result
# EVOLVE-BLOCK-END
