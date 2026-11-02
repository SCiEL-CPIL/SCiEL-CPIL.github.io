import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin
    
    # Build the multiplication tensor T[a, b, c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T."""
        # U, V, W are lists of lists, convert to np.array for einsum
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        T_recon = np.einsum('ad,bd,cd->abc', U_arr, V_arr, W_arr)
        return np.allclose(T_recon, T, atol=tol)
    
    def try_als(r, seed, max_iters=500):
        """Alternating least squares for rank r decomposition."""
        np.random.seed(seed)
        # Initialize with random values
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Update U
            # Minimize || sum_d U[:,d] V[b,d] W[c,d] - T ||
            # For fixed V, W, U is solved by linear system:
            # (V^T V) U^T = ... ? No.
            # T[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
            # Fix V, W. For each a, U[a,:] is unknown.
            # sum_d U[a,d] (V[:,d] W[:,d])^T = T[a,:,:].ravel()
            # Let M_U[b,c,d] = V[b,d] W[c,d]
            # Then for each a, sum_d U[a,d] M_U[:,:,d].ravel() = T[a,:,:].ravel()
            # This is a linear system U[a,:] M_U_flat = T[a,:].ravel()
            # M_U_flat has shape (r, m*p*p*n)
            
            # Efficient way using einsum:
            # h_U[a,d] = sum_{b,c} T[a,b,c] V[b,d] W[c,d]
            # G_U[d,e] = sum_{b,c} V[b,d] W[c,d] V[b,e] W[c,e]
            
            TW = V[:, np.newaxis, :] * W[np.newaxis, :, :]  # (m*p, p*n, r)
            h_U = np.einsum('abc,bcd->ad', T, TW)  # (n*m, r)
            G_U = np.einsum('bcd,bce->de', TW, TW)  # (r, r)
            
            try:
                # Solve G_U U^T = h_U^T
                U_new = np.linalg.lstsq(G_U, h_U.T, rcond=None)[0].T
            except np.linalg.LinAlgError:
                return None
            
            # Update V
            UW = U[:, np.newaxis, :] * W[np.newaxis, :, :]  # (n*m, p*n, r)
            h_V = np.einsum('abc,acd->bd', T, UW)  # (m*p, r)
            G_V = np.einsum('acd,ace->de', UW, UW)  # (r, r)
            
            try:
                V_new = np.linalg.lstsq(G_V, h_V.T, rcond=None)[0].T
            except np.linalg.LinAlgError:
                return None
            
            # Update W
            UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]  # (n*m, m*p, r)
            h_W = np.einsum('abc,abd->cd', T, UV)  # (p*n, r)
            G_W = np.einsum('abd,abe->de', UV, UV)  # (r, r)
            
            try:
                W_new = np.linalg.lstsq(G_W, h_W.T, rcond=None)[0].T
            except np.linalg.LinAlgError:
                return None
            
            U, V, W = U_new, V_new, W_new
            
            # Calculate error
            T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
            err = np.linalg.norm(T_recon - T)
            
            if err < 1e-10:
                break
                
            if err > prev_err * 1.1:
                # Diverging, break
                break
            prev_err = err
        
        # Try rounding to half-integers
        for scale in [2, 1]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            
            if verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                return (U_r, V_r, W_r)
        
        return None

    # Try ranks from 20 up to schoolbook
    # Lower bound is 15, but 20 is a reasonable start for 3x3x5
    ranks_to_try = list(range(20, rank_sb + 1))
    
    seeds = [42, 0, 1, 2, 3, 7, 13, 99, 123, 456, 789, 1000, 1001, 1002, 1003, 1004, 1005]
    
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
        if r >= best_rank:
            break
        
        for seed in seeds:
            if time.time() > deadline - 5:
                break
            
            result = try_als(r, seed)
            if result is not None:
                U_f, V_f, W_f = result
                if r < best_rank:
                    best_rank = r
                    U_list = [[float(x) for x in row] for row in U_f]
                    V_list = [[float(x) for x in row] for row in V_f]
                    W_list = [[float(x) for x in row] for row in W_f]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                break  # Found valid for this rank, move to next rank
    
    return best_result
