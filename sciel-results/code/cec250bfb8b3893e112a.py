import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    
    def to_half_int(arr):
        """Round to nearest half-integer."""
        return np.round(arr * 2) / 2.0
    
    def to_int(arr):
        """Round to nearest integer."""
        return np.round(arr)
    
    def to_small_rational(arr, max_denom=4):
        """Round to nearest rational with small denominator."""
        result = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            val = arr[idx]
            best_num, best_den = 0, 1
            best_err = abs(val)
            for den in range(1, max_denom + 1):
                num = round(val * den)
                err = abs(val - num / den)
                if err < best_err:
                    best_err = err
                    best_num, best_den = num, den
            result[idx] = best_num / best_den
        return result
    
    def check_exact(U, V, W, T, tol=1e-9):
        """Check if sum_c U[:,c] \otimes V[:,c] \otimes W[:,c] == T"""
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.allclose(recon, T, atol=tol)
    
    def try_rank(r, max_iters=200, seed=42, init_mode='random'):
        """Try to find a rank-r decomposition using ALS with various inits."""
        np.random.seed(seed)
        
        if init_mode == 'random':
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
        elif init_mode == 'structured':
            # Initialize using patterns based on the tensor structure
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            # Spread the schoolbook factors across r columns
            for c in range(r):
                # Each column gets a subset of the schoolbook pattern
                i = (c // (m * p)) % n
                j = (c // p) % m
                k = c % p
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
            # Add some randomness to break symmetry
            U += 0.1 * np.random.randn(n * m, r)
            V += 0.1 * np.random.randn(m * p, r)
            W += 0.1 * np.random.randn(p * n, r)
        else:
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U
            VW = np.einsum('bd,cd->bcd', V, W)
            VW_2d = VW.transpose(2, 0, 1).reshape(r, m * p * p * n)
            T_2d = T.reshape(n * m, m * p * p * n)
            try:
                U_new = T_2d @ np.linalg.pinv(VW_2d.T)
                # Damping to avoid divergence
                U = 0.5 * U + 0.5 * U_new
            except:
                pass
            
            # Update V
            UW = np.einsum('ad,cd->acd', U, W)
            UW_2d = UW.transpose(2, 0, 1).reshape(r, n * m * p * n)
            T_2d_v = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
            try:
                V_new = T_2d_v @ np.linalg.pinv(UW_2d.T)
                V = 0.5 * V + 0.5 * V_new
            except:
                pass
            
            # Update W
            UV = np.einsum('ad,bd->abd', U, V)
            UV_2d = UV.transpose(2, 0, 1).reshape(r, n * m * m * p)
            T_2d_w = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
            try:
                W_new = T_2d_w @ np.linalg.pinv(UV_2d.T)
                W = 0.5 * W + 0.5 * W_new
            except:
                pass
            
            # Check error
            recon = np.einsum('ad,bd,cd->abc', U, V, W)
            err = np.linalg.norm(recon - T)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-8:
                break
        
        # Try multiple rounding strategies
        for round_fn in [to_half_int, to_int, to_small_rational]:
            U_h = round_fn(best_U)
            V_h = round_fn(best_V)
            W_h = round_fn(best_W)
            
            if check_exact(U_h, V_h, W_h, T):
                return U_h, V_h, W_h
            
            # Try snapping: for each element, try nearby half-integers
            # This is expensive, so only do it if we're close
            if best_err < 1.0:
                U_snap = best_U.copy()
                V_snap = best_V.copy()
                W_snap = best_W.copy()
                # Snap each matrix independently to nearby half-integers
                for mat, name in [(U_snap, 'U'), (V_snap, 'V'), (W_snap, 'W')]:
                    for idx in np.ndindex(mat.shape):
                        val = mat[idx]
                        candidates = [round(val * 2) / 2, round(val * 2) / 2 + 0.5, round(val * 2) / 2 - 0.5]
                        best_val = val
                        best_local_err = np.inf
                        for cv in candidates:
                            mat[idx] = cv
                            recon_test = np.einsum('ad,bd,cd->abc', U_snap, V_snap, W_snap)
                            local_err = np.sum((recon_test - T)**2)
                            if local_err < best_local_err:
                                best_local_err = local_err
                                best_val = cv
                        mat[idx] = best_val
                
                if check_exact(U_snap, V_snap, W_snap, T):
                    return U_snap, V_snap, W_snap
        
        return None
    
    best_result = None
    
    # Try ranks from smaller to larger, with more seeds for smaller ranks
    # The theoretical lower bound for 3x4 x 4x5 is at least max(n*m, m*p, p*n) = 20
    # Known results suggest rank around 50-60 for this size
    for r in range(40, 60):
        if time.time() > deadline - 15:
            break
        
        found = None
        # More seeds for smaller ranks
        n_seeds = 10 if r <= 45 else 5
        for seed in range(n_seeds):
            if time.time() > deadline - 15:
                break
            for init_mode in ['random', 'structured']:
                if time.time() > deadline - 15:
                    break
                result = try_rank(r, max_iters=80, seed=seed * 7 + r, init_mode=init_mode)
                if result is not None:
                    found = result
                    break
            if found is not None:
                break
        
        if found is not None:
            U, V, W = found
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    U_list = [[float(x) for x in row] for row in U_sb]
    V_list = [[float(x) for x in row] for row in V_sb]
    W_list = [[float(x) for x in row] for row in W_sb]
    return {"U": U_list, "V": V_list, "W": W_list}
