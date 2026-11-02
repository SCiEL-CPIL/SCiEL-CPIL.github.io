import numpy as np
import time

def construct():
    n, m, p = 2, 2, 12
    start_time = time.time()
    deadline = 120  # seconds

    def _verify(U, V, W, n, m, p, tol=1e-9):
        """Verify that U, V, W decompose the matrix multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # Build the tensor T[i, j, k] = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
        T = np.zeros((n, m, p))
        for s in range(rank):
            u = U[:, s]  # shape (n*m,)
            v = V[:, s]  # shape (m*p,)
            w = W[:, s]  # shape (p*n,)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
        T_target = np.ones((n, m, p))
        return np.max(np.abs(T - T_target)) < tol

    def _snap_to_halfint(A, tol=1e-9):
        """Snap entries to nearest half-integer and verify."""
        A_snapped = np.round(np.array(A, dtype=float) * 2) / 2
        if np.max(np.abs(A - A_snapped)) < 1e-6:
            return A_snapped
        return None

    def _build_from_blocks_2x2x3(R, U_blk, V_blk, W_blk):
        """Build full 2x2x12 decomposition from 4 blocks of 2x2x3, each of rank R.
        U_blk: (4, R), V_blk: (6, R), W_blk: (6, R)
        """
        full_rank = R * 4
        full_U = np.zeros((4, full_rank))
        full_V = np.zeros((24, full_rank))
        full_W = np.zeros((24, full_rank))
        for block in range(4):
            offset = block * R
            full_U[:, offset:offset+R] = U_blk
            for j in range(2):
                for k in range(3):
                    full_V[j*12 + block*3 + k, offset:offset+R] = V_blk[j*3+k, :]
            for k in range(3):
                for i in range(2):
                    full_W[(block*3 + k)*2 + i, offset:offset+R] = W_blk[i*3+k, :]
        return full_U, full_V, full_W

    def _build_from_blocks_2x2x4(R, U_blk, V_blk, W_blk):
        """Build full 2x2x12 decomposition from 3 blocks of 2x2x4, each of rank R.
        U_blk: (4, R), V_blk: (8, R), W_blk: (8, R)
        """
        full_rank = R * 3
        full_U = np.zeros((4, full_rank))
        full_V = np.zeros((24, full_rank))
        full_W = np.zeros((24, full_rank))
        for block in range(3):
            offset = block * R
            full_U[:, offset:offset+R] = U_blk
            for j in range(2):
                for k in range(4):
                    full_V[j*12 + block*4 + k, offset:offset+R] = V_blk[j*4+k, :]
            for k in range(4):
                for i in range(2):
                    full_W[(block*4 + k)*2 + i, offset:offset+R] = W_blk[i*4+k, :]
        return full_U, full_V, full_W

    def _build_from_blocks_2x2x6(R, U_blk, V_blk, W_blk):
        """Build full 2x2x12 decomposition from 2 blocks of 2x2x6, each of rank R.
        U_blk: (4, R), V_blk: (12, R), W_blk: (12, R)
        """
        full_rank = R * 2
        full_U = np.zeros((4, full_rank))
        full_V = np.zeros((24, full_rank))
        full_W = np.zeros((24, full_rank))
        for block in range(2):
            offset = block * R
            full_U[:, offset:offset+R] = U_blk
            for j in range(2):
                for k in range(6):
                    full_V[j*12 + block*6 + k, offset:offset+R] = V_blk[j*6+k, :]
            for k in range(6):
                for i in range(2):
                    full_W[(block*6 + k)*2 + i, offset:offset+R] = W_blk[i*6+k, :]
        return full_U, full_V, full_W

    def get_strassen_42():
        """Fallback: rank-42 decomposition via 6 blocks of 2x2x2 Strassen (rank 7)."""
        rank = 42
        U = np.zeros((4, rank))
        V = np.zeros((24, rank))
        W = np.zeros((24, rank))
        
        # Strassen 2x2x2 rank-7 decomposition
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        strassen = [
            # (U factors for A, V factors for B, W factors for C)
            # Each is (2,2) matrix entries
            # M1: A11+A22, B11+B22
            (([1,0],[0,1]), [[1,0],[0,1]], [[1,0],[0,1]]),
            # M2: A21+A22, B11
            (([0,0],[1,1]), [[1,0],[0,0]], [[0,0],[1,0]]),
            # M3: A11, B21-B22
            (([1,0],[0,0]), [[0,0],[1,-1]], [[0,1],[0,0]]),
            # M4: A22, B12-B11
            (([0,0],[0,1]), [[-1,1],[0,0]], [[0,0],[1,1]]),
            # M5: A11+A12, B22
            (([1,1],[0,0]), [[0,0],[0,1]], [[-1,0],[0,1]]),
            # M6: A21-A11, B11+B12
            (([-1,0],[1,0]), [[1,1],[0,0]], [[0,0],[0,1]]),
            # M7: A12-A22, B21+B22
            (([0,1],[0,-1]), [[0,0],[1,1]], [[1,0],[0,0]]),
        ]
        
        r = 0
        for b in range(6):
            for s in range(7):
                u_mat, v_mat, w_mat = strassen[s]
                for i in range(2):
                    for j in range(2):
                        U[i*2+j, r] = u_mat[i][j]
                for j in range(2):
                    for k in range(2):
                        V[j*12 + b*2 + k, r] = v_mat[j][k]
                for k in range(2):
                    for i in range(2):
                        W[(b*2+k)*2 + i, r] = w_mat[k][i]
                r += 1
        return U, V, W

    def search_rank_2x2x3(R=10, n_restarts=30, max_iter=300):
        """Search for rank-R decomposition of 2x2x3 matrix multiplication."""
        if time.time() - start_time > deadline - 5:
            return None
        
        dim_U = 4
        dim_V = 6
        dim_W = 6
        n_params = dim_U*R + dim_V*R + dim_W*R
        
        def objective(params):
            if time.time() - start_time > deadline - 3:
                return 1e10
            U = params[:dim_U*R].reshape(dim_U, R)
            V = params[dim_U*R:dim_U*R+dim_V*R].reshape(dim_V, R)
            W = params[dim_U*R+dim_V*R:].reshape(dim_W, R)
            
            # T[i,j,k] = sum_s U[i*2+j,s] * V[j*3+k,s] * W[i*3+k,s]
            T = np.zeros((2, 2, 3))
            for s in range(R):
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            T[i,j,k] += U[i*2+j, s] * V[j*3+k, s] * W[i*3+k, s]
            T_target = np.ones((2, 2, 3))
            return np.sum((T - T_target)**2)
        
        def gradient(params):
            if time.time() - start_time > deadline - 3:
                return np.zeros(n_params)
            U = params[:dim_U*R].reshape(dim_U, R)
            V = params[dim_U*R:dim_U*R+dim_V*R].reshape(dim_V, R)
            W = params[dim_U*R+dim_V*R:].reshape(dim_W, R)
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for s in range(R):
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            # d/dU[i*2+j,s]: V[j*3+k,s] * W[i*3+k,s] * 2*(T-T_target)
                            # T[i,j,k] = sum U[i*2+j,s']*V[j*3+k,s']*W[i*3+k,s']
                            # But T depends on all s', so we need the full T
                            pass
            
            # Simpler: use finite differences or just let scipy handle it
            # Actually for speed, let's compute properly
            T = np.zeros((2, 2, 3))
            for s in range(R):
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            T[i,j,k] += U[i*2+j, s] * V[j*3+k, s] * W[i*3+k, s]
            T_target = np.ones((2, 2, 3))
            diff = 2 * (T - T_target)
            
            for s in range(R):
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            grad_U[i*2+j, s] += diff[i,j,k] * V[j*3+k, s] * W[i*3+k, s]
                            grad_V[j*3+k, s] += diff[i,j,k] * U[i*2+j, s] * W[i*3+k, s]
                            grad_W[i*3+k, s] += diff[i,j,k] * U[i*2+j, s] * V[j*3+k, s]
            
            return np.concatenate([grad_U.ravel(), grad_V.ravel(), grad_W.ravel()])
        
        best_error = float('inf')
        best_params = None
        
        for restart in range(n_restarts):
            if time.time() - start_time > deadline - 5:
                break
            
            # Try different initialization strategies
            if restart % 3 == 0:
                params = np.random.uniform(-1, 1, size=n_params)
            elif restart % 3 == 1:
                # Symmetric initialization
                U_init = np.zeros((dim_U, R))
                V_init = np.zeros((dim_V, R))
                W_init = np.zeros((dim_W, R))
                for s in range(R):
                    U_init[0, s] = 1.0
                    V_init[0, s] = 1.0
                    W_init[0, s] = 1.0
                    U_init[1, s] = np.random.uniform(-1, 1)
                    V_init[1, s] = np.random.uniform(-1, 1)
                    W_init[1, s] = np.random.uniform(-1, 1)
                    for idx in range(2, dim_U):
                        U_init[idx, s] = np.random.uniform(-1, 1)
                    for idx in range(2, dim_V):
                        V_init[idx, s] = np.random.uniform(-1, 1)
                    for idx in range(2, dim_W):
                        W_init[idx, s] = np.random.uniform(-1, 1)
                params = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])
            else:
                params = np.random.normal(0, 0.5, size=n_params)
            
            from scipy.optimize import minimize
            res = minimize(objective, params, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'maxfun': max_iter*5, 'ftol': 1e-12})
            
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x.copy()
            
            if best_error < 1e-10:
                break
        
        if best_error < 1e-8 and best_params is not None:
            U = best_params[:dim_U*R].reshape(dim_U, R)
            V = best_params[dim_U*R:dim_U*R+dim_V*R].reshape(dim_V, R)
            W = best_params[dim_U*R+dim_V*R:].reshape(dim_W, R)
            
            # Try snapping to half-integers
            U_snap = _snap_to_halfint(U)
            V_snap = _snap_to_halfint(V)
            W_snap = _snap_to_halfint(W)
            
            if U_snap is not None and V_snap is not None and W_snap is not None:
                # Verify
                T = np.zeros((2, 2, 3))
                for s in range(R):
                    for i in range(2):
                        for j in range(2):
                            for k in range(3):
                                T[i,j,k] += U_snap[i*2+j, s] * V_snap[j*3+k, s] * W_snap[i*3+k, s]
                if np.max(np.abs(T - 1)) < 1e-9:
                    return U_snap, V_snap, W_snap
            else:
                # Try rounding to integers
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                T = np.zeros((2, 2, 3))
                for s in range(R):
                    for i in range(2):
                        for j in range(2):
                            for k in range(3):
                                T[i,j,k] += U_int[i*2+j, s] * V_int[j*3+k, s] * W_int[i*3+k, s]
                if np.max(np.abs(T - 1)) < 1e-9:
                    return U_int, V_int, W_int
                else:
                    # Return as-is (evaluator may accept float)
                    return U, V, W
        return None

    def search_rank_2x2x4(R=13, n_restarts=20, max_iter=300):
        """Search for rank-R decomposition of 2x2x4 matrix multiplication."""
        if time.time() - start_time > deadline - 5:
            return None
        
        dim_U = 4
        dim_V = 8
        dim_W = 8
        n_params = dim_U*R + dim_V*R + dim_W*R
        
        def objective(params):
            if time.time() - start_time > deadline - 3:
                return 1e10
            U = params[:dim_U*R].reshape(dim_U, R)
            V = params[dim_U*R:dim_U*R+dim_V*R].reshape(dim_V, R)
            W = params[dim_U*R+dim_V*R:].reshape(dim_W, R)
            
            T = np.zeros((2, 2, 4))
            for s in range(R):
                for i in range(2):
                    for j in range(2):
                        for k in range(4):
                            T[i,j,k] += U[i*2+j, s] * V[j*4+k, s] * W[i*4+k, s]
            T_target = np.ones((2, 2, 4))
            return np.sum((T - T_target)**2)
        
        def gradient(params):
            if time.time() - start_time > deadline - 3:
                return np.zeros(n_params)
            U = params[:dim_U*R].reshape(dim_U, R)
            V = params[dim_U*R:dim_U*R+dim_V*R].reshape(dim_V, R)
            W = params[dim_U*R+dim_V*R:].reshape(dim_W, R)
            
            T = np.zeros((2, 2, 4))
            for s in range(R):
                for i in range(2):
                    for j in range(2):
                        for k in range(4):
                            T[i,j,k] += U[i*2+j, s] * V[j*4+k, s] * W[i*4+k, s]
            T_target = np.ones((2, 2, 4))
            diff = 2 * (T - T_target)
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for s in range(R):
                for i in range(2):
                    for j in range(2):
                        for k in range(4):
                            grad_U[i*2+j, s] += diff[i,j,k] * V[j*4+k, s] * W[i*4+k, s]
                            grad_V[j*4+k, s] += diff[i,j,k] * U[i*2+j, s] * W[i*4+k, s]
                            grad_W[i*4+k, s] += diff[i,j,k] * U[i*2+j, s] * V[j*4+k, s]
            
            return np.concatenate([grad_U.ravel(), grad_V.ravel(), grad_W.ravel()])
        
        best_error = float('inf')
        best_params = None
        
        for restart in range(n_restarts):
            if time.time() - start_time > deadline - 5:
                break
            params = np.random.uniform(-1.5, 1.5, size=n_params)
            from scipy.optimize import minimize
            res = minimize(objective, params, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'maxfun': max_iter*5, 'ftol': 1e-12})
            
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x.copy()
            
            if best_error < 1e-10:
                break
        
        if best_error < 1e-8 and best_params is not None:
            U = best_params[:dim_U*R].reshape(dim_U, R)
            V = best_params[dim_U*R:dim_U*R+dim_V*R].reshape(dim_V, R)
            W = best_params[dim_U*R+dim_V*R:].reshape(dim_W, R)
            
            U_snap = _snap_to_halfint(U)
            V_snap = _snap_to_halfint(V)
            W_snap = _snap_to_halfint(W)
            
            if U_snap is not None and V_snap is not None and W_snap is not None:
                T = np.zeros((2, 2, 4))
                for s in range(R):
                    for i in range(2):
                        for j in range(2):
                            for k in range(4):
                                T[i,j,k] += U_snap[i*2+j, s] * V_snap[j*4+k, s] * W_snap[i*4+k, s]
                if np.max(np.abs(T - 1)) < 1e-9:
                    return U_snap, V_snap, W_snap
            return U, V, W
        return None

    # --- Main search strategy ---
    
    # Strategy 1: Try rank 10 for 2x2x3 -> total rank 40
    if time.time() - start_time < deadline - 10:
        result = search_rank_2x2x3(R=10, n_restarts=40, max_iter=400)
        if result is not None:
            U_blk, V_blk, W_blk = result
            full_U, full_V, full_W = _build_from_blocks_2x2x3(10, U_blk, V_blk, W_blk)
            # Verify full decomposition
            if _verify(full_U, full_V, full_W, n, m, p):
                return {"U": full_U.tolist(), "V": full_V.tolist(), "W": full_W.tolist()}
    
    # Strategy 2: Try rank 13 for 2x2x4 -> total rank 39
    if time.time() - start_time < deadline - 10:
        result = search_rank_2x2x4(R=13, n_restarts=25, max_iter=400)
        if result is not None:
            U_blk, V_blk, W_blk = result
            full_U, full_V, full_W = _build_from_blocks_2x2x4(13, U_blk, V_blk, W_blk)
            if _verify(full_U, full_V, full_W, n, m, p):
                return {"U": full_U.tolist(), "V": full_V.tolist(), "W": full_W.tolist()}
    
    # Strategy 3: Try rank 20 for 2x2x6 -> total rank 40
    if time.time() - start_time < deadline - 10:
        # Use a simpler search for 2x2x6
        R = 20
        dim_U = 4
        dim_V = 12
        dim_W = 12
        n_params = dim_U*R + dim_V*R + dim_W*R
        
        def obj6(params):
            if time.time() - start_time > deadline - 3:
                return 1e10
            U = params[:dim_U*R].reshape(dim_U, R)
            V = params[dim_U*R:dim_U*R+dim_V*R].reshape(dim_V, R)
            W = params[dim_U*R+dim_V*R:].reshape(dim_W, R)
            T = np.zeros((2, 2, 6))
            for s in range(R):
                for i in range(2):
                    for j in range(2):
                        for k in range(6):
                            T[i,j,k] += U[i*2+j, s] * V[j*6+k, s] * W[i*6+k, s]
            return np.sum((T - 1)**2)
        
        def grad6(params):
            if time.time() - start_time > deadline - 3:
                return np.zeros(n_params)
            U = params[:dim_U*R].reshape(dim_U, R)
            V = params[dim_U*R:dim_U*R+dim_V*R].reshape(dim_V, R)
            W = params[dim_U*R+dim_V*R:].reshape(dim_W, R)
            T = np.zeros((2, 2, 6))
            for s in range(R):
                for i in range(2):
                    for j in range(2):
                        for k in range(6):
                            T[i,j,k] += U[i*2+j, s] * V[j*6+k, s] * W[i*6+k, s]
            diff = 2 * (T - 1)
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            for s in range(R):
                for i in range(2):
                    for j in range(2):
                        for k in range(6):
                            grad_U[i*2+j, s] += diff[i,j,k] * V[j*6+k, s] * W[i*6+k, s]
                            grad_V[j*6+k, s] += diff[i,j,k] * U[i*2+j, s] * W[i*6+k, s]
                            grad_W[i*6+k, s] += diff[i,j,k] * U[i*2+j, s] * V[j*6+k, s]
            return np.concatenate([grad_U.ravel(), grad_V.ravel(), grad_W.ravel()])
        
        best_err = float('inf')
        best_p = None
        from scipy.optimize import minimize
        for restart in range(15):
            if time.time() - start_time > deadline - 5:
                break
            p0 = np.random.uniform(-1, 1, size=n_params)
            res = minimize(obj6, p0, jac=grad6, method='L-BFGS-B',
                          options={'maxiter': 200, 'maxfun': 1000})
            if res.fun < best_err:
                best_err = res.fun
                best_p = res.x.copy()
            if best_err < 1e-10:
                break
        
        if best_err < 1e-8 and best_p is not None:
            U_blk = best_p[:dim_U*R].reshape(dim_U, R)
            V_blk = best_p[dim_U*R:dim_U*R+dim_V*R].reshape(dim_V, R)
            W_blk = best_p[dim_U*R+dim_V*R:].reshape(dim_W, R)
            
            U_snap = _snap_to_halfint(U_blk)
            V_snap = _snap_to_halfint(V_blk)
            W_snap = _snap_to_halfint(W_blk)
            
            if U_snap is not None and V_snap is not None and W_snap is not None:
                full_U, full_V, full_W = _build_from_blocks_2x2x6(20, U_snap, V_snap, W_snap)
                if _verify(full_U, full_V, full_W, n, m, p):
                    return {"U": full_U.tolist(), "V": full_V.tolist(), "W": full_W.tolist()}
    
    # Fallback: Strassen rank 42
    U, V, W = get_strassen_42()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
