import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    n, m, p = 3, 3, 3
    nm, mp, pn = 9, 9, 9

    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i

    # Precompute target tensor entries
    ones_indices = [(i*m+j, j*p+k, k*n+i) for i in range(n) for j in range(m) for k in range(p)]
    ones_set = set(ones_indices)
    all_indices = [(a, b, c) for a in range(nm) for b in range(mp) for c in range(pn)]
    zero_indices = [idx for idx in all_indices if idx not in ones_set]

    def verify(U, V, W, tol=1e-9):
        """
        Verify if U, V, W form a valid decomposition.
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Vectorized computation: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        T = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check ones
        for a, b, c in ones_indices:
            if abs(T[a, b, c] - 1.0) > tol:
                return False
        # Check zeros
        for a, b, c in zero_indices:
            if abs(T[a, b, c]) > tol:
                return False
        return True

    def snap_to_half_integers(U, V, W):
        """Snap to nearest half-integers and verify."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r, tol=1e-9):
            return U_r, V_r, W_r
        return None

    # --- Step 1: Construct Rank-26 Solution (Strassen Core + Border) ---
    R26 = 26
    U26 = np.zeros((nm, R26))
    V26 = np.zeros((mp, R26))
    W26 = np.zeros((pn, R26))
    
    # Strassen terms for 2x2 core (indices 0,1)
    # r=0: (A00+A11)(B00+B11)
    U26[idx_u(0,0), 0] = 1; U26[idx_u(1,1), 0] = 1
    V26[idx_v(0,0), 0] = 1; V26[idx_v(1,1), 0] = 1
    W26[idx_w(0,0), 0] = 1; W26[idx_w(1,1), 0] = 1
    
    # r=1: (A10+A11)B00
    U26[idx_u(1,0), 1] = 1; U26[idx_u(1,1), 1] = 1
    V26[idx_v(0,0), 1] = 1
    W26[idx_w(0,1), 1] = 1; W26[idx_w(1,1), 1] = -1
    
    # r=2: A00(B01-B11)
    U26[idx_u(0,0), 2] = 1
    V26[idx_v(0,1), 2] = 1; V26[idx_v(1,1), 2] = -1
    W26[idx_w(1,0), 2] = 1; W26[idx_w(1,1), 2] = 1
    
    # r=3: A11(B10-B00)
    U26[idx_u(1,1), 3] = 1
    V26[idx_v(1,0), 3] = 1; V26[idx_v(0,0), 3] = -1
    W26[idx_w(0,0), 3] = 1; W26[idx_w(0,1), 3] = 1
    
    # r=4: (A00+A01)B11
    U26[idx_u(0,0), 4] = 1; U26[idx_u(0,1), 4] = 1
    V26[idx_v(1,1), 4] = 1
    W26[idx_w(0,0), 4] = -1; W26[idx_w(1,0), 4] = 1
    
    # r=5: (A10-A00)(B00+B01)
    U26[idx_u(1,0), 5] = 1; U26[idx_u(0,0), 5] = -1
    V26[idx_v(0,0), 5] = 1; V26[idx_v(0,1), 5] = 1
    W26[idx_w(1,1), 5] = 1
    
    # r=6: (A01-A11)(B10+B11)
    U26[idx_u(0,1), 6] = 1; U26[idx_u(1,1), 6] = -1
    V26[idx_v(1,0), 6] = 1; V26[idx_v(1,1), 6] = 1
    W26[idx_w(0,0), 6] = 1
    
    # Border terms (indices involving 2)
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U26[idx_u(i,j), col] = 1
                V26[idx_v(j,k), col] = 1
                W26[idx_w(k,i), col] = 1
                col += 1
                
    assert col == 26, f"Expected 26 columns, got {col}"
    
    # Verify Rank 26
    if not verify(U26, V26, W26):
        # Fallback to Schoolbook Rank 27
        R27 = 27
        U27 = np.zeros((nm, R27))
        V27 = np.zeros((mp, R27))
        W27 = np.zeros((pn, R27))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U27[idx_u(i,j), c] = 1
                    V27[idx_v(j,k), c] = 1
                    W27[idx_w(k,i), c] = 1
                    c += 1
        return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
    
    best_U, best_V, best_W = U26, V26, W26
    best_rank = 26
    
    # --- Step 2: Fast exact rank-25 check via linear algebra ---
    # Each rank-r term contributes a rank-1 tensor u_r ⊗ v_r ⊗ w_r.
    # The 27 target entries (ones) must equal sum_r u[a,r]*v[b,r]*w[c,r].
    # For a given subset S of 25 columns, we need to find coefficients
    # alpha_r such that the tensor matches. This is a system of 27 equations
    # in 25 unknowns (if we fix the column values) — overdetermined.
    # 
    # Better approach: The tensor is determined by the 27 ones-entries.
    # Each column r defines a point (u_r, v_r, w_r) in R^9 × R^9 × R^9.
    # We need sum_r u_r ⊗ v_r ⊗ w_r = target tensor.
    # 
    # For rank-25: try all C(26,25) = 26 subsets of 25 columns.
    # For each subset, check if the target tensor lies in the span of
    # {u_r ⊗ v_r ⊗ w_r : r in subset}.
    # 
    # Vectorize: for each column r, the contribution to entry (a,b,c) is
    # U[a,r]*V[b,r]*W[c,r]. So the 27 target equations are:
    # sum_{r in S} U[a_r',r]*V[b_r',r]*W[c_r',r] = 1 for each (a,b,c) in ones
    # where (a_r',b_r',c_r') is the r-th ones entry... 
    # 
    # Actually simpler: build the 27×26 matrix M where M[entry_idx, r] = 
    # U[a_entry, r] * V[b_entry, r] * W[c_entry, r] for the r-th column.
    # Then we need M[:, S] @ alpha = ones(27) for some alpha in R^25.
    # This is solvable iff the system is consistent.
    
    def check_rank_subset(U_mat, V_mat, W_mat, keep_cols, tol=1e-9):
        """Check if keeping only keep_cols gives an exact decomposition."""
        R = len(keep_cols)
        U_s = U_mat[:, keep_cols]
        V_s = V_mat[:, keep_cols]
        W_s = W_mat[:, keep_cols]
        
        # Build 27×R matrix
        M = np.zeros((len(ones_indices), R))
        for idx, (a, b, c) in enumerate(ones_indices):
            M[idx, :] = U_s[a, :] * V_s[b, :] * W_s[c, :]
        
        # Solve M @ alpha = 1 (27 equations, R unknowns)
        try:
            alpha, residuals, rank, sv = np.linalg.lstsq(M, np.ones(len(ones_indices)), rcond=1e-12)
            # Check residual
            res = M @ alpha - np.ones(len(ones_indices))
            if np.max(np.abs(res)) > tol:
                return None
        except np.linalg.LinAlgError:
            return None
        
        # Verify the full tensor (including zeros)
        U_test = U_s @ np.diag(alpha)  # No, this changes the columns
        # Actually, the decomposition is sum_r alpha_r * u_r ⊗ v_r ⊗ w_r
        # which equals U_test ⊗ V_test ⊗ W_test where U_test[:,r] = alpha_r * U_s[:,r] etc.
        U_test = U_s * alpha[np.newaxis, :]
        V_test = V_s * alpha[np.newaxis, :]
        W_test = W_s * alpha[np.newaxis, :]
        
        if verify(U_test, V_test, W_test, tol=1e-9):
            return U_test, V_test, W_test
        return None
    
    # Try all 26 subsets (drop one column)
    found_25 = False
    for drop in range(26):
        if time.time() > deadline - 30:
            break
        keep = [r for r in range(26) if r != drop]
        result = check_rank_subset(U26, V26, W26, keep)
        if result is not None:
            U_25, V_25, W_25 = result
            # Try to snap to half-integers
            snapped = snap_to_half_integers(U_25, V_25, W_25)
            if snapped:
                best_U, best_V, best_W = snapped
                best_rank = 25
                found_25 = True
                break
            else:
                # Use the floating-point solution if it's exact
                best_U, best_V, best_W = U_25, V_25, W_25
                best_rank = 25
                found_25 = True
                break
    
    # --- Step 3: Try optimization for rank 25 if not found above ---
    def try_optimize_rank(R, time_budget, init_U=None, init_V=None, init_W=None):
        if time.time() > start_time + time_budget:
            return None
        
        # Precompute indices for speed
        a_idx = []
        b_idx = []
        c_idx = []
        targets = []
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    a_idx.append(a)
                    b_idx.append(b)
                    c_idx.append(c)
                    targets.append(1.0 if (a, b, c) in ones_set else 0.0)
        a_idx = np.array(a_idx)
        b_idx = np.array(b_idx)
        c_idx = np.array(c_idx)
        targets = np.array(targets)
        
        from scipy.optimize import least_squares
        
        best_cost = float('inf')
        best_x = None
        
        max_restarts = 10
        np.random.seed(42)
        
        for i in range(max_restarts):
            if time.time() > start_time + time_budget:
                break
            
            if i == 0 and init_U is not None:
                x0 = np.concatenate([
                    init_U.flatten(),
                    init_V.flatten(),
                    init_W.flatten()
                ])
            else:
                x0 = np.concatenate([
                    np.random.randn(nm, R) * 0.3,
                    np.random.randn(mp, R) * 0.3,
                    np.random.randn(pn, R) * 0.3
                ]).flatten()
            
            def residual(x):
                U_f = x[:nm*R].reshape(nm, R)
                V_f = x[nm*R:2*nm*R].reshape(mp, R)
                W_f = x[2*nm*R:3*nm*R].reshape(pn, R)
                
                T = np.einsum('ar,br,cr->abc', U_f, V_f, W_f)
                res = T.flatten() - targets
                return res

            try:
                sol = least_squares(residual, x0, method='trf', max_nfev=2000, ftol=1e-14, xtol=1e-14, gtol=1e-14)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_x = sol.x
                    if sol.cost < 1e-12:
                        break
            except Exception:
                pass

        if best_x is not None and best_cost < 1e-10:
            U_sol = best_x[:nm*R].reshape(nm, R)
            V_sol = best_x[nm*R:2*nm*R].reshape(mp, R)
            W_sol = best_x[2*nm*R:3*nm*R].reshape(pn, R)
            snapped = snap_to_half_integers(U_sol, V_sol, W_sol)
            if snapped:
                return snapped
        return None

    if not found_25:
        remaining_time = deadline - time.time()
        if remaining_time > 20:
            time_budget_25 = min(15.0, remaining_time - 5.0)
            
            found = False
            for drop_col in range(26):
                if time.time() > start_time + time_budget_25:
                    break
                
                U_init = np.delete(U26, drop_col, axis=1)
                V_init = np.delete(V26, drop_col, axis=1)
                W_init = np.delete(W26, drop_col, axis=1)
                
                time_per_attempt = max(0.5, min(2.0, (start_time + time_budget_25 - time.time()) / max(1, 26 - drop_col)))
                res = try_optimize_rank(25, time_budget=time_per_attempt, init_U=U_init, init_V=V_init, init_W=W_init)
                if res is not None:
                    best_U, best_V, best_W = res
                    best_rank = 25
                    found = True
                    break
            
            if not found:
                if time.time() < start_time + time_budget_25 - 2.0:
                    res = try_optimize_rank(25, time_budget=min(5.0, start_time + time_budget_25 - time.time() - 2.0), init_U=None, init_V=None, init_W=None)
                    if res is not None:
                        best_U, best_V, best_W = res
                        best_rank = 25

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
