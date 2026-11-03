import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor T[a, b, c] = 1 if (a//m == c%n) and (a%m == b//p) and (b%p == c//n) else 0
    # Indices: a in [0, nm), b in [0, mp), c in [0, pn)
    # a = i*m + j, b = j'*p + k, c = k'*n + i'
    # Condition: j == j', k == k', i == i'
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def get_baseline_rank36():
        """
        Constructs a valid rank-36 decomposition.
        Strategy: 
        Split A (2x4) into A_left (2x2) and A_right (2x2).
        Split B (4x5) into B_top (2x5) and B_bottom (2x5).
        C = A_left * B_top + A_right * B_bottom.
        
        For A_left * B_top (2x2 * 2x5):
        Split B_top into B_top_left (2x2) and B_top_right (2x2) and B_top_col4 (2x1).
        Rank(2x2*2x2) = 7 (Strassen).
        Rank(2x2*2x1) = 4 (Schoolbook, as 2x1 is thin).
        Total for one side: 7 + 7 + 4 = 18.
        Two sides: 18 * 2 = 36.
        """
        rank = 36
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        def add_strassen_222(i0, j0, k0):
            """Add rank-7 Strassen block for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]"""
            nonlocal col
            # Strassen formulas for 2x2 matrices
            # M1 = (A11 + A22)(B11 + B22)
            # M2 = (A21 + A22)B11
            # M3 = A11(B21 - B22)
            # M4 = A22(B11 - B12)
            # M5 = (A11 + A12)B22
            # M6 = (A21 - A11)(B11 + B12)
            # M7 = (A12 - A22)(B21 + B22)
            
            # Local indices:
            # A: (0,0)->i0,j0; (0,1)->i0,j0+1; (1,0)->i0+1,j0; (1,1)->i0+1,j0+1
            # B: (0,0)->j0,k0; (0,1)->j0,k0+1; (1,0)->j0+1,k0; (1,1)->j0+1,k0+1
            # C: (0,0)->i0,k0; (0,1)->i0,k0+1; (1,0)->i0+1,k0; (1,1)->i0+1,k0+1
            
            def idx_u(i, j): return (i0 + i) * m + (j0 + j)
            def idx_v(j, k): return (j0 + j) * p + (k0 + k)
            def idx_w(k, i): return (k0 + k) * n + (i0 + i)
            
            # M1
            U[idx_u(0,0), col] += 1; U[idx_u(1,1), col] += 1
            V[idx_v(0,0), col] += 1; V[idx_v(1,1), col] += 1
            W[idx_w(0,0), col] += 1; W[idx_w(1,1), col] += 1
            col += 1
            # M2
            U[idx_u(1,0), col] += 1; U[idx_u(1,1), col] += 1
            V[idx_v(0,0), col] += 1
            W[idx_w(0,1), col] += 1; W[idx_w(1,1), col] -= 1
            col += 1
            # M3
            U[idx_u(0,0), col] += 1
            V[idx_v(1,0), col] += 1; V[idx_v(1,1), col] -= 1
            W[idx_w(1,0), col] += 1; W[idx_w(1,1), col] += 1
            col += 1
            # M4
            U[idx_u(1,1), col] += 1
            V[idx_v(0,0), col] += 1; V[idx_v(0,1), col] -= 1
            W[idx_w(0,0), col] += 1; W[idx_w(1,0), col] += 1
            col += 1
            # M5
            U[idx_u(0,0), col] += 1; U[idx_u(0,1), col] += 1
            V[idx_v(1,1), col] += 1
            W[idx_w(0,1), col] -= 1; W[idx_w(1,1), col] += 1
            col += 1
            # M6
            U[idx_u(1,0), col] += 1; U[idx_u(0,0), col] -= 1
            V[idx_v(0,0), col] += 1; V[idx_v(0,1), col] += 1
            W[idx_w(1,1), col] += 1
            col += 1
            # M7
            U[idx_u(0,1), col] += 1; U[idx_u(1,1), col] -= 1
            V[idx_v(1,0), col] += 1; V[idx_v(1,1), col] += 1
            W[idx_w(0,0), col] += 1
            col += 1

        def add_schoolbook_2x2x1(i0, j0, k0):
            """Add rank-4 schoolbook block for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+1] -> C[i0:i0+2, k0:k0+1]"""
            nonlocal col
            # C[i, k] = sum_j A[i, j] B[j, k]
            # For each i in 0..1, j in 0..1:
            # Term: A[i, j] * B[j, k] -> contributes to C[i, k]
            for i in range(2):
                for j in range(2):
                    U[(i0 + i) * m + (j0 + j), col] = 1
                    V[(j0 + j) * p + k0, col] = 1
                    W[k0 * n + (i0 + i), col] = 1
                    col += 1

        # Block 1: A[0:2, 0:2] * B[0:2, 0:5]
        # B[0:2, 0:5] split into B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:5]
        add_strassen_222(0, 0, 0) # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        add_strassen_222(0, 0, 2) # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
        add_schoolbook_2x2x1(0, 0, 4) # A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5]

        # Block 2: A[0:2, 2:4] * B[2:4, 0:5]
        # Note: A is 2x4, so rows 0:2, cols 2:4.
        # B is 4x5, so rows 2:4, cols 0:5.
        # Result C rows 0:2, cols 0:5.
        # Wait, the matrix multiplication is A (2x4) * B (4x5) = C (2x5).
        # The inner dimension is 4 (m).
        # So we sum over j=0..3.
        # C[i,k] = sum_{j=0}^3 A[i,j] B[j,k].
        # We split the sum into j=0,1 and j=2,3.
        # Term 1: A[0:2, 0:2] * B[0:2, 0:5]
        # Term 2: A[0:2, 2:4] * B[2:4, 0:5]
        
        # My previous call added A[0:2, 0:2] * B[0:2, ...]
        # Now add A[0:2, 2:4] * B[2:4, ...]
        
        add_strassen_222(0, 2, 0) # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        add_strassen_222(0, 2, 2) # A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
        add_schoolbook_2x2x1(0, 2, 4) # A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5]
        
        assert col == 36, f"Expected col 36, got {col}"
        
        return U, V, W

    def verify(U, V, W, tol=1e-9):
        # T_rec = sum_r U[:,r] V[:,r] W[:,r]
        T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.max(np.abs(T_rec - T)) < tol

    def snap_to_half_integers(U, V, W):
        return np.round(U * 2) / 2.0, np.round(V * 2) / 2.0, np.round(W * 2) / 2.0

    def residual(x, R):
        size_U = nm * R
        size_V = mp * R
        U = x[0:size_U].reshape(nm, R)
        V = x[size_U:size_U+size_V].reshape(mp, R)
        W = x[size_U+size_V:].reshape(pn, R)
        T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return (T_rec - T).flatten()

    def gradient(x, R):
        size_U = nm * R
        size_V = mp * R
        U = x[0:size_U].reshape(nm, R)
        V = x[size_U:size_U+size_V].reshape(mp, R)
        W = x[size_U+size_V:].reshape(pn, R)
        T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        res = T_rec - T
        grad_U = np.einsum('abc,br,cr->ar', res, V, W, optimize=True)
        grad_V = np.einsum('abc,ar,cr->br', res, U, W, optimize=True)
        grad_W = np.einsum('abc,ar,br->cr', res, U, V, optimize=True)
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    U36, V36, W36 = get_baseline_rank36()
    
    # Verify baseline
    if not verify(U36, V36, W36):
        # Fallback to pure schoolbook if Strassen construction failed (should not happen)
        rank_school = n * m * p
        U_school = np.zeros((nm, rank_school))
        V_school = np.zeros((mp, rank_school))
        W_school = np.zeros((pn, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j, col] = 1
                    V_school[j*p+k, col] = 1
                    W_school[k*n+i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    best_rank = 36
    best_U, best_V, best_W = U36, V36, W36

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def try_optimize(U_init, V_init, W_init, R):
        size_U = nm * R
        size_V = mp * R
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = opt.minimize(
                lambda x: 0.5 * np.sum(residual(x, R)**2),
                x0,
                method='L-BFGS-B',
                jac=lambda x: gradient(x, R),
                maxiter=2000,
                maxfun=4000,
                options={'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            x_opt = res.x
            U_opt = x_opt[0:size_U].reshape(nm, R)
            V_opt = x_opt[size_U:size_U+size_V].reshape(mp, R)
            W_opt = x_opt[size_U+size_V:].reshape(pn, R)
            
            T_rec = np.einsum('ar,br,cr->abc', U_opt, V_opt, W_opt, optimize=True)
            max_res = np.max(np.abs(T_rec - T))
            return U_opt, V_opt, W_opt, max_res
        except Exception:
            return None, None, None, np.inf

    # Try to reduce rank to 35
    # Strategy: 
    # 1. Try removing each of the 36 columns from the baseline and optimizing the rest.
    # 2. Random restarts.
    
    found_35 = False
    for R in range(35, 23, -1):
        if time.time() > deadline:
            break
            
        # If R < 35, we just do random restarts as dropping multiple columns is harder
        if R == 35:
            # Try dropping each column
            for drop_idx in range(36):
                if time.time() > deadline:
                    break
                
                U_init = np.delete(U36, drop_idx, axis=1)
                V_init = np.delete(V36, drop_idx, axis=1)
                W_init = np.delete(W36, drop_idx, axis=1)
                
                U_opt, V_opt, W_opt, max_res = try_optimize(U_init, V_init, W_init, R)
                
                if U_opt is None:
                    continue
                
                if max_res < 1e-6:
                    # Polish
                    U_snap, V_snap, W_snap = snap_to_half_integers(U_opt, V_opt, W_opt)
                    if verify(U_snap, V_snap, W_snap):
                        best_rank = R
                        best_U, best_V, best_W = U_snap, V_snap, W_snap
                        found_35 = True
                        break
        
        if found_35:
            break
            
        # Random restarts for lower ranks or if dropping didn't work
        remaining_time = deadline - time.time()
        if remaining_time < 10:
            break
            
        # Number of attempts based on time
        n_attempts = int(remaining_time / 2)
        n_attempts = min(n_attempts, 50)
        
        for attempt in range(n_attempts):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if attempt % 3 == 0:
                # Random Gaussian
                scale = 1.0
                U_init = np.random.normal(0, scale, (nm, R))
                V_init = np.random.normal(0, scale, (mp, R))
                W_init = np.random.normal(0, scale, (pn, R))
            elif attempt % 3 == 1:
                # Sparse random
                U_init = np.zeros((nm, R))
                V_init = np.zeros((mp, R))
                W_init = np.zeros((pn, R))
                for r in range(R):
                    idx_u = np.random.choice(nm, size=3, replace=False)
                    idx_v = np.random.choice(mp, size=3, replace=False)
                    idx_w = np.random.choice(pn, size=3, replace=False)
                    U_init[idx_u, r] = np.random.choice([-1, 1], size=3)
                    V_init[idx_v, r] = np.random.choice([-1, 1], size=3)
                    W_init[idx_w, r] = np.random.choice([-1, 1], size=3)
            else:
                # Truncate baseline + noise
                if R <= 36:
                    U_init = U36[:, :R].copy()
                    V_init = V36[:, :R].copy()
                    W_init = W36[:, :R].copy()
                    U_init += np.random.normal(0, 0.1, (nm, R))
                    V_init += np.random.normal(0, 0.1, (mp, R))
                    W_init += np.random.normal(0, 0.1, (pn, R))
                else:
                    U_init = np.random.normal(0, 1.0, (nm, R))
                    V_init = np.random.normal(0, 1.0, (mp, R))
                    W_init = np.random.normal(0, 1.0, (pn, R))
            
            U_opt, V_opt, W_opt, max_res = try_optimize(U_init, V_init, W_init, R)
            
            if U_opt is None:
                continue
            
            if max_res < 1e-6:
                U_snap, V_snap, W_snap = snap_to_half_integers(U_opt, V_opt, W_opt)
                if verify(U_snap, V_snap, W_snap):
                    best_rank = R
                    best_U, best_V, best_W = U_snap, V_snap, W_snap
                    break # Found valid solution for this rank, try lower

    # Final check and return
    U_snap, V_snap, W_snap = snap_to_half_integers(best_U, best_V, best_W)
    if verify(U_snap, V_snap, W_snap):
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    else:
        # Fallback to un-snapped if snapping broke it, though it shouldn't if verify passed
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
