import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 135.0  # 15s safety margin before the 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def time_left():
        return deadline - time.time()

    def make_schoolbook():
        rank = nm * mp * pn
        U = np.zeros((nm, rank), dtype=np.int64)
        V = np.zeros((mp, rank), dtype=np.int64)
        W = np.zeros((pn, rank), dtype=np.int64)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        return U, V, W, rank

    def residuals(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # We need sum_c U[i*m+j1, c] * V[j2*p+k, c] * W[k*n+i, c] == delta[j1,j2]
        # for all i,k,j1,j2.
        R = U.shape[1]
        # Precompute for speed: for each c, U_c is vector of length nm, V_c length mp, W_c length pn
        # We'll compute residuals by looping over (i,k) and using matrix products.
        # For fixed i,k, the residual for (j1,j2) is sum_c U[i*m+j1,c] * V[j2*p+k,c] * W[k*n+i,c]
        # Let a_c = U[i*m+j1,c], b_c = V[j2*p+k,c], w_c = W[k*n+i,c]
        # This is equivalent to: for each c, a_c * w_c * b_c summed over c.
        # We can form for each c: M_c[j1,j2] = a_c * b_c * w_c, but that's R matrices.
        # Better: for fixed i,k, define U_ik = U[i*m:(i+1)*m, :]  (m x R)
        #          V_ik = V[:, k]  but V is indexed by (j2*p+k), so for fixed k, 
        #          the rows of V that matter are V[j2*p+k] for j2 in 0..m-1.
        #          So V_ik = V[np.arange(m)*p + k, :]  (m x R)
        #          W_ik = W[k*n+i]  (scalar per c)
        # Then residual[j1,j2] = sum_c U_ik[j1,c] * V_ik[j2,c] * W_ik[c]
        # = sum_c (U_ik[j1,c] * W_ik[c]) * V_ik[j2,c]
        # Let A_ik = U_ik * W_ik[None,:]  (m x R)
        # Then residual = A_ik @ V_ik.T  (m x m)
        
        res = np.zeros((n, p, m, m))
        for i in range(n):
            U_ik = U[i * m:(i + 1) * m, :]  # (m, R)
            for k in range(p):
                V_ik = V[np.arange(m) * p + k, :]  # (m, R)
                w_scalar = W[k * n + i, :]  # (R,)
                A_ik = U_ik * w_scalar[None, :]  # (m, R)
                res[i, k] = A_ik @ V_ik.T  # (m, m)
        
        # Flatten residuals
        flat = np.zeros(n * p * m * m)
        idx = 0
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        target = 1.0 if j1 == j2 else 0.0
                        flat[idx] = res[i, k, j1, j2] - target
                        idx += 1
        return flat

    def is_exact(U, V, W, tol=1e-9):
        r = residuals(U, V, W)
        return np.max(np.abs(r)) < tol

    def snap_to_half_int(M):
        return np.round(M * 2) / 2.0

    def to_list_of_lists(M):
        # M is (rows, rank), convert to list of lists
        return [[float(x) for x in row] for row in M]

    # Start with schoolbook
    U, V, W, rank = make_schoolbook()
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank

    # Try to reduce rank by zeroing out columns (sparsification)
    # We'll use a randomized local search on half-integer matrices.
    # First, convert to float for optimization
    U_f = U.astype(np.float64)
    V_f = V.astype(np.float64)
    W_f = W.astype(np.float64)

    # Local search: try to reduce error by perturbing entries
    def local_search(U_f, V_f, W_f, max_iters=10000):
        current_res = residuals(U_f, V_f, W_f)
        current_err = np.dot(current_res, current_res)
        
        R = U_f.shape[1]
        for it in range(max_iters):
            if time_left() < 5.0:
                break
            
            # Randomly pick a matrix and entry
            mat_choice = np.random.randint(3)
            if mat_choice == 0:
                mat = U_f
                rows, cols = U_f.shape
            elif mat_choice == 1:
                mat = V_f
                rows, cols = V_f.shape
            else:
                mat = W_f
                rows, cols = W_f.shape
            
            i = np.random.randint(rows)
            j = np.random.randint(cols)
            old_val = mat[i, j]
            
            # Try small steps (half-integers)
            steps = [0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0]
            best_step = None
            best_err = current_err
            
            for step in steps:
                new_val = old_val + step
                # Snap to half-integer
                new_val = round(new_val * 2) / 2.0
                if new_val == old_val:
                    continue
                mat[i, j] = new_val
                new_res = residuals(U_f, V_f, W_f)
                new_err = np.dot(new_res, new_res)
                if new_err < best_err - 1e-12:
                    best_err = new_err
                    best_step = new_val
                mat[i, j] = old_val
            
            if best_step is not None:
                mat[i, j] = best_step
                current_err = best_err
                if current_err < 1e-10:
                    break
        
        return U_f, V_f, W_f, current_err

    # Try to reduce rank by zeroing columns
    def try_reduce_rank(U_f, V_f, W_f, target_rank, max_iters=5000):
        # We'll try to zero out columns one by one
        R = U_f.shape[1]
        if R <= target_rank:
            return U_f, V_f, W_f, R, True
        
        # Start with full rank
        U_cur = U_f.copy()
        V_cur = V_f.copy()
        W_cur = W_f.copy()
        
        # Try to zero out columns from the end
        for c in range(R - 1, target_rank - 1, -1):
            if time_left() < 10.0:
                break
            
            # Try to zero out column c
            U_cur[:, c] = 0.0
            V_cur[:, c] = 0.0
            W_cur[:, c] = 0.0
            
            # Now try to recover by adjusting other entries
            U_cur, V_cur, W_cur, err = local_search(U_cur, V_cur, W_cur, max_iters=max_iters)
            
            if err < 1e-10:
                # Success, continue to next column
                continue
            else:
                # Failed, restore column c
                U_cur[:, c] = U_f[:, c]
                V_cur[:, c] = V_f[:, c]
                W_cur[:, c] = W_f[:, c]
                break
        
        # Check if we achieved target rank
        R_final = U_cur.shape[1]
        # Count non-zero columns
        non_zero_cols = 0
        for c in range(R_final):
            if np.any(U_cur[:, c] != 0) or np.any(V_cur[:, c] != 0) or np.any(W_cur[:, c] != 0):
                non_zero_cols += 1
        
        if non_zero_cols <= target_rank and is_exact(U_cur, V_cur, W_cur):
            # Trim to non-zero columns
            keep = [c for c in range(R_final) if np.any(U_cur[:, c] != 0) or np.any(V_cur[:, c] != 0) or np.any(W_cur[:, c] != 0)]
            U_trim = U_cur[:, keep]
            V_trim = V_cur[:, keep]
            W_trim = W_cur[:, keep]
            return U_trim, V_trim, W_trim, len(keep), True
        
        return U_f, V_f, W_f, R, False

    # Main search loop
    # Try to reduce rank from 105 down
    for target_rank in range(104, 34, -1):
        if time_left() < 20.0:
            break
        
        U_try, V_try, W_try, rank_try, success = try_reduce_rank(U_f, V_f, W_f, target_rank, max_iters=3000)
        
        if success and rank_try < best_rank:
            best_U, best_V, best_W, best_rank = U_try.copy(), V_try.copy(), W_try.copy(), rank_try
            # Update base for further reduction
            U_f, V_f, W_f = best_U.copy(), best_V.copy(), best_W.copy()
        
        if time_left() < 15.0:
            break

    # Final verification
    if not is_exact(best_U, best_V, best_W):
        # Fall back to schoolbook
        best_U, best_V, best_W, best_rank = make_schoolbook()

    # Convert to lists
    U_list = to_list_of_lists(best_U)
    V_list = to_list_of_lists(best_V)
    W_list = to_list_of_lists(best_W)

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
