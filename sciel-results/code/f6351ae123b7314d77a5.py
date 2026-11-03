import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def build_rank55():
        rank = 55
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        # Part A: j=0 (2 x 10 = 20 terms)
        for i in range(n):
            for k in range(p):
                U[i * m + 0, col] = 1
                V[0 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
        
        # Part B: j=1,2 with Strassen on 5 blocks of 2x2
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2
            
            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1
            
            # M1 = (a11 + a22)(b11 + b22)
            U[u_11, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1; W[w_11, col] += 1
            col += 1
            
            # M2 = (a21 + a22)b11
            U[u_21, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1; W[w_11, col] += -1
            col += 1
            
            # M3 = a11(b12 - b22)
            U[u_11, col] += 1
            V[v_12, col] += 1; V[v_22, col] += -1
            W[w_01, col] += 1; W[w_11, col] += 1
            col += 1
            
            # M4 = a22(b21 - b11)
            U[u_22, col] += 1
            V[v_21, col] += 1; V[v_11, col] += -1
            W[w_00, col] += 1; W[w_10, col] += 1
            col += 1
            
            # M5 = (a11 + a12)b22
            U[u_11, col] += 1; U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1; W[w_01, col] += 1
            col += 1
            
            # M6 = (a21 - a11)(b11 + b12)
            U[u_21, col] += 1; U[u_11, col] += -1
            V[v_11, col] += 1; V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M7 = (a12 - a22)(b21 + b22)
            U[u_12, col] += 1; U[u_22, col] += -1
            V[v_21, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1
        
        return U, V, W, rank
    
    def compute_error(U, V, W, r):
        """Fast vectorized error computation"""
        err_max = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U[i*m + j, :]
                for k in range(p):
                    v_row = V[j*p + k, :]
                    w_row = W[k*n + i, :]
                    val = np.dot(u_row, v_row * w_row)
                    err_max = max(err_max, abs(val - 1.0))
        return err_max
    
    def verify(U, V, W, r, tol=1e-8):
        err = compute_error(U, V, W, r)
        return err < tol, err
    
    def als_optimize(U_init, V_init, W_init, target_rank, max_iters=100, tol=1e-8):
        """ALS with correct element-wise products for trilinear tensor decomposition"""
        U_cur, V_cur, W_cur = U_init.copy(), V_init.copy(), W_init.copy()
        best_err = float('inf')
        b_p = np.ones(p)
        b_n = np.ones(n)
        b_m = np.ones(m)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U rows: for each (i,j), solve A @ u = 1_p
            # where A[k, r] = V[j*p+k, r] * W[k*n+i, r]
            for i in range(n):
                for j in range(m):
                    idx = i * m + j
                    A = V_cur[j*p:(j+1)*p, :] * W_cur[np.arange(p)*n + i, :]
                    try:
                        u_sol = np.linalg.solve(A.T @ A + 1e-12*np.eye(target_rank), A.T @ b_p)
                    except np.linalg.LinAlgError:
                        u_sol, _, _, _ = np.linalg.lstsq(A, b_p, rcond=None)
                    U_cur[idx, :] = u_sol
            
            # Update V rows: for each (j,k), solve A @ v = 1_n
            # where A[i, r] = U[i*m+j, r] * W[k*n+i, r]
            for j in range(m):
                for k in range(p):
                    idx = j * p + k
                    A = U_cur[np.arange(n)*m + j, :] * W_cur[k*n:(k+1)*n, :]
                    try:
                        v_sol = np.linalg.solve(A.T @ A + 1e-12*np.eye(target_rank), A.T @ b_n)
                    except np.linalg.LinAlgError:
                        v_sol, _, _, _ = np.linalg.lstsq(A, b_n, rcond=None)
                    V_cur[idx, :] = v_sol
            
            # Update W rows: for each (k,i), solve A @ w = 1_m
            # where A[j, r] = U[i*m+j, r] * V[j*p+k, r]
            for k in range(p):
                for i in range(n):
                    idx = k * n + i
                    A = U_cur[i*m:(i+1)*m, :] * V_cur[k*p:(k+1)*p, :]
                    try:
                        w_sol = np.linalg.solve(A.T @ A + 1e-12*np.eye(target_rank), A.T @ b_m)
                    except np.linalg.LinAlgError:
                        w_sol, _, _, _ = np.linalg.lstsq(A, b_m, rcond=None)
                    W_cur[idx, :] = w_sol
            
            # Check error every few iterations for speed
            if iteration % 3 == 0 or iteration == max_iters - 1:
                err_val = compute_error(U_cur, V_cur, W_cur, target_rank)
                if err_val < best_err:
                    best_err = err_val
                    if err_val < tol:
                        break
                if err_val < 1e-10:
                    break
        
        # Final error check
        final_err = compute_error(U_cur, V_cur, W_cur, target_rank)
        best_err = min(best_err, final_err)
        
        return U_cur, V_cur, W_cur, best_err
    
    # Start with known rank-55 solution
    U_best, V_best, W_best, rank_best = build_rank55()
    valid, err = verify(U_best, V_best, W_best, rank_best)
    
    # Try to reduce rank using iterative optimization
    if time.time() < deadline - 20:
        for target_rank in range(rank_best - 1, 45, -1):
            if time.time() > deadline - 15:
                break
            
            found = False
            max_restarts = 12 if target_rank >= 50 else 6
            
            for trial in range(max_restarts):
                if time.time() > deadline - 10:
                    break
                
                np.random.seed(trial * 7919 + target_rank * 13)
                
                if trial % 3 == 0:
                    # Strategy 1: truncate + small noise
                    U_init = U_best[:, :target_rank].copy()
                    V_init = V_best[:, :target_rank].copy()
                    W_init = W_best[:, :target_rank].copy()
                    noise_scale = 0.01
                    U_init += noise_scale * np.random.randn(nm, target_rank)
                    V_init += noise_scale * np.random.randn(mp, target_rank)
                    W_init += noise_scale * np.random.randn(pn, target_rank)
                elif trial % 3 == 1:
                    # Strategy 2: random column selection + noise
                    perm = np.random.permutation(rank_best)
                    keep = perm[:target_rank]
                    U_init = U_best[:, keep].copy()
                    V_init = V_best[:, keep].copy()
                    W_init = W_best[:, keep].copy()
                    noise_scale = 0.01
                    U_init += noise_scale * np.random.randn(nm, target_rank)
                    V_init += noise_scale * np.random.randn(mp, target_rank)
                    W_init += noise_scale * np.random.randn(pn, target_rank)
                else:
                    # Strategy 3: random init with structure
                    U_init = np.random.randn(nm, target_rank) * 0.1
                    V_init = np.random.randn(mp, target_rank) * 0.1
                    W_init = np.random.randn(pn, target_rank) * 0.1
                    # Seed with some of the known solution
                    for c in range(min(target_rank, rank_best)):
                        U_init[:, c] = U_best[:, c] * 0.5 + U_init[:, c] * 0.5
                        V_init[:, c] = V_best[:, c] * 0.5 + V_init[:, c] * 0.5
                        W_init[:, c] = W_best[:, c] * 0.5 + W_init[:, c] * 0.5
                
                # Run ALS optimization
                U_cur, V_cur, W_cur, best_err = als_optimize(
                    U_init, V_init, W_init, target_rank, max_iters=100
                )
                
                if best_err < 1e-7:
                    U_best, V_best, W_best, rank_best = U_cur, V_cur, W_cur, target_rank
                    found = True
                    break
                elif best_err < 1e-3 and time.time() < deadline - 12:
                    # Polish with more iterations
                    U_cur2, V_cur2, W_cur2, best_err2 = als_optimize(
                        U_cur, V_cur, W_cur, target_rank, max_iters=80, tol=1e-9
                    )
                    if best_err2 < 1e-7:
                        U_best, V_best, W_best, rank_best = U_cur2, V_cur2, W_cur2, target_rank
                        found = True
                        break
            
            if found:
                continue  # Try next lower rank
            else:
                break  # Can't reduce further
    
    # Final verification and cleanup
    # Try to snap to half-integers for exactness
    def snap_to_halfint(mat):
        snapped = np.round(mat * 2) / 2.0
        return snapped
    
    U_check = snap_to_halfint(U_best)
    V_check = snap_to_halfint(V_best)
    W_check = snap_to_halfint(W_best)
    err_snapped = compute_error(U_check, V_check, W_check, rank_best)
    if err_snapped < 1e-8:
        U_best, V_best, W_best = U_check, V_check, W_check
    
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
