import numpy as np
import random
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school

    start_time = time.time()
    time_limit = 140  # safety margin before 150s kill

    def is_nice_matrix(matrix, tol=1e-8):
        """Check if matrix entries are close to half-integers. Returns (bool, nice_matrix)"""
        arr = np.array(matrix, dtype=float)
        doubled = arr * 2
        rounded = np.round(doubled)
        if not np.allclose(doubled, rounded, atol=tol):
            return False, None
        nice = rounded / 2.0
        result = []
        for row in nice:
            res_row = []
            for val in row:
                if val == int(val):
                    res_row.append(int(val))
                else:
                    res_row.append(val)
            result.append(res_row)
        return True, result

    def verify_decomposition(U, V, W, rank):
        """Verify decomposition using tensor contraction: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # For each (i,j,k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Vectorized: iterate over k
        for k in range(p):
            # V rows: for each j, V[j*p+k, :]
            # W rows: for each i, W[k*n+i, :]
            # U rows: for each (i,j), U[i*m+j, :]
            V_rows = V_arr[k::p, :]  # shape (m, rank) -- row j is V[j*p+k, :]
            W_rows = W_arr[k::n, :]  # shape (n, rank) -- row i is W[k*n+i, :]
            # For each (i, j), we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # = sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r])
            # For fixed k, W[k*n+i, :] depends on i, V[j*p+k, :] depends on j
            # Build a 3D tensor: T[i, j, r] = V[j, r] * W[i, r]
            T = V_rows.T[:, :, None] * W_rows.T[None, :, :]  # shape (m, n, rank)
            # Now for each (i, j): dot(U[i*m+j, :], T[j, i, :])
            # U_arr has shape (nm, rank), row i*m+j is U[i*m+j, :]
            # We need sum_r U[i*m+j, r] * T[j, i, r]
            # Reshape U to (n, m, rank)
            U_reshaped = U_arr.reshape(n, m, rank)
            # Contract over r: result[i, j] = sum_r U_reshaped[i, j, r] * T[j, i, r]
            # T[j, i, r] -> transpose to (i, j, r)
            T_reshaped = T.transpose(1, 0, 2)  # shape (n, m, rank)
            result = np.sum(U_reshaped * T_reshaped, axis=2)  # shape (n, m)
            if not np.allclose(result, 1.0, atol=1e-8):
                return False
        return True

    # Precompute index mappings for building A efficiently
    # A has shape (n*m*p, nm*R)
    # Row (i, j, k): A[row, (i*m+j)*R : (i*m+j)*R+R] = V[j*p+k, :] * W[k*n+i, :]
    
    def build_A(V_rand, W_rand, R):
        """Build the linear system matrix A for given V, W and rank R."""
        A = np.zeros((n * m * p, nm * R))
        row_idx = 0
        for i in range(n):
            for j in range(m):
                u_start = (i * m + j) * R
                for k in range(p):
                    v_row = V_rand[j * p + k]
                    w_row = W_rand[k * n + i]
                    A[row_idx, u_start:u_start + R] = v_row * w_row
                    row_idx += 1
        return A

    def try_rank(R, num_restarts, deadline):
        """Try to find a decomposition of rank R with given number of restarts."""
        np.random.seed(random.randint(0, 2**31))
        
        for r_idx in range(num_restarts):
            if time.time() > deadline:
                return None
            
            # Random V, W
            # Strategy 1: {-1, 0, 1}
            V_rand = np.random.choice([-1, 0, 1], size=(mp, R))
            W_rand = np.random.choice([-1, 0, 1], size=(pn, R))
            
            # Build A
            A = build_A(V_rand, W_rand, R)
            b = np.ones(n * m * p)
            
            # Solve least squares
            try:
                U_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            # If system is overdetermined (n*m*p > nm*R), check residual
            if n * m * p > nm * R:
                if len(residuals) > 0 and residuals[0] > 1e-6:
                    continue
            
            U_sol = U_flat.reshape((nm, R))
            
            # Check if U_sol is nice (half-integer)
            ok, U_nice = is_nice_matrix(U_sol)
            if ok:
                # Verify with the nice values
                if verify_decomposition(U_nice, V_rand.tolist(), W_rand.tolist(), R):
                    return {"U": U_nice, "V": V_rand.tolist(), "W": W_rand.tolist()}
        
        return None

    # Try ranks in ascending order
    ranks_to_try = [20, 22, 24, 26, 28, 30, 32, 35, 40, 45, 50, 55, 60]
    
    for R in ranks_to_try:
        if R >= best_rank:
            break
            
        if time.time() > deadline:
            break
        
        # More restarts for smaller ranks (harder to find)
        if R <= 24:
            num_restarts = 5000
        elif R <= 30:
            num_restarts = 3000
        else:
            num_restarts = 1500
        
        result = try_rank(R, num_restarts, deadline)
        if result is not None:
            best_rank = R
            best_U = result["U"]
            best_V = result["V"]
            best_W = result["W"]
            return {"U": best_U, "V": best_V, "W": best_W}

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
