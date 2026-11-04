"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # Start with schoolbook as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank
    
    # Efficient verification using numpy
    def verify_numpy(U_mat, V_mat, W_mat, n, m, p):
        """Efficient verification using numpy."""
        rank = U_mat.shape[1]
        errors = 0
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U_mat[i*m+j, :]
                for k in range(p):
                    v_row = V_mat[j*p+k, :]
                    w_row = W_mat[k*n+i, :]
                    val = np.dot(u_row, v_row * w_row)
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
                    if err > 1e-8:
                        errors += 1
        return errors == 0, max_err
    
    def clean_and_round(mat_list):
        """Clean up near-zero and round to exact values."""
        result = []
        for row in mat_list:
            new_row = []
            for x in row:
                x = float(x)
                if abs(x) < 1e-10:
                    new_row.append(0.0)
                elif abs(x - round(x)) < 1e-10:
                    new_row.append(float(round(x)))
                elif abs(x - round(x*2)/2) < 1e-10:
                    new_row.append(float(round(x*2)/2))
                elif abs(x - round(x*4)/4) < 1e-10:
                    new_row.append(float(round(x*4)/4))
                else:
                    new_row.append(x)
            result.append(new_row)
        return result
    
    def try_als(U_init, V_init, W_init, n, m, p, max_iters=200, tol=1e-10):
        """Alternating least squares for tensor decomposition.
        
        We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        For fixed V, W: solve for U (each row of U independently)
        For fixed U, W: solve for V
        For fixed U, V: solve for W
        """
        rank = U_init.shape[1]
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                return None, float('inf')
            
            # Solve for U: for each (i,j) row, find u such that
            # sum_r u[r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
            # This is: u @ diag(V_jk * W_ki) = ones(p) for each k... 
            # Actually: for fixed (i,j), we need: sum_r u[r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            # Let M[i,j,k,r] = V[j*p+k, r] * W[k*n+i, r]
            # Then: for each (i,j), solve: sum_k... no, for each k independently
            # u @ (V[j*p+k,:] * W[k*n+i,:]) = 1 for each k
            # This is a system: for each k, one equation in u
            # We have p equations (one per k) and rank unknowns
            # If rank >= p, this is underdetermined; use least squares
            
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # Build system: A @ u = b where A[k, r] = V[j*p+k, r] * W[k*n+i, r], b[k] = 1
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    # Solve least squares
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U_new[i*m+j, :] = u_sol
                    except:
                        U_new[i*m+j, :] = U[i*m+j, :]
            
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    # For fixed (j,k): sum_r v[r] * U[i*m+j, r] * W[k*n+i, r] = 1 for all i
                    # A[i, r] = U[i*m+j, r] * W[k*n+i, r], b[i] = 1
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V_new[j*p+k, :] = v_sol
                    except:
                        V_new[j*p+k, :] = V[j*p+k, :]
            
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    # For fixed (k,i): sum_r w[r] * U[i*m+j, r] * V[j*p+k, r] = 1 for all j
                    # A[j, r] = U[i*m+j, r] * V[j*p+k, r], b[j] = 1
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W_new[k*n+i, :] = w_sol
                    except:
                        W_new[k*n+i, :] = W[k*n+i, :]
            
            # Check convergence
            U_diff = np.max(np.abs(U_new - U))
            V_diff = np.max(np.abs(V_new - V))
            W_diff = np.max(np.abs(W_new - W))
            
            U, V, W = U_new, V_new, W_new
            
            if max(U_diff, V_diff, W_diff) < tol:
                break
        
        # Final verification
        valid, max_err = verify_numpy(U, V, W, n, m, p)
        if valid:
            return (U, V, W), 0.0
        return None, max_err
    
    # Try ranks from lower bound up
    lower_bound = max(n*p, n*m, m*p)  # 30
    
    # Precompute index mappings for efficiency
    # For each (i,j,k), we need: U row i*m+j, V row j*p+k, W row k*n+i
    
    for target_rank in range(lower_bound, min(best_rank, lower_bound + 30)):
        if time.time() > deadline:
            break
        
        n_tries = 10
        found = False
        
        for trial in range(n_tries):
            if time.time() > deadline:
                break
            
            np.random.seed(12345 + trial * 7919 + target_rank * 104729)
            
            # Random initialization with varied scales
            scale = np.random.uniform(0.5, 2.0)
            U_init = np.random.randn(n*m, target_rank) * scale
            V_init = np.random.randn(m*p, target_rank) * scale
            W_init = np.random.randn(p*n, target_rank) * scale
            
            result, err = try_als(U_init, V_init, W_init, n, m, p)
            if result is not None:
                U_sol, V_sol, W_sol = result
                U_list = clean_and_round(U_sol)
                V_list = clean_and_round(V_sol)
                W_list = clean_and_round(W_sol)
                
                # Final verification of cleaned values
                U_clean = np.array(U_list)
                V_clean = np.array(V_list)
                W_clean = np.array(W_list)
                valid, _ = verify_numpy(U_clean, V_clean, W_clean, n, m, p)
                if valid:
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END

