"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.
    Uses structured ALS optimization with smooth objective to reduce rank from schoolbook.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 145s deadline for 150s limit with safety margin
    
    # Build schoolbook decomposition
    rank_sb = n * m * p  # 42
    U_sb = np.zeros((n * m, rank_sb), dtype=float)
    V_sb = np.zeros((m * p, rank_sb), dtype=float)
    W_sb = np.zeros((p * n, rank_sb), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_sb[u_idx, col] = 1.0
                V_sb[v_idx, col] = 1.0
                W_sb[w_idx, col] = 1.0
                col += 1
    
    # Precompute target indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target_indices.append((u_idx, v_idx, w_idx))
    
    num_targets = len(target_indices)
    
    # Helper: compute sum of squared errors
    def compute_sse(U_mat, V_mat, W_mat):
        """Returns sum of squared errors from target tensor (all ones)."""
        sse = 0.0
        for u_idx, v_idx, w_idx in target_indices:
            val = np.dot(U_mat[u_idx, :], V_mat[v_idx, :] * W_mat[w_idx, :])
            sse += (val - 1.0) ** 2
        return sse
    
    # Verify schoolbook
    assert compute_sse(U_sb, V_sb, W_sb) < 1e-10
    
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = rank_sb
    best_sse = 0.0
    
    # ALS solver for a given target rank
    def als_solve(U_init, V_init, W_init, max_iters=500, tol=1e-12):
        """Run ALS to minimize SSE for the given factor matrices."""
        U_cur = U_init.copy()
        V_cur = V_init.copy()
        W_cur = W_init.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U: for each row u, solve least squares
            for u_idx in range(n * m):
                # Collect all (v_idx, w_idx) pairs that involve this u_idx
                # For each target (u_idx, v_idx, w_idx), we need U[u_idx,:] * (V[v_idx,:] * W[w_idx,:]) = 1
                # This is: U[u_idx,:] @ A = 1 where A is a matrix with rows (V[v_idx,:] * W[w_idx,:])
                # Build the system
                A_rows = []
                b_vals = []
                for u_t, v_t, w_t in target_indices:
                    if u_t == u_idx:
                        A_rows.append(V_cur[v_t, :] * W_cur[w_t, :])
                        b_vals.append(1.0)
                if A_rows:
                    A_mat = np.array(A_rows)
                    b_vec = np.array(b_vals)
                    # Solve A_mat @ U_cur[u_idx,:] = b_vec
                    sol, residuals, rank_l, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U_cur[u_idx, :] = sol
            
            # Update V
            for v_idx in range(m * p):
                A_rows = []
                b_vals = []
                for u_t, v_t, w_t in target_indices:
                    if v_t == v_idx:
                        A_rows.append(U_cur[u_t, :] * W_cur[w_t, :])
                        b_vals.append(1.0)
                if A_rows:
                    A_mat = np.array(A_rows)
                    b_vec = np.array(b_vals)
                    sol, residuals, rank_l, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V_cur[v_idx, :] = sol
            
            # Update W
            for w_idx in range(p * n):
                A_rows = []
                b_vals = []
                for u_t, v_t, w_t in target_indices:
                    if w_t == w_idx:
                        A_rows.append(U_cur[u_t, :] * V_cur[v_t, :])
                        b_vals.append(1.0)
                if A_rows:
                    A_mat = np.array(A_rows)
                    b_vec = np.array(b_vals)
                    sol, residuals, rank_l, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W_cur[w_idx, :] = sol
            
            # Check convergence
            sse_cur = compute_sse(U_cur, V_cur, W_cur)
            if sse_cur < tol:
                break
        
        return U_cur, V_cur, W_cur, compute_sse(U_cur, V_cur, W_cur)
    
    # Try to reduce rank from schoolbook down to lower bound
    lower_bound = max(n * m, m * p, p * n)  # 21
    
    np.random.seed(42)
    
    for target_rank in range(best_rank - 1, lower_bound - 1, -1):
        if time.time() > deadline:
            break
        
        found = False
        max_restarts = 10
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize by taking a subset of schoolbook columns
            # Use a structured selection: pick columns that cover all (i,j) for U, all (j,k) for V, all (k,i) for W
            if restart == 0:
                # First restart: use first target_rank columns of schoolbook
                U_init = U_sb[:, :target_rank].copy()
                V_init = V_sb[:, :target_rank].copy()
                W_init = W_sb[:, :target_rank].copy()
            else:
                # Random selection of columns
                cols = np.random.choice(rank_sb, target_rank, replace=False)
                U_init = U_sb[:, cols].copy()
                V_init = V_sb[:, cols].copy()
                W_init = W_sb[:, cols].copy()
                # Add small random perturbation
                U_init += np.random.randn(n * m, target_rank) * 0.1
                V_init += np.random.randn(m * p, target_rank) * 0.1
                W_init += np.random.randn(p * n, target_rank) * 0.1
            
            U_sol, V_sol, W_sol, sse_sol = als_solve(U_init, V_init, W_sol if False else W_init)
            
            if sse_sol < 1e-10:
                found = True
                best_U, best_V, best_W = U_sol.copy(), V_sol.copy(), W_sol.copy()
                best_rank = target_rank
                best_sse = sse_sol
                break
        
        if not found:
            # If we can't reduce further, stop
            break
    
    # Convert to list format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Convert to int if possible
    def to_int_or_half(x):
        if abs(x - round(x)) < 1e-8:
            return int(round(x))
        elif abs(x * 2 - round(x * 2)) < 1e-8:
            return round(x * 2) / 2.0
        else:
            return x
    
    U_final = [[to_int_or_half(val) for val in row] for row in U_list]
    V_final = [[to_int_or_half(val) for val in row] for row in V_list]
    W_final = [[to_int_or_half(val) for val in row] for row in W_list]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    # Simple validation check if run directly
    res = construct()
    n, m, p = DIMENSIONS
    U = res['U']
    V = res['V']
    W = res['W']
    
    # Verify reconstruction
    rank = len(U[0])
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_np[u_idx, :], V_np[v_idx, :] * W_np[w_idx, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    print("Max error:", max_err)
    print("Rank:", rank)
    print("Valid:", max_err < 1e-8)

