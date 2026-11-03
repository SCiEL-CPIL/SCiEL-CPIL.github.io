"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Schoolbook decomposition
    U_sb = np.zeros((n * m, rank_school))
    V_sb = np.zeros((m * p, rank_school))
    W_sb = np.zeros((p * n, rank_school))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1

    # Target Tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1

    def is_valid(U, V, W, tol=1e-9):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for r in range(R):
                        val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    if abs(val - 1) > tol:
                        return False
        return True

    start_time = time.time()
    deadline = 140.0  # Safety margin

    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_school

    # Search for lower ranks
    for R in range(rank_school - 1, 15, -1):
        if time.time() > deadline:
            break
            
        # Try random integer/half-integer U, V and solve for W
        for attempt in range(200):
            if time.time() > deadline:
                break
                
            # Random initialization with small integers
            U = np.random.randint(-2, 3, size=(n * m, R))
            V = np.random.randint(-2, 3, size=(m * p, R))
            
            # Construct linear system for W
            # Variables: W[k*n+i, r] for k in 0..p-1, i in 0..n-1, r in 0..R-1
            # Total variables: p * n * R
            # Equations: n * m * p = 36
            # Eq (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            num_vars = p * n * R
            A_mat = np.zeros((n * m * p, num_vars))
            b_vec = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_base = k * n + i
                        for r in range(R):
                            var_idx = w_base * R + r
                            A_mat[eq_idx, var_idx] = U[u_idx, r] * V[v_idx, r]
                        eq_idx += 1
            
            # Solve A_mat @ W_flat = b_vec
            try:
                # Use least squares if underdetermined, or exact solve if square
                if num_vars >= n * m * p:
                    W_flat, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                else:
                    W_flat = np.linalg.solve(A_mat, b_vec)
                
                # Check if W_flat is close to integers
                W_int = np.round(W_flat)
                if np.allclose(A_mat @ W_int, b_vec, atol=1e-9):
                    W_sol = W_int.reshape((p * n, R))
                    if is_valid(U, V, W_sol):
                        best_U, best_V, best_W = U, V, W_sol
                        best_rank = R
                        break # Found a valid decomposition for this rank, try lower
                
                # Check if W_flat is close to half-integers
                W_half = np.round(W_flat * 2) / 2
                if np.allclose(A_mat @ W_half, b_vec, atol=1e-9):
                    W_sol = W_half.reshape((p * n, R))
                    if is_valid(U, V, W_sol, tol=1e-9):
                        best_U, best_V, best_W = U, V, W_sol
                        best_rank = R
                        break

            except np.linalg.LinAlgError:
                continue

    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
