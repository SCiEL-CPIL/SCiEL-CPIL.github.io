import numpy as np
import random
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 5]

def check_decomposition_exact(U, V, W, n, m, p):
    """
    Exact check for integer/half-integer decomposition.
    U, V, W are lists of lists of floats (representing ints/half-ints).
    """
    R = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for r in range(R):
                    val += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                if not np.isclose(val, 1.0, atol=1e-9):
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    
    # Generate Schoolbook
    U_school = np.zeros((n * m, base_rank), dtype=float)
    V_school = np.zeros((m * p, base_rank), dtype=float)
    W_school = np.zeros((p * n, base_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Try to find lower rank decomposition using optimization
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    total_dim_factor = dim_U + dim_V + dim_W
    
    # Precompute indices for the 60 constraints
    # idx -> (i, j, k, u_idx, v_idx, w_idx)
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                constraints.append((u_idx, v_idx, w_idx))

    def residuals(x, R):
        # x is flat vector
        U_flat = x[:R * dim_U]
        V_flat = x[R * dim_U : R * dim_U + R * dim_V]
        W_flat = x[R * dim_U + R * dim_V : ]
        
        # Instead of reshaping and dotting in a loop, we can use einsum or simple loops
        # For speed, we stick to loops as R is small
        U_mat = U_flat.reshape(dim_U, R)
        V_mat = V_flat.reshape(dim_V, R)
        W_mat = W_flat.reshape(dim_W, R)
        
        res = np.zeros(len(constraints))
        for c_idx, (u_idx, v_idx, w_idx) in enumerate(constraints):
            # val = sum_r U[u,r] * V[v,r] * W[w,r]
            res[c_idx] = np.dot(U_mat[u_idx], V_mat[v_idx] * W_mat[w_idx]) - 1.0
        return res

    best_solution = None
    
    # Search for R < base_rank
    # We limit the search to save time
    # Rank 59 is the most likely immediate improvement
    search_ranks = range(base_rank - 1, max(1, base_rank - 5), -1)
    max_attempts_per_rank = 3
    
    for R in search_ranks:
        found = False
        for attempt in range(max_attempts_per_rank):
            # Random initialization with small integers
            x0 = np.random.randint(-2, 3, size=R * total_dim_factor).astype(float)
            
            try:
                sol = least_squares(residuals, x0, args=(R,), method='trf', max_nfev=20000)
                
                if sol.success and np.max(np.abs(sol.fun)) < 1e-6:
                    x_sol = sol.x
                    U_sol = x_sol[:R * dim_U].reshape(dim_U, R)
                    V_sol = x_sol[R * dim_U : R * dim_U + R * dim_V].reshape(dim_V, R)
                    W_sol = x_sol[R * dim_U + R * dim_V : ].reshape(dim_W, R)
                    
                    # Check proximity to half-integers
                    U_2 = U_sol * 2
                    V_2 = V_sol * 2
                    W_2 = W_sol * 2
                    
                    U_round = np.round(U_2)
                    V_round = np.round(V_2)
                    W_round = np.round(W_2)
                    
                    # Threshold for "close enough"
                    tol = 1e-4
                    if (np.max(np.abs(U_2 - U_round)) < tol and
                        np.max(np.abs(V_2 - V_round)) < tol and
                        np.max(np.abs(W_2 - W_round)) < tol):
                        
                        U_final = (U_round / 2).tolist()
                        V_final = (V_round / 2).tolist()
                        W_final = (W_round / 2).tolist()
                        
                        if check_decomposition_exact(U_final, V_final, W_final, n, m, p):
                            best_solution = {"U": U_final, "V": V_final, "W": W_final}
                            found = True
                            break
            except Exception:
                pass
                
        if found:
            # Continue searching for lower rank?
            # Given time constraints, returning the first found lower rank is a good trade-off.
            # But the task says "rank-minimal".
            # If we find 59, we should try 58.
            # The loop continues.
            pass

    if best_solution:
        return best_solution
    else:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

# EVOLVE-BLOCK-END
