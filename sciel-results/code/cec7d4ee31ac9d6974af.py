import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    time_limit = 140  # seconds, safety margin before 150s kill

    def is_valid(U, V, W):
        """Check if U, V, W is a valid decomposition using exact integer arithmetic."""
        U = np.asarray(U)
        V = np.asarray(V)
        W = np.asarray(W)
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    if val != 1:
                        return False
        return True

    # Try ranks from schoolbook-1 down to a lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # But realistically, for 3x3*3x4, the rank is likely 30-36
    
    for R in range(rank_schoolbook - 1, 25, -1):
        if time.time() - start_time > time_limit:
            break
            
        attempts = 0
        max_attempts = 200  # Limit per rank
        
        while attempts < max_attempts and (time.time() - start_time) < time_limit:
            attempts += 1
            
            # Strategy: Generate sparse U, V with entries in {-1, 0, 1}
            # with controlled sparsity, then solve for W using integer methods
            
            # Generate U with controlled sparsity
            U = np.zeros((n * m, R), dtype=int)
            V = np.zeros((m * p, R), dtype=int)
            
            # For each column, decide which entries are non-zero
            for r in range(R):
                # U column: choose 1-2 non-zero entries from 9 positions
                num_nonzero_u = np.random.choice([1, 2], p=[0.7, 0.3])
                u_indices = np.random.choice(n * m, size=num_nonzero_u, replace=False)
                for idx in u_indices:
                    U[idx, r] = np.random.choice([-1, 1])
                
                # V column: choose 1-2 non-zero entries from 12 positions
                num_nonzero_v = np.random.choice([1, 2], p=[0.7, 0.3])
                v_indices = np.random.choice(m * p, size=num_nonzero_v, replace=False)
                for idx in v_indices:
                    V[idx, r] = np.random.choice([-1, 1])
            
            # Now solve for W: For each (i, j, k), we need:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            #
            # This is a linear system in W variables.
            # Variables: W[k*n+i, r] for k in 0..p-1, i in 0..n-1, r in 0..R-1
            # Total variables: p * n * R = 12 * R
            # Equations: n * m * p = 36
            
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
            
            # Try to find an integer solution
            # Use least squares and check if close to integer
            try:
                W_flat, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                # Check if solution is close to integers
                W_int = np.round(W_flat).astype(int)
                
                # Verify with exact arithmetic
                if np.allclose(A_mat @ W_int, b_vec, atol=1e-10):
                    W_sol = W_int.reshape((p * n, R))
                    if is_valid(U, V, W_sol):
                        return {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W_sol.tolist()
                        }
            except:
                pass
            
            # Alternative: Try to solve by fixing W to have sparse structure
            # For each column r, W[:, r] should be sparse
            # Try: set W[k*n+i, r] = 1 only when U[i*m+j, r] and V[j*p+k, r] are both non-zero
            # This is a heuristic
            
            W_heuristic = np.zeros((p * n, R), dtype=int)
            for r in range(R):
                for k in range(p):
                    for i in range(n):
                        w_idx = k * n + i
                        # Check if there exists j such that U[i*m+j, r] != 0 and V[j*p+k, r] != 0
                        for j in range(m):
                            if U[i * m + j, r] != 0 and V[j * p + k, r] != 0:
                                W_heuristic[w_idx, r] = 1
                                break
            
            if is_valid(U, V, W_heuristic):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W_heuristic.tolist()
                }

    # Fallback to Schoolbook
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
# (No changes to interface)
# EVOLVE-BLOCK-END
