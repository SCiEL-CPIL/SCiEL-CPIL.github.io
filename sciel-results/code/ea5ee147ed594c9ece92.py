"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin before 150s kill

    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    best_U, best_V, best_W = U, V, W

    def validate_fast(U_mat, V_mat, W_mat):
        """Check if the decomposition exactly represents matrix multiplication using vectorized ops."""
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        rank = U_arr.shape[1]
        
        # Compute all 75 tensor entries at once
        # Tensor entry (i,j,k) = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Reshape to (n, m, p)
        results = np.zeros((n, m, p))
        for r in range(rank):
            u_col = U_arr[:, r]  # shape (n*m,)
            v_col = V_arr[:, r]  # shape (m*p,)
            w_col = W_arr[:, r]  # shape (p*n,)
            # u_col reshaped to (n, m), v_col to (m, p), w_col to (p, n)
            u_reshaped = u_col.reshape(n, m)
            v_reshaped = v_col.reshape(m, p)
            w_reshaped = w_col.reshape(p, n)
            # Outer product: (n,m) x (m,p) x (p,n) -> need elementwise
            # (i,j,k) = u_reshaped[i,j] * v_reshaped[j,k] * w_reshaped[k,i]
            # This is a tensor contraction
            for i in range(n):
                for j in range(m):
                    results[i, j, :] += u_reshaped[i, j] * v_reshaped[j, :] * w_reshaped[:, i]
        
        if not np.allclose(results, 1.0, atol=1e-6):
            return False
        return True

    def try_rank(R, max_attempts=50):
        """Try to find a rank-R decomposition with integer/half-integer entries."""
        if time.time() > deadline - 15:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Generate random integer U, V with various patterns
            if attempt % 3 == 0:
                # Pattern 1: {-1, 0, 1}
                U_rand = np.random.randint(-1, 2, (n * m, R)).astype(float)
                V_rand = np.random.randint(-1, 2, (m * p, R)).astype(float)
            elif attempt % 3 == 1:
                # Pattern 2: {-1, 1}
                U_rand = np.random.choice([-1.0, 1.0], size=(n * m, R))
                V_rand = np.random.choice([-1.0, 1.0], size=(m * p, R))
            else:
                # Pattern 3: {-2, -1, 0, 1, 2}
                U_rand = np.random.randint(-2, 3, (n * m, R)).astype(float)
                V_rand = np.random.randint(-2, 3, (m * p, R)).astype(float)
            
            # Build linear system A x = b for W
            # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Unknowns are W flattened: W[k*n+i, r] indexed as (k*n+i)*R + r
            num_eqs = n * m * p
            num_vars = p * n * R
            
            # Build A more efficiently using precomputed indices
            A = np.zeros((num_eqs, num_vars))
            b = np.ones(num_eqs)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    u_row = i * m + j
                    for k in range(p):
                        v_row = j * p + k
                        w_row = k * n + i
                        for r in range(R):
                            A[idx, w_row * R + r] = U_rand[u_row, r] * V_rand[v_row, r]
                        idx += 1
            
            try:
                # Check if system is consistent and has a solution
                # Use lstsq to find minimum-norm solution
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                W_rand = W_flat.reshape(p * n, R)
                
                # Check if W_rand is close to half-integers
                W_check = W_rand * 2
                dist = np.abs(W_check - np.round(W_check))
                max_dist = np.max(dist)
                
                if max_dist < 1e-4:
                    W_int = np.round(W_check) / 2.0
                    
                    # Validate the exact solution
                    if validate_fast(U_rand.tolist(), V_rand.tolist(), W_int.tolist()):
                        return {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_int.tolist()}
                        
            except Exception:
                pass
                
        return None

    # Try ranks from 74 down to 70 with increasing effort
    for R in range(rank_schoolbook - 1, 69, -1):
        if time.time() > deadline - 25:
            break
            
        # Allocate more attempts for lower ranks (harder to find)
        remaining_time = deadline - time.time() - 10
        if remaining_time < 5:
            break
            
        max_att = max(10, int(remaining_time * 3))
        result = try_rank(R, max_attempts=max_att)
        if result:
            return result
            
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
