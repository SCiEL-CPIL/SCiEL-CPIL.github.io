import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    n, m, p = DIMENSIONS
    N = n * m
    M = m * p
    P = p * n
    
    def validate(U, V, W, tol=1e-9):
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        for idx in range(len(u_idx)):
            u_row = U[u_idx[idx], :]
            v_row = V[v_idx[idx], :]
            w_row = W[w_idx[idx], :]
            s = np.sum(u_row * v_row * w_row)
            if abs(s - 1.0) > tol:
                return False
        return True

    def schoolbook_decomposition():
        rank = n * m * p
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook_decomposition()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    rng = np.random.RandomState(42)
    
    # Try to find a lower rank decomposition using linear solve for W
    # We fix U and V randomly and solve for W.
    # If W is close to integers, we round and check.
    
    for rank in range(44, 35, -1):
        if time.time() > deadline - 5:
            break
        
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Initialize U and V with sparse random integers
            U = np.zeros((N, rank))
            V = np.zeros((M, rank))
            
            # Ensure each column has at least one non-zero in U and V
            # and that the system is solvable
            for r in range(rank):
                # Randomly select indices to place 1s and -1s
                # Keep it sparse
                num_nonzeros_u = rng.randint(1, 4)
                num_nonzeros_v = rng.randint(1, 4)
                
                u_indices = rng.choice(N, size=num_nonzeros_u, replace=False)
                v_indices = rng.choice(M, size=num_nonzeros_v, replace=False)
                
                for idx in u_indices:
                    U[idx, r] = rng.choice([-1, 1])
                for idx in v_indices:
                    V[idx, r] = rng.choice([-1, 1])
            
            # Construct the linear system A * vec(W) = 1
            # A has shape (n*m*p, P*rank)
            # A[i*m*p + j*p + k, (k*n+i)*rank + r] = U[i*m+j, r] * V[j*p+k, r]
            
            # We can build A efficiently
            A = np.zeros((n*m*p, P*rank))
            
            i_idx = np.repeat(np.arange(n), m * p)
            j_idx = np.tile(np.repeat(np.arange(m), p), n)
            k_idx = np.tile(np.arange(p), n * m)
            
            u_idx = i_idx * m + j_idx
            v_idx = j_idx * p + k_idx
            w_idx = k_idx * n + i_idx
            
            # For each equation index (0..44) and each variable index (0..P*rank-1)
            # We need to fill A
            # This loop is 45 * P * rank. 45 * 15 * 40 = 27000 ops. Very fast.
            
            for eq_idx in range(n*m*p):
                i = i_idx[eq_idx]
                j = j_idx[eq_idx]
                k = k_idx[eq_idx]
                
                u_val = U[u_idx[eq_idx], :]
                v_val = V[v_idx[eq_idx], :]
                
                # The contribution of W[k*n+i, r] is u_val[r] * v_val[r] * W[k*n+i, r]
                # So the coefficient for variable (k*n+i)*rank + r is u_val[r] * v_val[r]
                
                coeffs = u_val * v_val
                if np.any(coeffs != 0):
                    # Map to column indices in A
                    # Variable index for W[k*n+i, r] is (k*n+i)*rank + r
                    # r ranges from 0 to rank-1
                    r_indices = np.arange(rank)
                    var_indices = (k*n+i)*rank + r_indices
                    A[eq_idx, var_indices] = coeffs
            
            # Solve A * vec_W = 1
            # Use lstsq
            try:
                vec_W, residuals, rank_A, s = np.linalg.lstsq(A, np.ones(n*m*p), rcond=None)
            except:
                continue
                
            # Check residual
            if residuals is not None and len(residuals) > 0:
                if residuals[0] > 1e-6:
                    continue
            else:
                # If rank < 45, check manually
                if np.linalg.norm(A @ vec_W - np.ones(n*m*p)) > 1e-6:
                    continue
            
            # Reshape vec_W to W
            W = vec_W.reshape((P, rank))
            
            # Check if W is close to integers
            # Round to nearest integer
            W_int = np.round(W)
            
            # Check if W_int is a valid solution
            if validate(U, V, W_int, tol=1e-9):
                best_U, best_V, best_W = U, V, W_int
                best_rank = rank
                break
        
        if best_rank < 45:
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
