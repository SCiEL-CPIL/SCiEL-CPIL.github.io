import numpy as np
import time
import random

DIMENSIONS = [4, 4, 9]
RANK_TARGET = 135
DEADLINE = 145.0  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    R = RANK_TARGET
    
    # Schoolbook fallback (rank 144)
    def schoolbook():
        rank_school = n * m * p
        U_s = np.zeros((n * m, rank_school))
        V_s = np.zeros((m * p, rank_school))
        W_s = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, col] = 1
                    V_s[j * p + k, col] = 1
                    W_s[k * n + i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Precompute equation indices
    eq_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_indices.append((i, j, k))
    num_eqs = n * m * p  # 144
    num_W_vars = R * p * n  # 135 * 36 = 4860
    
    # For each equation (i,j,k) and each column c, the coefficient of W[k*n+i, c]
    # is U[i*m+j, c] * V[j*p+k, c]
    
    # Strategy: sample U, V with controlled sparsity, solve for W
    
    def solve_for_W(U, V):
        """Given U (n*m x R) and V (m*p x R), solve for W."""
        # Build A matrix: A[eq_idx, w_idx] = U[i*m+j, c] * V[j*p+k, c]
        # where w_idx = c * p * n + k * n + i
        A = np.zeros((num_eqs, num_W_vars))
        b = np.ones(num_eqs)
        
        for eq_idx, (i, j, k) in enumerate(eq_indices):
            for c in range(R):
                u_val = U[i * m + j, c]
                v_val = V[j * p + k, c]
                if u_val != 0 and v_val != 0:
                    w_idx = c * p * n + k * n + i
                    A[eq_idx, w_idx] = u_val * v_val
        
        # Solve A W = b using least squares
        try:
            W_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            return None
        
        # Check if W is half-integer
        W_rounded = np.round(W_flat * 2) / 2
        if np.allclose(W_flat, W_rounded, atol=1e-8):
            # Verify exactness
            W = W_rounded.reshape(R, p, n)
            # Verify: sum_c U[i,j,c] * V[j,k,c] * W[c,k,i] == 1 for all i,j,k
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(R):
                            s += U[i * m + j, c] * V[j * p + k, c] * W[c, k, i]
                        if abs(s - 1.0) > 1e-8:
                            return None
            # Convert to required format
            W_out = np.zeros((p * n, R))
            for c in range(R):
                for k in range(p):
                    for i in range(n):
                        W_out[k * n + i, c] = W[c, k, i]
            return U, V, W_out
        return None
    
    # Sampling strategy: for each column, pick a small number of non-zero entries in U and V
    # to keep the system well-conditioned
    
    random.seed(42)
    
    for attempt in range(2000):
        if time.time() - start_time > DEADLINE:
            break
        
        # Sample U and V with structured sparsity
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        
        for c in range(R):
            # For each column, choose a random subset of entries to be non-zero
            # U has n*m = 16 entries per column
            # V has m*p = 36 entries per column
            
            # Choose ~4-6 non-zero entries in U column
            num_U_nz = random.randint(3, 7)
            U_indices = random.sample(range(n * m), min(num_U_nz, n * m))
            for idx in U_indices:
                U[idx, c] = 1.0
            
            # Choose ~6-10 non-zero entries in V column
            num_V_nz = random.randint(5, 11)
            V_indices = random.sample(range(m * p), min(num_V_nz, m * p))
            for idx in V_indices:
                V[idx, c] = 1.0
        
        result = solve_for_W(U, V)
        if result is not None:
            U_mat, V_mat, W_mat = result
            best_result = {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
            best_rank = R
            break  # Found a valid decomposition at rank 135
    
    # If we haven't found rank 135, try rank 140 with more attempts
    if best_rank == n * m * p and time.time() - start_time < DEADLINE - 10:
        R2 = 140
        num_W_vars2 = R2 * p * n
        
        for attempt in range(500):
            if time.time() - start_time > DEADLINE:
                break
            
            U = np.zeros((n * m, R2))
            V = np.zeros((m * p, R2))
            
            for c in range(R2):
                num_U_nz = random.randint(3, 7)
                U_indices = random.sample(range(n * m), min(num_U_nz, n * m))
                for idx in U_indices:
                    U[idx, c] = 1.0
                
                num_V_nz = random.randint(5, 11)
                V_indices = random.sample(range(m * p), min(num_V_nz, m * p))
                for idx in V_indices:
                    V[idx, c] = 1.0
            
            # Build A for rank 140
            A = np.zeros((num_eqs, num_W_vars2))
            b = np.ones(num_eqs)
            
            for eq_idx, (i, j, k) in enumerate(eq_indices):
                for c in range(R2):
                    u_val = U[i * m + j, c]
                    v_val = V[j * p + k, c]
                    if u_val != 0 and v_val != 0:
                        w_idx = c * p * n + k * n + i
                        A[eq_idx, w_idx] = u_val * v_val
            
            try:
                W_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
            
            W_rounded = np.round(W_flat * 2) / 2
            if np.allclose(W_flat, W_rounded, atol=1e-8):
                W = W_rounded.reshape(R2, p, n)
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(R2):
                                s += U[i * m + j, c] * V[j * p + k, c] * W[c, k, i]
                            if abs(s - 1.0) > 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    W_out = np.zeros((p * n, R2))
                    for c in range(R2):
                        for k in range(p):
                            for i in range(n):
                                W_out[k * n + i, c] = W[c, k, i]
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W_out.tolist()}
                    best_rank = R2
                    break
    
    return best_result
