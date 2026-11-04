"""Matrix multiplication 4,4,5 rank-minimal decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 135  # seconds, safety margin before 150s kill
    
    # The tensor T has T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # We need rank-r decomposition: sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1
    
    # Start with schoolbook rank = n*m*p = 80
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize schoolbook
    U = np.zeros((n * m, best_rank), dtype=float)
    V = np.zeros((m * p, best_rank), dtype=float)
    W = np.zeros((p * n, best_rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Try to reduce rank by local search
    # For each target rank r from 79 down, try to find a valid decomposition
    for target_rank in range(best_rank - 1, 15, -1):
        if time.time() - start_time > deadline:
            break
        
        # Try multiple random restarts
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() - start_time > deadline:
                break
            
            # Random initialization: start from schoolbook and zero out some columns,
            # then try to fix the remaining
            U_try = np.zeros((n * m, target_rank), dtype=float)
            V_try = np.zeros((m * p, target_rank), dtype=float)
            W_try = np.zeros((p * n, target_rank), dtype=float)
            
            # Initialize with random small values plus some structure
            # Use a mix: keep some schoolbook columns, add noise
            num_schoolbook = min(target_rank, 40)
            for c in range(num_schoolbook):
                i = c % n
                j = (c // n) % m
                k = (c // (n * m)) % p
                U_try[i * m + j, c] = 1.0
                V_try[j * p + k, c] = 1.0
                W_try[k * n + i, c] = 1.0
            
            # Add random perturbations to remaining columns
            for c in range(num_schoolbook, target_rank):
                U_try[:, c] = np.random.uniform(-0.5, 0.5, n * m)
                V_try[:, c] = np.random.uniform(-0.5, 0.5, m * p)
                W_try[:, c] = np.random.uniform(-0.5, 0.5, p * n)
            
            # Local search: gradient-free, try to minimize error
            # Error: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 1
            # We can compute the "product tensor" and check against all-ones
            
            def compute_error(U_mat, V_mat, W_mat):
                """Compute max absolute error from 1."""
                # For each output (i,k), the value is sum_j A[i,j]*B[j,k]
                # In tensor form: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                # Let's compute M[i,k] = sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                # This should equal 1 for all i,k... wait no.
                # Actually the condition is: for all (i,j,k), the coefficient is 1.
                # That means: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for ALL i,j,k
                # This is a much stronger condition!
                
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(target_rank):
                                val += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                            max_err = max(max_err, abs(val - 1))
                return max_err
            
            # Use a simpler error: compute the "multiplication matrix" 
            # Actually, let's use a different formulation.
            # The tensor T_{i,j,k} = 1. We need sum_c U_c[i*m+j] V_c[j*p+k] W_c[k*n+i] = 1.
            
            # Let's use a vectorized approach for speed
            # For fixed (i,k), sum_j sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = n*m... no
            # The condition is elementwise: for each (i,j,k), the sum must be 1.
            
            # Let me compute it more efficiently:
            # Define M_{i,k}^{(j)} = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # We need M_{i,k}^{(j)} = 1 for all i,j,k.
            
            # Alternative: think of it as, for each j, define matrix P_j = sum_c (U[:, j] outer something)
            # This is getting complex. Let me just do direct computation but vectorized.
            
            # Actually, let's reformulate. For each c, we have a rank-1 contribution.
            # The total is sum_c (u_c ⊗ v_c ⊗ w_c) where u_c is in R^{nm}, v_c in R^{mp}, w_c in R^{pn}
            # and the tensor product is evaluated as: for indices (i,j,k), 
            # contribution is u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
            
            # Let me just compute it directly but use numpy for the inner sum
            def compute_error_fast(U_mat, V_mat, W_mat):
                errors = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            vals = U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :]
                            val = np.sum(vals)
                            errors.append(abs(val - 1.0))
                return max(errors) if errors else 0
            
            current_err = compute_error_fast(U_try, V_try, W_try)
            
            # Local search with small perturbations
            for step in range(200):
                if time.time() - start_time > deadline:
                    break
                if current_err < 1e-10:
                    break
                    
                # Pick a random column and perturb it
                c = random.randint(0, target_rank - 1)
                # Random direction
                dU = np.random.normal(0, 0.1, n * m)
                dV = np.random.normal(0, 0.1, m * p)
                dW = np.random.normal(0, 0.1, p * n)
                
                U_try[:, c] += dU
                V_try[:, c] += dV
                W_try[:, c] += dW
                
                new_err = compute_error_fast(U_try, V_try, W_try)
                
                if new_err < current_err:
                    current_err = new_err
                else:
                    # Revert
                    U_try[:, c] -= dU
                    V_try[:, c] -= dV
                    W_try[:, c] -= dW
            
            if current_err < 1e-10:
                # Found a valid decomposition!
                best_rank = target_rank
                best_U = U_try.copy()
                best_V = V_try.copy()
                best_W = W_try.copy()
                break
        
        if best_rank == target_rank:
            # Found it, try to go lower
            continue
    
    # Round to nearest half-integer for exactness
    def to_half_integer(arr):
        return np.round(arr * 2) / 2
    
    U_out = to_half_integer(best_U)
    V_out = to_half_integer(best_V)
    W_out = to_half_integer(best_W)
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(best_rank):
                    val += U_out[i*m+j, c] * V_out[j*p+k, c] * W_out[k*n+i, c]
                if abs(val - 1) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to schoolbook
        U_out = to_half_integer(best_U) if best_U is not None else None
        if U_out is None:
            rank = n * m * p
            U_out = np.zeros((n * m, rank))
            V_out = np.zeros((m * p, rank))
            W_out = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_out[i * m + j, col] = 1
                        V_out[j * p + k, col] = 1
                        W_out[k * n + i, col] = 1
                        col += 1
            best_rank = rank
    
    # Convert to lists
    return {
        "U": U_out.tolist(),
        "V": V_out.tolist(),
        "W": W_out.tolist()
    }
