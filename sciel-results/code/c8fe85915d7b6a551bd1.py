"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    schoolbook_rank = n * m * p  # 80
    
    # Build the target tensor T as a flat vector
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Indexing: U index = i*m + j, V index = j*p + k, W index = k*n + i
    T = np.ones(n * m * p)
    
    # Schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = schoolbook_rank
    
    deadline = time.time() + 140
    
    # Try to find lower rank decompositions using structured search
    # The matrix multiplication tensor for A(n×m) * B(m×p) = C(n×p)
    # has entries T[i,j,k] where (i,j,k) corresponds to C[i,k] += A[i,j]*B[j,k]
    # Rank-1 term r: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # We'll use a smooth optimization approach:
    # Minimize ||sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] - T||^2
    # with the constraint that entries are "nice" (integers or half-integers)
    
    # For each target rank, use multiple random restarts with gradient descent
    def flatten_factors(U, V, W, rank):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def unflatten_factors(x, n, m, p, rank):
        u_size = n * m * rank
        v_size = m * p * rank
        w_size = p * n * rank
        U = x[:u_size].reshape(n * m, rank)
        V = x[u_size:u_size + v_size].reshape(m * p, rank)
        W = x[u_size + v_size:].reshape(p * n, rank)
        return U, V, W
    
    def objective(x):
        U, V, W = unflatten_factors(x, n, m, p, len(x) // (n*m + m*p + p*n))
        rank = U.shape[1]
        # Compute tensor sum
        # T_comp[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # Flatten to 1D for comparison
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = U[i * m + j]
                    vk = V[j * p + k]
                    wi = W[k * n + i]
                    val = np.dot(ui, vk)  # This is wrong, need element-wise then sum with W
                    # Actually: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                    val = np.sum(ui * vk * wi)
                    error += (val - 1.0) ** 2
        return error
    
    def objective_fast(x, n, m, p, rank):
        u_size = n * m * rank
        v_size = m * p * rank
        U = x[:u_size].reshape(n * m, rank)
        V = x[u_size:u_size + v_size].reshape(m * p, rank)
        W = x[u_size + v_size:].reshape(p * n, rank)
        
        # Compute all entries efficiently
        error = 0.0
        for i in range(n):
            for j in range(m):
                ui = U[i * m + j]  # shape (rank,)
                for k in range(p):
                    vk = V[j * p + k]  # shape (rank,)
                    wi = W[k * n + i]  # shape (rank,)
                    val = np.sum(ui * vk * wi)
                    error += (val - 1.0) ** 2
        return error
    
    # Try ranks from schoolbook-1 down
    for target_rank in range(schoolbook_rank - 1, max(15, schoolbook_rank // 2 - 5), -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        n_attempts = 3 if target_rank >= 70 else 2
        
        for attempt in range(n_attempts):
            if time.time() > deadline - 20:
                break
            
            # Initialize near schoolbook but with some structure
            # Try random initialization with small perturbations
            U_init = np.random.randn(n * m, target_rank) * 0.05
            V_init = np.random.randn(m * p, target_rank) * 0.05
            W_init = np.random.randn(p * n, target_rank) * 0.05
            
            # Also try initialization close to schoolbook (first target_rank columns)
            if target_rank <= schoolbook_rank:
                U_init2 = U_school[:, :target_rank].copy()
                V_init2 = V_school[:, :target_rank].copy()
                W_init2 = W_school[:, :target_rank].copy()
                # Add small noise to escape local minima
                U_init2 += np.random.randn(n * m, target_rank) * 0.01
                V_init2 += np.random.randn(m * p, target_rank) * 0.01
                W_init2 += np.random.randn(p * n, target_rank) * 0.01
                x_init2 = flatten_factors(U_init2, V_init2, W_init2, target_rank)
            else:
                x_init2 = None
            
            x_init = flatten_factors(U_init, V_init, W_init, target_rank)
            
            # Use Nelder-Mead (derivative-free) for small problems
            # For larger ranks, use BFGS with finite differences
            if target_rank <= 25:
                method = 'Nelder-Mead'
                options = {'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            else:
                method = 'BFGS'
                options = {'maxiter': 500, 'gtol': 1e-8}
            
            try:
                result = minimize(
                    objective_fast, x_init, 
                    args=(n, m, p, target_rank),
                    method=method,
                    options=options
                )
                
                if result.fun < 1e-8:
                    U_found, V_found, W_found = unflatten_factors(result.x, n, m, p, target_rank)
                    # Round to nearest integer or half-integer
                    def round_to_half(arr):
                        return np.round(arr * 2) / 2
                    U_r = round_to_half(U_found)
                    V_r = round_to_half(V_found)
                    W_r = round_to_half(W_found)
                    
                    # Verify
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i])
                                err += (val - 1.0) ** 2
                    
                    if err < 1e-6:
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = target_rank
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break
    
    # Convert to lists with integer/half-integer entries
    def to_list_with_ints(arr):
        result = []
        for i in range(arr.shape[0]):
            row = []
            for j in range(arr.shape[1]):
                val = arr[i, j]
                # Check if it's close to an integer or half-integer
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2
                if abs(val - nearest_int) < 1e-6:
                    row.append(int(nearest_int))
                elif abs(val - nearest_half) < 1e-6:
                    row.append(nearest_half)
                else:
                    # Round to nearest half-integer anyway
                    row.append(round(val * 2) / 2)
            result.append(row)
        return result
    
    U_list = to_list_with_ints(best_U)
    V_list = to_list_with_ints(best_V)
    W_list = to_list_with_ints(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
