"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe within 150s limit
    
    n, m, p = DIMENSIONS
    
    # Build the matrix multiplication tensor T[i,j,k] = 1 if output (i,k) depends on A[i,j]*B[j,k]
    # T is a 3-tensor of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # We want to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # This is equivalent to: for all (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # And for all other combinations, the sum should be 0.
    
    # Actually, the tensor is: M[i,j,k] = delta[i,k] * delta[j,j] ... no.
    # Matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all valid (i,j,k) in the "diagonal" sense.
    # More precisely, the multiplication tensor is sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C where 
    # C[i,k] = sum_j A[i,j]B[j,k], so T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k].
    # The tensor is: T = sum_{i,j,k} (e_i ⊗ e_j ⊗ e_k) where the indices are (input_A, input_B, output)
    # with input_A = (i,j), input_B = (j,k), output = (i,k)
    
    # So T[(i,j),(j,k),(i,k)] = 1, and 0 otherwise.
    
    # Number of nonzero entries: n*m*p = 125
    
    # Let's try to find a low-rank decomposition using optimization.
    # We'll parameterize U, V, W and minimize the error.
    
    def build_tensor():
        """Build the matrix multiplication tensor as a vector."""
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        return T
    
    T = build_tensor()
    
    # Try different ranks and find the smallest that works
    # Start with a reasonable upper bound and try to go lower
    
    # First, let's try rank 100-125 with random initialization and optimization
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Initialize with schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    def compute_error(U, V, W):
        """Compute the error of the tensor decomposition."""
        # For each (i,j,k) in the tensor, check if sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = T[i,j,k]
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    errors.append(val - 1.0)
        # Also check that off-diagonal terms are zero
        # The tensor is zero for all (a,b,c) where c != (i,k) for the corresponding (i,j)
        # Actually, we need to check all (row_U, col_V, row_W) combinations
        # row_U = i*m+j, col_V = j'*p+k', row_W = k''*n+i''
        # The tensor value is 1 if i''==i and k''==k and j==j' (from A[i,j] and B[j,k] contributing to C[i,k])
        # Wait, I need to be more careful.
        
        # The multiplication tensor T has:
        # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some (i,j,k)
        # T[a, b, c] = 0 otherwise
        
        # So we need: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b,c
        
        # This is a lot of constraints: (n*m)*(m*p)*(p*n) = 125^3 = 1953125 constraints
        # That's too many to check all. Let's just check the 125 nonzero ones and a sample of zeros.
        
        return np.sum(np.array(errors)**2)
    
    # This is getting complex. Let me try a different approach.
    # Use the fact that we can verify by actually multiplying matrices.
    
    def verify_decomposition(U, V, W, num_tests=10):
        """Verify the decomposition by testing with random matrices."""
        np.random.seed(42)
        for _ in range(num_tests):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            # Compute C using the decomposition
            # C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # Wait, that's not right. Let me reconsider.
            
            # The decomposition says: for the multiplication tensor,
            # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            # where a indexes A entries, b indexes B entries, c indexes C entries.
            
            # To compute C[i,k] = sum_j A[i,j] * B[j,k]:
            # We can write this as: C[i,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] * (1/U[i*m+j,r]) ... no.
            
            # Actually, the standard way: if T = sum_r u_r ⊗ v_r ⊗ w_r, then
            # C = sum_r (u_r * A_flattened) ⊙ (v_r * B_flattened) reshaped... 
            # No, let me think again.
            
            # The multiplication tensor is T = sum_{i,j,k} e_{i,j}^A ⊗ e_{j,k}^B ⊗ e_{i,k}^C
            # A rank-r decomposition is T = sum_{r=1}^R U_r ⊗ V_r ⊗ W_r
            # where U_r is a vector in R^{n*m}, V_r in R^{m*p}, W_r in R^{p*n}
            
            # Then: C = sum_r (U_r^T * A_flat) * (V_r^T * B_flat) * W_r^T
            # where A_flat is A reshaped to (n*m,), B_flat to (m*p,), and W_r is reshaped to (p,n) then transposed to (n,p)?
            
            # Let me verify: (U_r^T * A_flat) = sum_{i,j} U_r[i*m+j] * A[i,j]
            # (V_r^T * B_flat) = sum_{j,k} V_r[j*p+k] * B[j,k]
            # Then (U_r^T * A_flat) * (V_r^T * B_flat) * W_r[k*n+i] summed over r gives C[i,k]?
            
            # For the schoolbook: U_r[i*m+j]=1, V_r[j*p+k]=1, W_r[k*n+i]=1 for specific (i,j,k)
            # Then (U_r^T * A_flat) = A[i,j], (V_r^T * B_flat) = B[j,k], W_r[k*n+i]=1
            # Sum over r: sum_{i',j',k'} A[i',j'] * B[j',k'] * delta[i',i]*delta[j',j]*delta[k',k] ... 
            # No, each r corresponds to a specific (i,j,k), so U_r is 1 only at (i,j), V_r only at (j,k), W_r only at (k,i).
            # So (U_r^T * A_flat) = A[i,j], (V_r^T * B_flat) = B[j,k], W_r[k*n+i] = 1
            # Sum over all r: sum_{i',j',k'} A[i',j'] * B[j',k'] = sum_j' A[i,j'] * B[j',k] ... 
            # Wait, for a given r corresponding to (i',j',k'), the term is A[i',j'] * B[j',k'] * 1 (since W_r[k*n+i] = 1 only if k=k' and i=i')
            # So the sum is sum_{i',j',k': i'=i, k'=k} A[i',j'] * B[j',k'] = sum_{j'} A[i,j'] * B[j',k] = C[i,k] ✓
            
            C_computed = np.zeros((n, p))
            for r in range(U.shape[1]):
                u_val = np.dot(U[:, r], A.flatten())
                v_val = np.dot(V[:, r], B.flatten())
                C_computed += u_val * v_val * W[:, r].reshape(p, n).T
            
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True
    
    # Let me try a smarter approach: use known decompositions.
    # For 5x5x5, let's try to use a block structure with Winograd for 3x3.
    
    # Actually, let me just try to optimize for lower ranks starting from schoolbook.
    
    # Try ranks from 100 to 125
    for target_rank in [100, 105, 110, 115, 120, 125]:
        if time.time() > deadline - 20:
            break
            
        # Initialize with random values
        np.random.seed(42 + target_rank)
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        # Optimize to minimize error
        def objective(params):
            U_p = params[:n*m*target_rank].reshape(n*m, target_rank)
            V_p = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_p = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_p[i*m+j, :], V_p[j*p+k, :] * W_p[k*n+i, :])
                        error += (val - 1.0)**2
            return error
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use Nelder-Mead for small problems, or just try a few iterations
        # This might be too slow for large parameter spaces. Let's try a simpler approach.
        
        # Actually, let's just verify the schoolbook and return it if we can't do better.
        if verify_decomposition(U_school, V_school, W_school):
            if best_result is None:
                best_result = (U_school, V_school, W_school)
                best_rank = rank_school
    
    # Convert to lists for output
    if best_result is not None:
        U_out, V_out, W_out = best_result
    else:
        U_out, V_out, W_out = U_school, V_school, W_school
    
    return {
        "U": U_out.tolist(),
        "V": V_out.tolist(),
        "W": W_out.tolist()
    }
# EVOLVE-BLOCK-END
