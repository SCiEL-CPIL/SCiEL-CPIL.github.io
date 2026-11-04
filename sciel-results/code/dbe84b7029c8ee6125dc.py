import numpy as np
import time
import random

DIMENSIONS = [2, 4, 6]

def _verify_decomposition(U, V, W, n, m, p, num_tests=5, seed=42):
    """Verify that the tensor decomposition exactly matches matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    rng = np.random.RandomState(seed)
    for _ in range(num_tests):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        
        # Efficient verification using einsum if possible, or loops
        # C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is a batched matrix multiplication structure if we reshape.
        # Let's stick to loops for clarity and small sizes, but optimize inner loop.
        
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    u_row = U[i*m + j, :]
                    v_row = V[j*p + k, :]
                    w_val = W[k*n + i, :]
                    # Sum over r: sum_r u_row[r] * v_row[r] * w_val[r]
                    # This is dot(u_row * v_row, w_val)
                    s += np.dot(u_row * v_row, w_val)
                C_computed[i, k] = s
                
        if not np.allclose(C_computed, C_expected, atol=1e-8, rtol=1e-8):
            return False
    return True

def _build_trivial(n, m, p):
    """Schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m*p) + j * p + k
                U[i*m + j, r] = 1
                V[j*p + k, r] = 1
                W[k*n + i, r] = 1
    return U, V, W

def _als_solve(U, V, W, n, m, p, target_rank, time_limit, seed):
    """
    Alternating Least Squares with integer/half-integer projection.
    Initializes sparsely and attempts to fit the tensor.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Initialize with random sparse vectors
    # We want entries to be 0, 0.5, 1, -1, -0.5
    # Let's use a small probability of non-zero.
    p_nonzero = 0.2
    
    def init_matrix(rows, rank):
        M = np.zeros((rows, rank))
        for i in range(rows):
            for r in range(rank):
                if rng.rand() < p_nonzero:
                    # Choose from {-1, -0.5, 0.5, 1}
                    M[i, r] = rng.choice([-1, -0.5, 0.5, 1])
        return M

    U = init_matrix(n*m, target_rank)
    V = init_matrix(m*p, target_rank)
    W = init_matrix(p*n, target_rank)
    
    # Pre-compute indices for faster access
    # We need to solve for U given V, W, etc.
    # The tensor T is defined by T[i,j,k] = 1 if i==j==k? No.
    # The tensor for matrix mult is T_{i,j,k} such that C[i,k] = sum_{j} A[i,j] B[j,k] T_{i,j,k}?
    # No, the standard tensor is T_{i, j, k} where the contraction is:
    # C[i,k] = sum_{j, r} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This is equivalent to the tensor T having components T_{(i,j), (j,k), (k,i)}?
    # Actually, the "tensor" we are decomposing is the structure of the multiplication.
    # Let's define the target tensor T with dimensions (n*m, m*p, p*n) is too big.
    # Instead, we view it as a set of equations.
    
    # For each (i, k) in C, we have:
    # C[i,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Let's define a matrix M_ik of size (m*rank, m*rank)? No.
    
    # Let's use the standard ALS formulation for tensor decomposition.
    # We are minimizing || T - sum_r U_r V_r W_r ||_F
    # Where T is the 3-way tensor with T[i, j, k] = 1 if i,j,k form a valid multiplication path?
    # No, the "Tensor" of matrix multiplication M_{n,m,p} has entries:
    # T[i, j, k] = 1 if we are computing C[i,k] from A[i,j] and B[j,k].
    # Wait, the tensor is usually defined such that the multiplication is a contraction.
    # T_{i, j, k} where i indexes A, j indexes B, k indexes C?
    # Standard definition: T is a tensor of size n x m x p? No.
    # The tensor is of size (n*m) x (m*p) x (p*n)?
    # Let's stick to the equation form.
    
    # Equation for C[i,k]:
    # sum_{j=0}^{m-1} sum_{r=0}^{R-1} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 (if we consider the "1" coefficient)
    # Wait, the target is not a fixed tensor of 1s.
    # The target is the *operation* itself.
    # For any A, B, the result is A@B.
    # This implies that the decomposition must satisfy:
    # sum_r U[:,r] V[:,r] W[:,r] (as a tensor) represents the multiplication tensor.
    
    # The multiplication tensor T has non-zero entries T[i, j, k] = 1 for all i,j,k?
    # No.
    # Let's look at the definition of the tensor for matrix multiplication.
    # It is a tensor T in U* x V* x W* such that T(A,B) = A B.
    # In coordinates: T_{i, j, k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    # So T_{i, j, k} = 1 if the indices match?
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So the tensor T has T_{i, j, k} = 1 for all i, j, k?
    # No, the indices i, j, k are coupled.
    # The tensor is actually of size n x m x p? No.
    # The tensor is of size (n*m) x (m*p) x (p*n)?
    
    # Let's re-read the task description carefully.
    # "Factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "W[k*n+i, r]"
    # This confirms the indices.
    # The condition is:
    # For all A, B:
    # (A B)_{i,k} = sum_{j, r} A_{i,j} B_{j,k} U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    
    # This must hold for ALL A, B.
    # This implies that the tensor defined by the sum must match the identity mapping for the multiplication.
    # Let's fix A and B to basis vectors to extract the tensor components.
    # Let A = E_{i,j} (1 at i,j, 0 elsewhere).
    # Let B = E_{j',k} (1 at j',k, 0 elsewhere).
    # Then (A B)_{i', k'} is 1 if i'=i, k'=k, and j=j'. Otherwise 0.
    # So (A B)_{i,k} = 1 if i'=i, k'=k, j=j'.
    
    # LHS: (A B)_{i', k'}
    # RHS: sum_{j, r} A_{i,j} B_{j,k} ...
    # A_{i,j} is non-zero only if i=i, j=j. So A_{i,j} = 1.
    # B_{j,k} is non-zero only if j=j', k=k. So B_{j,k} = 1 if j=j'.
    # So the sum is over j.
    # If j != j', B_{j,k} = 0.
    # So the sum reduces to the term where j=j'.
    # RHS = sum_r U_{i*m+j', r} V_{j'*p+k, r} W_{k*n+i, r}
    
    # We require RHS = LHS.
    # LHS is 1 if i'=i and k'=k and j=j'.
    # But i' and k' are the output indices.
    # In the sum, i and k are the input indices to U and V?
    # Wait, the indices in U are (i, j). In V are (j, k).
    # The output is C_{i, k}.
    # So for a specific basis A=E_{i,j} and B=E_{j,k}, the output C should be E_{i,k}.
    # So C_{i,k} should be 1, and all other C_{i',k'} should be 0.
    
    # Let's check the equation for a specific output (i_out, k_out).
    # C_{i_out, k_out} = sum_{j_in, r} A_{i_out, j_in} B_{j_in, k_out} U_{i_out*m+j_in, r} V_{j_in*p+k_out, r} W_{k_out*n+i_out, r}
    
    # If A = E_{i0, j0} and B = E_{j1, k0}:
    # A_{i_out, j_in} is 1 only if i_out=i0 and j_in=j0.
    # B_{j_in, k_out} is 1 only if j_in=j1 and k_out=k0.
    # So for the term to be non-zero, we need:
    # i_out = i0
    # j_in = j0
    # j_in = j1  => j0 = j1
    # k_out = k0
    
    # So if j0 != j1, then for ALL i_out, k_out, the sum is 0.
    # This matches C=0.
    
    # If j0 = j1 = j:
    # Then for i_out=i0 and k_out=k0:
    # Sum_r U_{i0*m+j, r} V_{j*p+k0, r} W_{k0*n+i0, r} should be 1.
    # For any other i_out != i0 or k_out != k0:
    # The sum should be 0.
    
    # So the conditions are:
    # 1. For all i, j, k: sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1
    # 2. For all (i,j,k) and (i',j,k') with i'!=i or k'!=k:
    #    sum_r U_{i'*m+j, r} V_{j*p+k', r} W_{k'*n+i', r} = 0 ?
    #    Wait, if we pick A=E_{i,j} and B=E_{j,k'}, the output is E_{i,k'}.
    #    So C_{i,k'} should be 1.
    #    So the condition is simply:
    #    For all i, j, k: sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1
    
    # Is that it?
    # Let's check cross terms.
    # What if A has multiple non-zero entries?
    # Linearity handles that.
    # So the ONLY constraint is:
    # For all i in [0,n), j in [0,m), k in [0,p):
    # sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p equations.
    # Unknowns: R * (n*m + m*p + p*n).
    # We want minimal R.
    
    # This is a much simpler formulation!
    # We need to find matrices U, V, W such that:
    # For all (i,j,k): <U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]> = 1
    
    # This is a set of polynomial equations.
    # We can use ALS to minimize the error:
    # Error = sum_{i,j,k} (sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} - 1)^2
    
    # Let's implement ALS for this specific error function.
    
    def project(M):
        # Project to half-integers
        # Round to nearest 0.5
        return np.round(M * 2) / 2.0

    best_error = float('inf')
    
    for step in range(100):
        if time.time() - start_time > time_limit:
            break
            
        # Update U
        # For each row i*m+j of U, we want to minimize:
        # sum_{k} (sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} - 1)^2
        # This is not a simple linear least squares because of the product of 3 terms.
        # However, if we fix V and W, the term inside the square is linear in U.
        # Let x = U_{i,j, :} (vector of length R)
        # Let b_{i,j,k} = V_{j,k, :] * W_{k,i, :} (vector of length R)
        # We want to minimize sum_k (x . b_{i,j,k} - 1)^2
        # This is a linear least squares problem for x!
        # Minimize || B x - 1 ||^2 where B is a matrix with rows b_{i,j,k}^T.
        
        # Construct B for a given (i,j)
        # Rows correspond to k in 0..p-1
        # B[k, r] = V[j*p+k, r] * W[k*n+i, r]
        
        # Similarly for V and W.
        
        # Update U
        for i in range(n):
            for j in range(m):
                # Build matrix B (p x R)
                B_mat = np.zeros((p, target_rank))
                for k in range(p):
                    B_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve min || B_mat @ x - 1 ||^2
                # x is U[i*m+j, :]
                # Using lstsq
                try:
                    x, _, _, _ = np.linalg.lstsq(B_mat, np.ones(p), rcond=None)
                    U[i*m+j, :] = x
                except:
                    pass
                    
        # Update V
        for j in range(m):
            for k in range(p):
                # Build matrix B (n x R)
                # For fixed j, k, we vary i.
                # Term: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                # Let y = V[j*p+k, :]
                # For each i, b_i = U[i*m+j, :] * W[k*n+i, :]
                # Minimize sum_i (y . b_i - 1)^2
                B_mat = np.zeros((n, target_rank))
                for i in range(n):
                    B_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    y, _, _, _ = np.linalg.lstsq(B_mat, np.ones(n), rcond=None)
                    V[j*p+k, :] = y
                except:
                    pass
                    
        # Update W
        for k in range(p):
            for i in range(n):
                # Build matrix B (m x R)
                # For fixed k, i, we vary j.
                # Term: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                # Let z = W[k*n+i, :]
                # For each j, b_j = U[i*m+j, :] * V[j*p+k, :]
                # Minimize sum_j (z . b_j - 1)^2
                B_mat = np.zeros((m, target_rank))
                for j in range(m):
                    B_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    z, _, _, _ = np.linalg.lstsq(B_mat, np.ones(m), rcond=None)
                    W[k*n+i, :] = z
                except:
                    pass
        
        # Project to half-integers
        U = project(U)
        V = project(V)
        W = project(W)
        
        # Calculate error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                    error += (s - 1.0)**2
                    if error > 1e-6:
                        break
                if error > 1e-6:
                    break
            if error > 1e-6:
                break
                
        if error < best_error:
            best_error = error
            if error < 1e-8:
                return U, V, W, True
                
    return U, V, W, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Target rank: Try 40 first (theoretical lower bound for 2x4x6 is 40? 
    # Actually, R(2,4,6) >= R(2,4,4)+R(2,4,2)? 
    # R(2,4,2) is 8? R(2,4,4) is 20? 
    # 20+8=28? 
    # Let's check known ranks.
    # R(2,2,2)=7.
    # R(2,4,6) is likely around 40-42.
    # Let's try 40, then 41, then 42.
    
    target_ranks = [40, 41, 42]
    
    best_rank = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Fallback to trivial
    U48, V48, W48 = _build_trivial(n, m, p)
    best_rank = 48
    best_U, best_V, best_W = U48, V48, W48
    
    for rank in target_ranks:
        if time.time() > deadline:
            break
            
        # Try multiple seeds
        for seed in range(10):
            if time.time() > deadline:
                break
                
            time_per_seed = min(10, (deadline - time.time()) / (10 - seed))
            if time_per_seed < 1:
                break
                
            U, V, W, success = _als_solve(U48[:, :rank], V48[:, :rank], W48[:, :rank], n, m, p, rank, time_per_seed, seed)
            
            if success:
                # Verify
                if _verify_decomposition(U, V, W, n, m, p, num_tests=5, seed=seed+100):
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    break # Found rank, stop searching for this rank
            
            # Even if not fully converged, check if error is small enough?
            # No, must be exact.
            
        if best_rank == rank:
            break # Found the lowest possible rank we tried
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
