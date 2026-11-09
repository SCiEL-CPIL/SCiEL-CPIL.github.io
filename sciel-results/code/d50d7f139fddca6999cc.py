import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Schoolbook decomposition as baseline
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank decompositions
    # The matrix multiplication tensor T[i,k,j] = delta structure
    # We want U (n*m x r), V (m*p x r), W (p*n x r) such that
    # for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 if this is a valid term, 0 otherwise
    # Actually the tensor is T[i,j,k] where the multiplication is C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor has T[i,j,k] = 1 for the "identity" in the middle index
    
    # Let's try various ranks and use a simple iterative approach
    # For each candidate rank r, try to solve for U,V,W
    
    # Known: for 2x5x6, let's try ranks from 55 down to 40
    for r in range(min(58, best_rank - 1), 35, -1):
        if time.time() > deadline - 10:
            break
        
        # Initialize with random small values
        np.random.seed(r * 1000 + 42)
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Iterative optimization: minimize ||tensor_product(U,V,W) - T||_F
        # where tensor_product gives the 3D array with [i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        
        def compute_tensor(U, V, W, r):
            T = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
            return T
        
        # Target tensor
        T_target = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i, j, k] = 1  # Each (i,j,k) triple contributes to C[i,k]
        
        # Actually, the tensor for matrix mult is: T[i,j,k] such that C[i,k] = sum_j A[i,j] B[j,k]
        # The rank-1 decomposition: C[i,k] = sum_c U[i*m+j_c, c] * V[j_c*p+k_c, c] * W[k_c*n+i_c, c]
        # where the indices are coupled. Let me reconsider.
        
        # The correct formulation: we want to represent the bilinear map
        # (A, B) -> C where C[i,k] = sum_j A[i,j] * B[j,k]
        # as C[i,k] = sum_c u_c[i] * v_c[k] * (sum_j U_c[i,j] * V_c[j,k] ... )
        
        # Standard tensor network: C[i,k] = sum_c (sum_j U[i,j,c] * A[i,j]) * (sum_j V[j,k,c] * B[j,k]) ... no
        
        # Let me use the standard formulation from the schoolbook:
        # C[i,k] = sum_{j} A[i,j] * B[j,k]
        # Schoolbook: for each (i,j,k), one term: U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1
        # Then C[i,k] = sum_c W[k*n+i, c] * (sum_{j'} U[i*m+j', c] * A[i,j']) * (sum_{k'} V[j'*p+k', c] * B[j',k'])
        # Wait, that's not right either.
        
        # The correct interpretation: U, V, W define:
        # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
        # No, that's a product of two sums, which is quadratic in the wrong way.
        
        # Actually, the standard is:
        # C[i,k] = sum_c U[i, c] * V[k, c] * W[c] where U,V,W are derived from A,B
        # More precisely: define intermediate vectors:
        # x_c = sum_j U[i*m+j, c] * A[i,j]  (this depends on i and c)
        # y_c = sum_j V[j*p+k, c] * B[j,k]  (this depends on k and c)
        # Then C[i,k] = sum_c W[k*n+i, c] * x_c * y_c? No, that's cubic.
        
        # I think the correct formulation is that the tensor T has entries T[i,j,k] and
        # the rank-1 decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # where the indices are coupled through j and k.
        
        # From the schoolbook code: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # The product is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Summing over c: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This should equal the tensor entry. For matrix mult, the tensor is:
        # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k], but actually
        # the tensor for the bilinear map is T[i,j,k] where the output index is (i,k)
        # and input indices are (i,j) and (j,k).
        
        # I think the tensor is: T[i,k,j] = 1 (the middle index j is shared)
        # And the decomposition is: T[i,k,j] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # So for each (i,j,k), we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # And for "invalid" combinations... actually every (i,j,k) triple is valid.
        
        # So the target is: for all i in [0,n), j in [0,m), k in [0,p):
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # This is a system of n*m*p equations. Let's set up the target.
        
        T_target = np.ones((n, m, p))
        
        # Try alternating minimization
        max_iters = 200
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each (i,j,c), U[i*m+j,c] = sum over k of (T_target[i,j,k] * V[j*p+k,c] * W[k*n+i,c]) / (sum over k of (V[j*p+k,c] * W[k*n+i,c])**2 + eps)
            # This is solving a linear system for each fixed (i,j)
            
            for i in range(n):
                for j in range(m):
                    # For fixed i,j: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = T_target[i,j,k] for all k
                    # This is: U[i*m+j,:] * (V[j*p+0,:] * W[0*n+i,:], ..., V[j*p+(p-1),:] * W[(p-1)*n+i,:])^T = T_target[i,j,:]
                    # So U[i*m+j,:] * M = T_target[i,j,:] where M[k,c] = V[j*p+k,c] * W[k*n+i,c]
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve U[i*m+j,:] * M = T_target[i,j,:]
                    # U[i*m+j,:] = T_target[i,j,:] * pinv(M)
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M.T, T_target[i,j,:], rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k,c), V[j*p+k,c] = sum over i of (T_target[i,j,k] * U[i*m+j,c] * W[k*n+i,c]) / (sum over i of (U[i*m+j,c] * W[k*n+i,c])**2 + eps)
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(M.T, T_target[:,j,k], rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i,c), W[k*n+i,c] = sum over j of (T_target[i,j,k] * U[i*m+j,c] * V[j*p+k,c]) / (sum over j of (U[i*m+j,c] * V[j*p+k,c])**2 + eps)
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(M.T, T_target[i,:,k], rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            if it % 20 == 0:
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            error += (val - 1.0) ** 2
                if error < 1e-10:
                    # Found a valid decomposition!
                    best_rank = r
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
            if time.time() > deadline - 5:
                break
        
        if best_rank == r:
            break
    
    # Convert to lists
    def to_list(arr):
        return arr.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
