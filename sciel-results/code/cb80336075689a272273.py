"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Strategy:
    1. Try a structured block-decomposition: partition into 2x2x2 blocks and use Strassen (rank 5 per block).
       4x4x5 can be partitioned into four 2x2x2 sub-tensors (since 5 = 2+2+1, we need to be careful).
       Actually, 4x4x5: we can split the last dimension into 2+2+1, but that's uneven.
       
       Better approach: Use known results. The tensor for multiplying an (n x m) by (m x p) matrix 
       has rank at most n*m*p (schoolbook). For 4x4x5, schoolbook is 80.
       
       A better approach is to use bilinear algorithms. Let me think about this more carefully.
       
       The matrix multiplication tensor T_{i,j,k} corresponds to C[i,k] += A[i,j] * B[j,k].
       For a rank-r decomposition, we need:
       T = sum_{c=1}^{r} U_c ⊗ V_c ⊗ W_c
       where U_c is a vector in R^{n*m}, V_c in R^{m*p}, W_c in R^{p*n} (with W indexed as [k*n+i, r]).
       
       Actually, let me re-read the convention. W[k*n+i, r] means the output index is k*n + i, 
       so for output C[i,k], the index is k*n + i.
       
       Let me try a direct approach: use a known low-rank algorithm for small matrices.
       
       For 2x2x2, Strassen gives rank 7 (not 5; rank 5 is not known for 2x2x2, the best known is 7).
       Wait, the problem says "rank-5 Strassen variant" - that's not standard. Strassen is rank 7.
       
       Let me just implement a general search using tensor decomposition techniques.
       
       Actually, given the time constraints, let me try a few known approaches:
       
       1. Schoolbook: rank 80
       2. Try to find a better decomposition using numerical optimization
       
       Let me implement a simple iterative refinement approach starting from a random or structured initialization.
    """
    deadline = time.time() + 140  # Leave 10s margin for evaluation overhead
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Try to improve using ALS (Alternating Least Squares) for tensor decomposition
    # The matrix multiplication tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1
    # We want to find U, V, W such that T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    
    # Let's try a rank-r decomposition for various r
    # Start with r = 20 (as suggested by the block decomposition idea)
    
    def verify(U, V, W, tol=1e-8):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        # Build the target tensor
        # T[i*m+j, j*p+k, k*n+i] = 1
        # Check: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1
        # And for other combinations, it should be 0
        
        # More efficient: build the tensor and check
        # T is a 3-way tensor of shape (n*m, m*p, p*n)
        # But we can check by building the matrix multiplication
        
        # Actually, the condition is:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # With the decomposition, C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * A[i,j]) * (sum_k V[j*p+k,c] * B[j,k])
        
        # Let's verify by checking the tensor directly
        # For the decomposition to be correct:
        # sum_c U[a,c] * V[b,c] * W[c_,c] = T[a,b,c_]
        # where a = i*m+j, b = j*p+k, c_ = k*n+i
        
        # The target tensor T[a,b,c_] is 1 if a, b, c_ are consistent (i.e., there exist i,j,k such that 
        # a=i*m+j, b=j*p+k, c_=k*n+i) and 0 otherwise.
        
        # Let's build the target tensor
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c_ = k * n + i
                    T[a, b, c_] = 1
        
        # Compute the decomposed tensor
        # T_decomp[a,b,c_] = sum_c U[a,c] * V[b,c] * W[c_,c]
        # This is a 3-way tensor contraction
        T_decomp = np.einsum('ac,bc,dc->abd', U, V, W)
        
        # Check
        return np.allclose(T_decomp, T, atol=tol)
    
    def try_rank(r, max_iters=50, tol=1e-8, seed=42):
        """Try to find a rank-r decomposition using ALS."""
        if time.time() > deadline - 5:
            return None
        
        np.random.seed(seed)
        n, m, p = DIMENSIONS
        
        # Initialize
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Build target tensor
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c_ = k * n + i
                    T[a, b, c_] = 1
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each a, U[a,:] should satisfy sum_b,c_ T[a,b,c_] = sum_c U[a,c] * (sum_b V[b,c] * W[c_,c])
            # This is not quite right. Let me think again.
            
            # The tensor equation is: T[a,b,c_] = sum_c U[a,c] * V[b,c] * W[c_,c]
            # For fixed a: T[a,b,c_] = sum_c U[a,c] * (V[b,c] * W[c_,c])
            # Let M_U[a] be the matrix with entries M_U[a][b, c_] = V[b,c] * W[c_,c] summed over c
            # Actually, for fixed a, we have a matrix T[a,:,:] of shape (m*p, p*n)
            # and we want to express it as sum_c U[a,c] * (V[:,c] ⊗ W[:,c])
            # where V[:,c] ⊗ W[:,c] is the outer product of V[:,c] and W[:,c]
            
            # So for each a, T[a,:,:] = sum_c U[a,c] * np.outer(V[:,c], W[:,c])
            # This is a linear system in U[a,:]
            
            # Build the design matrix: D[a] has columns D[a][:,c] = V[:,c] ⊗ W[:,c]
            # Then T[a,:,:].flatten() = D[a] @ U[a,:]
            
            # Let's do this for all a at once
            # D is a 4D array: D[a, b, c_, c] = V[b,c] * W[c_,c]
            # Shape: (n*m, m*p, p*n, r)
            
            D = np.einsum('bc,dc->bdc', V, W)  # Shape: (m*p, p*n, r)
            # Now T[a,b,c_] = sum_c D[b,c_,c] * U[a,c]
            # Reshape: T_flat[a, (b,c_)] = sum_c D_flat[(b,c_), c] * U[a, c]
            # So T_flat = D_flat @ U.T where T_flat has shape (n*m, m*p*p*n)
            
            T_flat = T.reshape(n*m, -1)  # Shape: (n*m, m*p*p*n)
            D_flat = D.reshape(m*p*p*n, r)  # Shape: (m*p*p*n, r)
            
            # Solve: T_flat = D_flat @ U.T
            # U.T = D_flat^+ @ T_flat
            # U = (D_flat^+ @ T_flat).T
            
            try:
                U_new = np.linalg.lstsq(D_flat, T_flat, rcond=None)[0].T
            except:
                break
            
            # Update V: similarly
            D_V = np.einsum('ac,dc->adc', U, W)  # Shape: (n*m, p*n, r)
            T_flat_V = T.transpose(1, 0, 2).reshape(m*p, -1)  # Shape: (m*p, n*m*p*n)
            D_flat_V = D_V.reshape(n*m*p*n, r)  # Shape: (n*m*p*n, r)
            
            try:
                V_new = np.linalg.lstsq(D_flat_V, T_flat_V, rcond=None)[0].T
            except:
                break
            
            # Update W
            D_W = np.einsum('ac,bc->abc', U, V)  # Shape: (n*m, m*p, r)
            T_flat_W = T.transpose(2, 0, 1).reshape(p*n, -1)  # Shape: (p*n, n*m*m*p)
            D_flat_W = D_W.reshape(n*m*m*p, r)  # Shape: (n*m*m*p, r)
            
            try:
                W_new = np.linalg.lstsq(D_flat_W, T_flat_W, rcond=None)[0].T
            except:
                break
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            T_decomp = np.einsum('ac,bc,dc->abd', U, V, W)
            err = np.linalg.norm(T_decomp - T)
            if err < tol:
                return U, V, W, r
        
        return None
    
    # Try various ranks
    for r in [20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80]:
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, max_iters=30, seed=42)
        if result is not None:
            U, V, W, rank = result
            if verify(U, V, W, tol=1e-8):
                best_U, best_V, best_W, best_rank = U, V, W, rank
                break  # Found a valid decomposition, stop
        
        # Also try with different seeds
        for seed in [1, 7, 13, 99, 123]:
            if time.time() > deadline - 10:
                break
            result = try_rank(r, max_iters=20, seed=seed)
            if result is not None:
                U, V, W, rank = result
                if verify(U, V, W, tol=1e-8):
                    best_U, best_V, best_W, best_rank = U, V, W, rank
                    break
        if best_rank < 80:
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
