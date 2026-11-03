"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Schoolbook rank is n*m*p = 60
    # We'll try to find a lower rank decomposition using randomized search
    # with verification against the matrix multiplication tensor.
    
    # The matrix multiplication tensor T has components T[i,j,k] where
    # (i,j) indexes A (n x m), j indexes B's rows / A's cols, k indexes B's cols
    # Actually, let's be precise about the tensor structure.
    
    # For multiplying A (n x m) by B (m x p) to get C (n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The bilinear form: given vectors a (flattened A, size n*m) and b (flattened B, size m*p),
    # c = T(a,b) where c[i,k] = sum_j a[i*m+j] * b[j*p+k]
    #
    # A rank-r decomposition gives:
    # c = sum_{t=1}^{r} (u_t . a) * (v_t . b) * w_t
    # where u_t is in R^{n*m}, v_t is in R^{m*p}, w_t is in R^{n*p}
    #
    # In matrix form with U (n*m x r), V (m*p x r), W (n*p x r):
    # C = (U^T * a) * diag * (W^T * b) ... no, let me think again.
    #
    # Standard: C[i,k] = sum_t U[i*m+j... wait.
    #
    # Let me use the convention from the problem:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Note W is (p*n, rank) not (n*p, rank) - trace convention W[k*n+i, r]
    #
    # So W is indexed by (k, i) where k is column index (0..p-1) and i is row index (0..n-1)
    # W[k*n + i, t]
    #
    # The decomposition: C[i,k] = sum_t U[i*m + j... no, U is indexed by (i,j) flattened
    # U[i*m + j, t] * V[j*p + k, t] * W[k*n + i, t] summed over j and t?
    #
    # Wait, that's not right either. Let me re-read.
    #
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "exactly decompose the matrix multiplication tensor"
    #
    # The tensor for matrix mult: T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # 
    # A rank-r decomposition of this tensor means:
    # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    #
    # Wait, but that would mean for each (i,j,k), the sum over t of the product equals 1 if it's a valid term, 0 otherwise.
    # But actually in matrix multiplication, every (i,j,k) with j being the shared index is "active".
    # 
    # Hmm, I think the tensor is: for the bilinear map (A,B) -> C, the structure constants are:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    #
    # So the "tensor" has nonzero entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # And a rank-r decomposition expresses this as:
    # 1 = sum_{t=1}^r U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] for all i,j,k
    #
    # Wait no, that's not how tensor decomposition works for bilinear maps.
    
    # Let me reconsider. The matrix multiplication tensor T is a 3-tensor with T[i,j,k] = delta_{j,j} (always 1 for valid indices) 
    # but really it's defined such that the bilinear form is:
    # <T, a \otimes b> = sum_{i,j,k} T[i,j,k] * a[i,j] * b[j,k]
    # 
    # For matrix mult, T[i,j,k] = 1 for all i,j,k (since every combination contributes).
    # 
    # A rank-r decomposition: T = sum_t u_t \otimes v_t \otimes w_t
    # where u_t[i,j], v_t[j,k], w_t[i,k]
    # 
    # So: 1 = sum_t u_t[i,j] * v_t[j,k] * w_t[i,k] for all i,j,k
    #
    # In flattened form:
    # U[i*m+j, t] = u_t[i,j]
    # V[j*p+k, t] = v_t[j,k]  
    # W[k*n+i, t] = w_t[i,k]  (note the trace convention: W is indexed by k*n+i)
    
    # So we need: sum_{t=1}^r U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,j,k
    
    # This is a system of n*m*p = 60 equations in 3*r variables (well, with structure).
    
    # Strategy: Use the known rank-5 decomposition for 2x2 matrix multiplication
    # and combine it with other techniques.
    
    # Actually, let's try a direct approach: use randomized initialization and 
    # iterative refinement to minimize the error.
    
    # First, let's try some known constructions.
    
    # For 3x4 * 4x5, one approach is to split into blocks.
    # 3x4 * 4x5 can be seen as combining 2x2, 2x1, 1x2, 1x1 multiplications.
    
    # Let's try: split the 3x4 into [2x4; 1x4] and 4x5 into [4x3 4x2]
    # Then C = [2x4; 1x4] * [4x3 4x2] = [2x3 2x2; 1x3 1x2]
    
    # The 2x3 multiplication can be done with rank... 2x3 matrix mult has rank at most 2*3=6 schoolbook,
    # but can it be lower? For 2x3, the tensor rank is 5 (I believe).
    # 1x3 has rank 3 (just 3 scalar multiplications)
    # 2x2 has rank 5 (Strassen)
    # 1x2 has rank 2
    
    # So total: 5 + 3 + 5 + 2 = 15? That seems too optimistic. Let me think more carefully.
    
    # Actually, the block structure doesn't directly give a decomposition because the 
    # intermediate dimensions don't align simply.
    
    # Let me try a different approach: use the fact that we can write the problem as
    # a sum of smaller matrix multiplications.
    
    # Alternative: Just do a numerical search. Initialize U, V, W randomly and 
    # use gradient descent or alternating minimization to satisfy the constraints.
    
    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition is exact."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(len(U[0])):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def compute_error(U, V, W):
        """Compute the maximum error of the decomposition."""
        n, m, p = DIMENSIONS
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(len(U[0])):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    # Try a few different ranks starting from low
    # Theoretical lower bound: for n x m * m x p, rank >= max(n*m, m*p, n*p) / something
    # Actually, the trivial lower bound is max(n*m, m*p, n*p) is not right.
    # The rank must be at least the "slice rank" or "subrank" lower bound.
    # For 3x4*4x5: n*m=12, m*p=20, n*p=15
    # A simple lower bound is max(n*m, m*p, n*p) / min(n,m,p) ... no that's not standard.
    # 
    # Let's just try ranks from 20 upward (well below 60) and see if we can find a solution.
    
    import random
    random.seed(42)
    np.random.seed(42)
    
    best_result = None
    best_rank = 60  # schoolbook
    
    # First, let's get the schoolbook solution as a fallback
    U_school = [[0]*60 for _ in range(n*m)]
    V_school = [[0]*60 for _ in range(m*p)]
    W_school = [[0]*60 for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    # Try to find lower rank solutions
    # Use a simple approach: for each candidate rank r, initialize randomly and try to refine
    
    def try_rank(r, max_iters=2000):
        """Try to find a rank-r decomposition."""
        n, m, p = DIMENSIONS
        # Initialize with small random values
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        # Normalize columns to avoid scale issues
        for t in range(r):
            norm_u = np.linalg.norm(U[:, t])
            norm_v = np.linalg.norm(V[:, t])
            norm_w = np.linalg.norm(W[:, t])
            if norm_u < 1e-10 or norm_v < 1e-10 or norm_w < 1e-10:
                continue
            # We want U[:,t] * V[:,t] * W[:,t] to sum to 1
            # Scale so that product of norms is reasonable
            scale = 1.0 / (norm_u * norm_v * norm_w) ** (1.0/3.0)
            U[:, t] *= scale
            V[:, t] *= scale
            W[:, t] *= scale
        
        # Alternating minimization: fix two, solve for the third
        # This is non-convex, so we'll do several random restarts
        
        for restart in range(10):
            if time.time() > deadline - 10:
                break
            
            # Reinitialize
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Normalize
            for t in range(r):
                nu = np.linalg.norm(U[:, t])
                nv = np.linalg.norm(V[:, t])
                nw = np.linalg.norm(W[:, t])
                if nu < 1e-10 or nv < 1e-10 or nw < 1e-10:
                    continue
                scale = 1.0 / (nu * nv * nw) ** (1.0/3.0)
                U[:, t] *= scale
                V[:, t] *= scale
                W[:, t] *= scale
            
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                # Compute current error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for t in range(r):
                                s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                            e = s - 1.0
                            err += e * e
                
                if err < 1e-12:
                    # Convert to lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    if verify(U_list, V_list, W_list, 1e-6):
                        return U_list, V_list, W_list
                    else:
                        return None
                
                # Update U: for each (i,j), solve for U[i*m+j, :]
                # sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1 for all k
                # This is a linear system in U[i*m+j, :]
                for i in range(n):
                    for j in range(m):
                        # Build system: for each k, sum_t U[i*m+j,t] * (V[j*p+k,t] * W[k*n+i,t]) = 1
                        A_mat = np.zeros((p, r))
                        for k in range(p):
                            for t in range(r):
                                A_mat[k, t] = V[j*p+k, t] * W[k*n+i, t]
                        b_vec = np.ones(p)
                        # Solve least squares
                        try:
                            sol, residuals, rank_s, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                            U[i*m+j, :] = sol
                        except:
                            pass
                
                # Update V: for each (j,k), solve for V[j*p+k, :]
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((n, r))
                        for i in range(n):
                            for t in range(r):
                                A_mat[i, t] = U[i*m+j, t] * W[k*n+i, t]
                        b_vec = np.ones(n)
                        try:
                            sol, residuals, rank_s, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                            V[j*p+k, :] = sol
                        except:
                            pass
                
                # Update W: for each (k,i), solve for W[k*n+i, :]
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((m, r))
                        for j in range(m):
                            for t in range(r):
                                A_mat[j, t] = U[i*m+j, t] * V[j*p+k, t]
                        b_vec = np.ones(m)
                        try:
                            sol, residuals, rank_s, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                            W[k*n+i, :] = sol
                        except:
                            pass
                
                # Check convergence
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for t in range(r):
                                s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                            e = s - 1.0
                            err += e * e
                
                if err < 1e-10:
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    if verify(U_list, V_list, W_list, 1e-6):
                        return U_list, V_list, W_list
        
        return None
    
    # Try ranks from low to high
    # Lower bound considerations:
    # For 3x4*4x5, let's try from rank 20 up
    for r in range(20, 61):
        if time.time() > deadline - 20:
            break
        
        result = try_rank(r, max_iters=500)
        if result is not None:
            U_list, V_list, W_list = result
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # If we couldn't find a lower rank, return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
