"""Find rank-minimal tensor decomposition for 3x4 * 4x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    """Construct a tensor decomposition for matrix multiplication.
    
    Returns dict with 'U', 'V', 'W' as lists of lists.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    """
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Build the matrix multiplication tensor T[i,j,k] = 1 if (i,j) in U, (j,k) in V, (k,i) in W
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # This is a 3-way tensor of size (n*m) x (m*p) x (p*n)
    
    # The schoolbook decomposition has rank n*m*p = 3*4*6 = 72
    # We want to find a lower rank decomposition.
    
    # Strategy: Use a block-structured approach.
    # Partition the 3x4 matrix into blocks and the 4x6 matrix into blocks.
    
    # Let's think about this differently. The matrix multiplication tensor for (n,m,p)
    # has entries T[(i,j),(j,k),(k,i)] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # Actually, the tensor is: T[a,b,c] where a indexes (i,j) in [0,n*m), b indexes (j',k) in [0,m*p), c indexes (k',i') in [0,p*n)
    # T[a,b,c] = 1 if j=j' and k=k' and i=i', else 0
    # More precisely: if a = i*m+j, b = j'*p+k, c = k'*n+i', then T[a,b,c] = 1 iff j=j' and k=k' and i=i'
    
    # Let's use a rank minimization approach via randomized tensor decomposition search.
    # We'll use the fact that we can try to find a low-rank decomposition by iterative methods.
    
    # First, let's establish the schoolbook rank as baseline
    schoolbook_rank = n * m * p  # 72
    
    # Try to find a better decomposition using a structured approach.
    # 
    # Key insight: For matrix multiplication, there are known lower bounds and constructions.
    # For 3x4 * 4x6, we can try block-based approaches.
    #
    # Let's partition:
    # A is 3x4, B is 4x6, C = A*B is 3x6
    #
    # We can write A as [A1; A2] where A1 is 2x4 and A2 is 1x4
    # B as [B1 B2] where B1 is 4x3 and B2 is 4x3
    # Then C = A1*B1_full + A2*B2_full
    #
    # Actually, let's use a more systematic approach.
    # 
    # The tensor decomposition rank for matrix multiplication of (n,m,p) matrices
    # is at most n*m*p (schoolbook) and at least max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    #
    # Known results: 
    # - 2x2x2: rank 7 (Strassen)
    # - 3x3x3: rank at most 23 (Laderman), at least 19
    # - For rectangular cases, various bounds exist
    #
    # For 3x4x6, let's try a block decomposition:
    # Split the 4 intermediate dimension into 2+2
    # A (3x4) = [A_left (3x2) | A_right (3x2)]
    # B (4x6) = [B_top (2x6); B_bottom (2x6)]
    # C = A_left * B_top + A_right * B_bottom
    #
    # Each product is (3x2)*(2x6) -> 3x6
    # Rank for (3,2,6) multiplication: at most 3*2*6=36, but can be better
    #
    # Alternatively, split into 1+1+2:
    # This gets complex. Let me try a direct computational approach.
    
    # Let's use a randomized approach to find a low-rank decomposition.
    # We'll use the tensor and try to find rank-r decompositions.
    
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Try to find a low-rank decomposition using iterative methods
    # We'll use a simple approach: try various ranks and use optimization
    
    def try_decomposition(rank, max_iters=200, seed=42):
        """Try to find a rank-r decomposition using ALS-like approach."""
        if time.time() > deadline - 30:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(n*m, rank)
        V = rng.randn(m*p, rank)
        W = rng.randn(p*n, rank)
        
        # Scale to match the tensor magnitude
        scale = (np.sum(T**2)) ** 0.5 / (rank ** 0.5)
        U *= scale
        V *= scale
        W *= scale
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 30:
                break
            
            # Update U
            # For each (a,b), we need sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
            # Fix V and W, solve for U
            # U[a,:] = argmin sum_{b,c} (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
            # This is linear in U[a,:]
            # Let's form the system: for each a, 
            # sum_{b,c} (T[a,b,c] - U[a,:]^T * (V[b,:] * W[c,:]))^2
            # = sum_{b,c} T[a,b,c]^2 - 2*sum_{b,c} T[a,b,c]*U[a,:]^T*(V[b,:]*W[c,:]) + sum_{b,c} (U[a,:]^T*(V[b,:]*W[c,:]))^2
            # The last term: sum_{b,c} (sum_r U[a,r]*V[b,r]*W[c,r])^2
            # = sum_{r,s} U[a,r]*U[a,s] * (sum_{b,c} V[b,r]*W[c,r]*V[b,s]*W[c,s])
            # Let M[r,s] = sum_{b,c} V[b,r]*W[c,r]*V[b,s]*W[c,s]
            # = sum_r' V[r',r]*W[r',r] ... wait, let me reindex
            # M[r,s] = sum_{b,c} V[b,r]*W[c,r]*V[b,s]*W[c,s]
            # Note that b and c are independent in the sum (b ranges over m*p, c over p*n)
            # So M[r,s] = (sum_b V[b,r]*V[b,s]) * (sum_c W[c,r]*W[c,s])
            # Let G_V[r,s] = sum_b V[b,r]*V[b,s] = (V^T V)[r,s]
            # Let G_W[r,s] = sum_c W[c,r]*W[c,s] = (W^T W)[r,s]
            # Then M[r,s] = G_V[r,s] * G_W[r,s] (element-wise product)
            
            G_V = V.T @ V  # (rank, rank)
            G_W = W.T @ W  # (rank, rank)
            M = G_V * G_W  # element-wise product, (rank, rank)
            
            # The gradient for U[a,:]:
            # d/dU[a,r] sum_{b,c} (T[a,b,c] - sum_s U[a,s]*V[b,s]*W[c,s])^2
            # = -2 * sum_{b,c} T[a,b,c]*V[b,r]*W[c,r] + 2 * sum_s U[a,s] * sum_{b,c} V[b,s]*W[c,s]*V[b,r]*W[c,r]
            # = -2 * (T[a] * (V*W))[r] + 2 * (M @ U[a,:])[r]
            # where (T[a] * (V*W))[r] = sum_{b,c} T[a,b,c]*V[b,r]*W[c,r]
            
            # Let's compute S[a,r] = sum_{b,c} T[a,b,c]*V[b,r]*W[c,r]
            # T[a,b,c] is nonzero only when a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
            # So for fixed a=i*m+j, b must be j*p+k for some k, and c must be k*n+i
            # S[a,r] = sum_k T[i*m+j, j*p+k, k*n+i] * V[j*p+k, r] * W[k*n+i, r]
            # = sum_k V[j*p+k, r] * W[k*n+i, r]  (since T=1 for these entries)
            
            S = np.zeros((n*m, rank))
            for i in range(n):
                for j in range(m):
                    a = i * m + j
                    for k in range(p):
                        b = j * p + k
                        c = k * n + i
                        S[a, :] += V[b, :] * W[c, :]
            
            # Now solve for U[a,:]: M @ U[a,:] = S[a,:]
            # U[a,:] = M^{-1} @ S[a,:]
            
            try:
                U_new = np.linalg.solve(M, S.T).T
            except np.linalg.LinAlgError:
                U_new = np.linalg.lstsq(M, S.T, rcond=None)[0].T
            
            # Update V
            G_U = U_new.T @ U_new
            G_W = W.T @ W
            M_v = G_U * G_W
            
            S_v = np.zeros((m*p, rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        S_v[b, :] += U_new[a, :] * W[c, :]
            
            try:
                V_new = np.linalg.solve(M_v, S_v.T).T
            except np.linalg.LinAlgError:
                V_new = np.linalg.lstsq(M_v, S_v.T, rcond=None)[0].T
            
            # Update W
            G_U = U_new.T @ U_new
            G_V = V_new.T @ V_new
            M_w = G_U * G_V
            
            S_w = np.zeros((p*n, rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        S_w[c, :] += U_new[a, :] * V_new[b, :]
            
            try:
                W_new = np.linalg.solve(M_w, S_w.T).T
            except np.linalg.LinAlgError:
                W_new = np.linalg.lstsq(M_w, S_w.T, rcond=None)[0].T
            
            # Compute error
            # err = sum_{a,b,c} (T[a,b,c] - sum_r U_new[a,r]*V_new[b,r]*W_new[c,r])^2
            # Only need to check the nonzero entries of T
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.dot(U_new[a, :], V_new[b, :] * W_new[c, :])
                        err += (1.0 - val) ** 2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
            
            U, V, W = U_new, V_new, W_new
            
            if best_err < 1e-10:
                break
        
        if best_err < 1e-8:
            return best_U, best_V, best_W, best_err
        return None
    
    # Try different ranks starting from a reasonable lower bound
    # Lower bound: max(n*m, m*p, p*n) = 24, but realistically higher
    # Let's try from 48 up to schoolbook rank
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Start with a structured approach that might give better results
    # Try ranks in increasing order
    for rank in range(48, schoolbook_rank + 1):
        if time.time() > deadline - 40:
            break
        
        result = try_decomposition(rank, max_iters=100, seed=42)
        if result is not None:
            U_mat, V_mat, W_mat, err = result
            best_result = (U_mat, V_mat, W_mat)
            best_rank = rank
            print(f"Found rank {rank} decomposition with error {err:.2e}")
            break
        
        # Also try with different seeds
        result = try_decomposition(rank, max_iters=100, seed=123)
        if result is not None:
            U_mat, V_mat, W_mat, err = result
            best_result = (U_mat, V_mat, W_mat)
            best_rank = rank
            print(f"Found rank {rank} decomposition with error {err:.2e}")
            break
    
    if best_result is None:
        # Fall back to schoolbook
        U_mat = np.zeros((n*m, schoolbook_rank))
        V_mat = np.zeros((m*p, schoolbook_rank))
        W_mat = np.zeros((p*n, schoolbook_rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_mat[i*m+j, col] = 1
                    V_mat[j*p+k, col] = 1
                    W_mat[k*n+i, col] = 1
                    col += 1
    else:
        U_mat, V_mat, W_mat = best_result
    
    # Convert to lists
    U_list = U_mat.tolist()
    V_list = V_mat.tolist()
    W_list = W_mat.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
