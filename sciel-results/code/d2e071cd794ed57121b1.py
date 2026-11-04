"""Matrix multiplication 3,3,4 - rank-minimal tensor decomposition search."""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [3, 3, 4]

# Known rank bounds:
# For 3x3 by 3x4: rank is between 17 and 36 (schoolbook)
# Strassen-like reductions exist. Let's search for rank < 36.

def _build_tensor(n, m, p):
    """Build the multiplication tensor T[i,j,k] = 1 if this corresponds to C[i,k] = sum_j A[i,j]*B[j,k]
    Actually the tensor for matrix multiplication M_{n,m,p} has entries:
    T[i, j, k] = delta_{i, k} ... no.
    
    The multiplication tensor T has T[i, j, k] = 1 if the (i,k)-th entry of C = A*B
    receives contribution from A[i,j]*B[j,k].
    
    In the factorization: C[i,k] = sum_c U[i, c] * V[j... 
    Wait, let me be careful with indices.
    
    We have A (n x m), B (m x p), C (n x p).
    C[i, k] = sum_{j=0}^{m-1} A[i, j] * B[j, k]
    
    The tensor T is a 3-tensor with indices (i, j, k) where:
    T[i, j, k] = 1 if this term appears in the sum for C[i, k]
    i.e., T[i, j, k] = 1 for all valid i, j, k (since every j contributes to C[i,k] via A[i,j]*B[j,k])
    
    Wait, that's not right either. The tensor represents the bilinear form:
    <T, a, b> = sum_{i,j,k} T[i,j,k] * a[i,j] * b[j,k] = sum_{i,k} C[i,k]
    
    Actually, the standard formulation: the multiplication tensor T_{n,m,p} is defined by
    T[i, j, k] = 1 if the product A[i,j] * B[j,k] contributes to C[i,k].
    Since C[i,k] = sum_j A[i,j] B[j,k], every (i,j,k) triple with valid indices has T[i,j,k]=1.
    
    But that gives rank n*m*p which is the schoolbook. The tensor is:
    T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k  (in the appropriate spaces)
    
    Hmm, let me reconsider. The tensor for matrix multiplication M_{n,m,p}:
    - First index i in [0, n) corresponds to row of A / row of C
    - Second index j in [0, m) corresponds to column of A / row of B  
    - Third index k in [0, p) corresponds to column of B / column of C
    
    T[i, j, k] = 1 for all i, j, k (since every A[i,j]*B[j,k] contributes to C[i,k])
    
    Wait no. Let me think about this differently.
    
    The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    
    As a tensor T in U* ⊗ V* ⊗ W* where:
    - U has dimension n*m (flattened A)
    - V has dimension m*p (flattened B)  
    - W has dimension n*p (flattened C)
    
    T[u, v, w] = 1 if the term corresponding to index u in A and index v in B
    contributes to the term corresponding to index w in C.
    
    u = i*m + j (A[i,j])
    v = j*p + k (B[j,k])
    w = k*n + i (C[i,k] using the trace convention W[k*n+i, r])
    
    T[u, v, w] = 1 if u = i*m+j, v = j'*p+k, w = k'*n+i' with i=i', j=j', k=k'
    i.e., T[i*m+j, j*p+k, k*n+i] = 1
    
    So T has exactly n*m*p ones, one for each (i,j,k) triple.
    """
    n, m, p = 3, 3, 4
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _check_decomposition(U, V, W, T, tol=1e-10):
    """Check if U @ diag @ V^T @ diag @ W^T reconstructs T.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    T = sum_c U[:,c] * V[:,c]^T * W[:,c]^T ... 
    
    Actually: T[i, j, k] = sum_c U[i, c] * V[j, c] * W[k, c]
    
    Using numpy: T = np.einsum('ic,jc,kc->ijk', U, V, W)
    """
    T_recon = np.einsum('ic,jc,kc->ijk', U, V, W, optimize=True)
    return np.allclose(T_recon, T, atol=tol)

def _check_decomposition_fast(U, V, W, T, tol=1e-10):
    """Faster check using einsum."""
    T_recon = np.einsum('ic,jc,kc->ijk', U, V, W, optimize=True)
    diff = T_recon - T
    return np.max(np.abs(diff)) < tol

def _try_rank_with_search(n, m, p, target_rank, max_iters=5000, seed=42):
    """Try to find a decomposition of given rank using random initialization + gradient descent.
    Returns (U, V, W) or None."""
    T = _build_tensor(n, m, p)
    rng = np.random.RandomState(seed)
    
    best_loss = np.inf
    best_U = best_V = best_W = None
    
    for attempt in range(20):  # multiple random restarts
        # Initialize with random small values
        U = rng.randn(n*m, target_rank) * 0.1
        V = rng.randn(m*p, target_rank) * 0.1
        W = rng.randn(p*n, target_rank) * 0.1
        
        # Scale to roughly match the tensor magnitude
        # Each term should be ~1/36 on average if uniform
        scale = (n*m*p / target_rank) ** (1.0/3.0) * 0.5
        U *= scale
        V *= scale
        W *= scale
        
        # Simple gradient descent
        lr = 0.01
        for it in range(max_iters):
            # Compute gradient
            T_recon = np.einsum('ic,jc,kc->ijk', U, V, W, optimize=True)
            error = T_recon - T
            
            # Gradient: dL/dU[i,c] = sum_{j,k} error[i,j,k] * V[j,c] * W[k,c]
            # = einsum('ijk,jc,kc->ic', error, V, W)
            dU = np.einsum('ijk,jc,kc->ic', error, V, W, optimize=True)
            dV = np.einsum('ijk,ic,kc->jc', error, U, W, optimize=True)
            dW = np.einsum('ijk,ic,jc->kc', error, U, V, optimize=True)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Periodic check
            if it % 200 == 0:
                loss = np.sum(error**2)
                if loss < best_loss:
                    best_loss = loss
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if loss < tol * tol * (n*m*p):
                    break
        
        loss = np.sum(error**2)
        if loss < best_loss:
            best_loss = loss
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if best_loss < tol * tol * (n*m*p):
            break
    
    if best_loss < tol * tol * (n*m*p):
        return best_U, best_V, best_W
    return None

def _try_rank_with_structural(n, m, p, target_rank, max_time=30):
    """Try to find decomposition using structured approach.
    
    For 3x3 by 3x4, we can try to exploit the structure.
    The tensor T[i*m+j, j*p+k, k*n+i] = 1
    
    Let's try a different approach: use the fact that for square matrix mult,
    Strassen gives rank 7 for 2x2. For 3x3, best known is 19-23.
    For 3x3 by 3x4, it's somewhere between 17 and 36.
    
    Let's try a column-pruning approach starting from schoolbook.
    """
    start_time = time.time()
    T = _build_tensor(n, m, p)
    
    # Start with schoolbook (rank 36)
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    # Try to merge/prune columns
    # A column c can be removed if its contribution can be represented by other columns
    # i.e., U[:,c] is in span of other U columns, similarly for V and W
    
    # Better approach: try to find linear dependencies
    # If sum of some columns = 0 in the tensor sense, we can reduce rank
    
    # Alternative: use SVD-like approach on the tensor
    # Reshape T into matrices and find low-rank approximations
    
    # Let's try: for each pair of "slices", compute the matrix and find its rank
    # T as a matrix: for fixed i, T[i, :, :] is a (m*p) x (p*n) matrix
    # The rank of the tensor is at least the max rank of any slice
    
    # Actually, let's try a more direct approach:
    # Use the fact that T[i*m+j, j*p+k, k*n+i] = 1
    # Group by j (the middle index):
    # For fixed j, we have a matrix M_j[i, k] = 1 for all i, k
    # This is an n x p all-ones matrix, which has rank 1!
    
    # So T = sum_{j=0}^{m-1} (e_j in U-space) ⊗ (e_j in V-space) ⊗ (J in W-space)
    # where J is the n x p all-ones matrix
    
    # Wait, that's not quite right. Let me re-examine.
    
    # T[u, v, w] where u = i*m+j, v = j'*p+k, w = k'*n+i'
    # T[u, v, w] = 1 iff u's j-index = v's j-index AND u's i-index = w's i-index AND v's k-index = w's k-index
    # i.e., T[i*m+j, j*p+k, k*n+i] = 1
    
    # So for the tensor to be nonzero at (u, v, w), we need:
    # j_u = j_v (the middle index matches)
    # i_u = i_w (the first index of A matches first index of C)
    # k_v = k_w (the second index of B matches second index of C)
    
    # This means T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    
    # We can try to factor this differently. Note that:
    # sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    
    # = sum_j (sum_i e_{i*m+j}) ⊗ (sum_k e_{j*p+k}) ⊗ (sum_{i,k} e_{k*n+i}) ... no that's wrong
    
    # Let me think about this as a matrix multiplication problem.
    # The tensor T can be viewed as a matrix M of size (n*m) x (m*p) for each "slice" of the third index.
    # Or as a 3-way array.
    
    # Key insight: T[i*m+j, j*p+k, k*n+i] = 1
    # This is equivalent to saying that the "flattened" matrix A (n*m x m*p) where
    # A[i*m+j, j*p+k] = 1 if the j-indexes match (j_u = j_v), else 0
    # is a block diagonal matrix with n blocks, each being an m x (m*p/m) = m x p matrix of ones? No...
    
    # Actually A[i*m+j, j'*p+k] = 1 iff j = j'. So A is block diagonal with m blocks,
    # each block being n x p all-ones.
    
    # Similarly for other flattening.
    
    # Let's try a different strategy: use known results.
    # For 3x3 matrix multiplication, best known rank is 19 (Bläser 2005, improved to 19).
    # Actually, I recall the best known for 3x3 is 19 or 20.
    # For 3x3 by 3x4, it should be less than 3x3 by 3x3 rank + something.
    
    # Let me just try random search with various ranks.
    
    # Try ranks from low to high
    for target_rank in range(17, 37):
        if time.time() - start_time > max_time:
            break
        
        result = _try_rank_with_search(n, m, p, target_rank, max_iters=2000, seed=target_rank*1000)
        if result is not None:
            U, V, W = result
            # Verify
            if _check_decomposition_fast(U, V, W, T, tol=1e-8):
                return U, V, W, target_rank
    
    return None

def _try_rank_with_svd_approach(n, m, p, target_rank, max_time=30):
    """Use SVD-based approach to find low-rank decomposition.
    
    The idea: fix two of the matrices and solve for the third, iterate.
    """
    start_time = time.time()
    T = _build_tensor(n, m, p)
    rng = np.random.RandomState(12345)
    
    best_result = None
    best_rank = target_rank + 1
    
    for attempt in range(5):
        U = rng.randn(n*m, target_rank) * 0.5
        V = rng.randn(m*p, target_rank) * 0.5
        W = rng.randn(p*n, target_rank) * 0.5
        
        for it in range(100):
            if time.time() - start_time > max_time:
                break
            
            # Fix U and V, solve for W
            # T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
            # For each k: sum_c (U V^T)_{i,j}^{(c)} ... 
            # Actually: T[:, :, k] = sum_c (U[:,c] * V[:,c]^T) * W[k,c]
            # Let M_c = U[:,c] * V[:,c]^T (outer product, size n*m x m*p)
            # Then T[:,:,k] = sum_c M_c * W[k,c]
            # So [T[:,:,0], T[:,:,1], ..., T[:,:,p*n-1]] = [M_0, M_1, ..., M_{rank-1}] @ W^T
            # Wait, W is (p*n, rank), so W^T is (rank, p*n)
            # T[:,:,k] is a matrix of size (n*m) x (m*p)
            # T[:,:,k] = sum_c W[k,c] * (U[:,c] @ V[:,c]^T)
            
            # Stack: Let A_mat be (rank, n*m, m*p) where A_mat[c] = U[:,c] @ V[:,c]^T
            # Then for each k: T[:,:,k] = sum_c W[k,c] * A_mat[c]
            # This is a linear system in W.
            
            # Flatten: for each (i,j) pair in T[:,:,k], we have:
            # T[i,j,k] = sum_c W[k,c] * A_mat[c, i, j]
            
            # Build the system: for each k, we have (n*m)*(m*p) equations... that's too many.
            # Instead, use least squares.
            
            # Reshape: T is (n*m, m*p, p*n)
            # We want: T = sum_c W[k,c] * (U[:,c] @ V[:,c]^T)
            # = sum_c W[:,c] (as column) outer (U[:,c] @ V[:,c]^T)
            
            # Let's define: for the W update, fix U and V.
            # T[i, j, k] = sum_c U[i,c] * V[j,c] * W[k,c]
            # For fixed (i,j): T[i,j,:] = (U[i,:] * V[j,:]) @ W^T
            # where U[i,:]*V[j,:] is element-wise product, a vector of length rank
            # and W^T is (rank, p*n)
            
            # So T[i,j,:] = diag(U[i,:]*V[j,:]) @ W^T ... no
            # T[i,j,:] = (U[i,:] * V[j,:]) @ W^T
            # where (U[i,:]*V[j,:]) is a 1D array of length rank, and W^T is (rank, p*n)
            
            # Stack all (i,j): T_flat has shape ((n*m)*(m*p), p*n)
            # B_mat has shape ((n*m)*(m*p), rank) where B_mat[i*m+j, c] = U[i,c]*V[j,c]
            # Then T_flat = B_mat @ W^T
            # So W^T = pinv(B_mat) @ T_flat
            # W = (pinv(B_mat) @ T_flat)^T
            
            B_mat = np.einsum('ic,jc->ijc', U, V).reshape(n*m*m*p, target_rank)
            T_flat = T.reshape(n*m*m*p, p*n)
            
            # Solve W^T = pinv(B_mat) @ T_flat
            W_new, _, _, _ = np.linalg.lstsq(B_mat, T_flat, rcond=None)
            W = W_new.T
            
            # Similarly for V update, fix U and W
            # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
            # For fixed (i,k): T[i,:,k] = (U[i,:] * W[k,:]) @ V^T
            # T_ik_flat has shape ((n)*(p*n), m*p) ... let me redo
            # For fixed i and k: T[i, j, k] for all j
            # T[i, j, k] = sum_c U[i,c] * V[j,c] * W[k,c]
            # = sum_c (U[i,c] * W[k,c]) * V[j,c]
            # For fixed (i,k): the vector over j is: V @ (U[i,:]*W[k,:])^T
            # T[i, :, k] (shape m*p) = V (shape m*p x rank) @ (U[i,:]*W[k,:]) (shape rank)
            
            # Stack over (i,k): 
            # T_ik has shape (n, m*p, p*n) -> reshape to (n*p*n, m*p)
            # C_mat has shape (n*p*n, rank) where C_mat[i*p*n+k, c] = U[i,c]*W[k,c]
            # T_ik_flat = C_mat @ V^T
            # V^T = pinv(C_mat) @ T_ik_flat
            T_ik = T.reshape(n, m*p, p*n).transpose(0, 2, 1).reshape(n*p*n, m*p)
            C_mat = np.einsum('ic,kc->ikc', U, W).reshape(n*p*n, target_rank)
            V_new, _, _, _ = np.linalg.lstsq(C_mat, T_ik, rcond=None)
            V = V_new.T
            
            # Similarly for U update, fix V and W
            # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
            # For fixed (j,k): T[:, j, k] (shape n*m... wait, i ranges over n, but first index is i*m+j)
            # Hmm, the first index is i*m+j, so for fixed j, the first index ranges over i*m+j for i in range(n)
            # That's not a contiguous block unless we reindex.
            
            # Let me reindex. First index a = i*m + j, so i = a // m, j = a % m
            # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
            # T[a, b, c] = 1 iff j = j', i = i', k = k'
            
            # For U update, fix V and W:
            # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
            # For fixed (b, c): T[:, b, c] (shape n*m) = U (shape n*m x rank) @ (V[b,:]*W[c,:]) (shape rank)
            # T_bc has shape ((n*m), m*p, p*n) -> for fixed (b,c): T[:, b, c] is shape (n*m)
            # Stack: T_bc_flat has shape (m*p*p*n, n*m)
            # D_mat has shape (m*p*p*n, rank) where D_mat[b*p*n+c, r] = V[b,r]*W[c,r]
            # T_bc_flat = D_mat @ U^T
            # U^T = pinv(D_mat) @ T_bc_flat
            T_bc = T.reshape(n*m, m*p, p*n).transpose(1, 2, 0).reshape(m*p*p*n, n*m)
            D_mat = np.einsum('br,cr->bcr', V, W).reshape(m*p*p*n, target_rank)
            U_new, _, _, _ = np.linalg.lstsq(D_mat, T_bc, rcond=None)
            U = U_new.T
            
            # Check convergence
            T_recon = np.einsum('ic,jc,kc->ijk', U, V, W, optimize=True)
            loss = np.sum((T_recon - T)**2)
            
            if loss < 1e-12 * (n*m*p):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = (U.copy(), V.copy(), W.copy())
                break
        
        if best_result is not None and target_rank == best_rank:
            break
    
    if best_result is not None:
        return best_result + (best_rank,)
    return None

def construct():
    """Find rank-minimal exact tensor decomposition for 3x3 by 3x4 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit, leave some margin
    
    n, m, p = 3, 3, 4
    T = _build_tensor(n, m, p)
    
    # Start with schoolbook as fallback
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to find lower rank decompositions
    # Use the iterative least squares approach
    for target_rank in range(17, 37):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_with_svd_approach(n, m, p, target_rank, max_time=30)
        if result is not None:
            U_r, V_r, W_r, r = result
            # Verify
            if _check_decomposition_fast(U_r, V_r, W_r, T, tol=1e-8):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = r
                break  # Found a valid decomposition, stop
    
    # Try gradient descent for remaining ranks if time permits
    if best_rank == 36 and time.time() < deadline - 15:
        for target_rank in range(20, 37):
            if time.time() > deadline - 5:
                break
            result = _try_rank_with_search(n, m, p, target_rank, max_iters=3000, seed=target_rank*777)
            if result is not None:
                U_r, V_r, W_r = result
                if _check_decomposition_fast(U_r, V_r, W_r, T, tol=1e-8):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = target_rank
                    break
    
    # Convert to lists for the interface
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
