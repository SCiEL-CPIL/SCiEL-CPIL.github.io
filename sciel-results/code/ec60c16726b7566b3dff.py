"""Matrix multiplication 3x4x6 - rank-minimal exact tensor decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 6]

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, tol=1e-6):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    # The matrix multiplication tensor T has T[i,j,k] = 1 if and only if
    # the (i,j) entry of A times (j,k) entry of B contributes to (i,k) of C.
    # Tensor decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, the convention is W[k*n+i, r], so W index is (k, i).
    # Let's reconstruct: for each r, the contribution is U[:, :, r] outer V[:, :, r] outer W[:, :, r]
    # where U is indexed by (i, j), V by (j, k), W by (k, i).
    
    # Reshape to 3D: U is (n*m, rank) -> (n, m, rank)
    # V is (m*p, rank) -> (m, p, rank)
    # W is (p*n, rank) -> (p, n, rank)
    
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    
    # Compute sum over r of U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # This is a tensor contraction.
    # For efficiency, we can do: for each r, outer product.
    # Or use einsum: 'ijr,jkr,kir->ijk'
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # The target tensor T[i,j,k] = 1 for all valid (i,j,k) since it's the multiplication tensor
    # Actually, the matrix multiplication tensor is T[i,j,k] = 1 for all i,j,k
    # because (AB)[i,k] = sum_j A[i,j] B[j,k], and each term A[i,j]*B[j,k] is a monomial.
    # The tensor of bilinear forms has T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    target = np.ones((n, m, p), dtype=np.float64)
    
    return np.allclose(result, target, atol=tol, rtol=0)

def _rank1_peel(U, V, W):
    """
    Attempt to find a rank-1 component in the tensor decomposition.
    A rank-1 component has U[:, r] = a * u_i, V[:, r] = b * v_j, W[:, r] = c * w_k
    where u_i is a vector over (i,j), v_j over (j,k), w_k over (k,i).
    
    For the matrix multiplication tensor, a rank-1 term corresponds to:
    U[i,j] = alpha * u[i,j], V[j,k] = beta * v[j,k], W[k,i] = gamma * w[k,i]
    such that for all i,j,k: u[i,j]*v[j,k]*w[k,i] = 1 (the target).
    
    This is only possible if u, v, w are all 1s (for the full tensor).
    But when peeling from a sum, we look for a rank-1 term that is a subset.
    
    Actually, for the multiplication tensor, rank-1 decompositions don't exist
    in the simple sense because the tensor is not rank-1. We need to find
    linear combinations.
    
    Let me reconsider. The de Zeeuw algorithm works by:
    1. Find a hyperplane that cuts off a corner of the tensor.
    2. Project and recurse.
    
    For our purposes, let's try a different approach: use the known fact that
    for small matrices, we can try to find structured decompositions.
    """
    pass

def _try_structured_search(deadline):
    """
    Try to find a lower-rank decomposition using structured search.
    
    Key insight: For 3x4x6 matrix multiplication, the schoolbook rank is 72.
    We can try to exploit the structure. One approach: group terms.
    
    For example, for fixed j, the terms involving A[i,j] and B[j,k] for all i,k
    form a rank-(n*p) sub-tensor. We might be able to share computations.
    
    Let's try a block approach: for each j, we compute the outer product of
    column j of A (length n) and row j of B (length p), giving an n x p matrix.
    The total sum over j gives C.
    
    This is still rank 72 = sum_j (n*p) = 4 * 18 = 72.
    
    To do better, we need to find linear dependencies. One idea: use the fact that
    some entries can be shared if we allow intermediate results.
    
    Let's try a different structured approach: use Strassen-like ideas.
    For 2x2x2, Strassen uses 7 instead of 8.
    
    For 3x4x6, let's try to find a decomposition with rank < 72.
    """
    
    # Try random linear combinations to see if we can reduce rank
    # This is a heuristic search
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Start with schoolbook
    U0, V0, W0 = _build_schoolbook()
    rank0 = n * m * p
    
    best_rank = rank0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # Try to find linear dependencies by checking if some columns are linearly dependent
    # in a certain sense.
    
    # Actually, let's try a more direct approach: use the tensor rank definition.
    # The tensor T is in R^{n x m x p}. We want to write it as sum of rank-1 tensors.
    # A rank-1 tensor is u ⊗ v ⊗ w where u in R^n, v in R^m, w in R^p.
    # But our decomposition is different: U[i,j,r], V[j,k,r], W[k,i,r].
    
    # The standard tensor for matrix multiplication is T[i,j,k] = 1.
    # A rank-1 term in the tensor decomposition is u_i * v_j * w_k.
    # For this to contribute to T[i,j,k] = 1, we need u_i * v_j * w_k to be part of the sum.
    
    # In the "factor" formulation:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_r (sum_j U[i,j,r] A[i,j]) (sum_j V[j,k,r] B[j,k]) ... no, that's not right.
    
    # The correct formulation:
    # C[i,k] = sum_r U[i,k,r] * (something)
    # Actually, the standard formulation for matrix multiplication as a bilinear map:
    # Given A (n x m) and B (m x p), C = A B (n x p).
    # C[i,k] = sum_j A[i,j] B[j,k].
    # We want: C[i,k] = sum_r (sum_j a_{i,j,r} A[i,j]) (sum_j b_{j,k,r} B[j,k])
    # No, that's not right either. The correct form is:
    # C[i,k] = sum_r U_{i,k,r} * (sum_j V_{j,r} * A[i,j] * ... )
    
    # Let me look at the interface more carefully.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # The decomposition says: for the matrix multiplication tensor,
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # And T[i,j,k] = 1 for all i,j,k (the multiplication tensor).
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    # This is a 3-way tensor factorization problem.
    
    # Let's try to find a low-rank solution using iterative methods.
    # Start with random initialization and use ALS (Alternating Least Squares).
    
    np.random.seed(42)
    
    # Try different ranks
    for target_rank in range(50, 73):
        if time.time() > deadline:
            break
        
        # Initialize randomly
        U = np.random.randn(nm, target_rank) * 0.1
        V = np.random.randn(mp, target_rank) * 0.1
        W = np.random.randn(pn, target_rank) * 0.1
        
        # Scale to make the product approximately 1
        # The sum of products should be 1, so scale appropriately
        scale = (1.0 / (target_rank ** 0.5))
        U *= scale
        V *= scale
        W *= scale
        # Actually, U*V*W summed should be 1, so if each is ~scale, then scale^3 * rank ~ 1
        # scale ~ (1/rank)^(1/3)
        scale = (1.0 / target_rank) ** (1.0/3.0)
        U = np.random.randn(nm, target_rank) * scale
        V = np.random.randn(mp, target_rank) * scale
        W = np.random.randn(pn, target_rank) * scale
        
        # ALS iterations
        converged = False
        for it in range(100):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each i,j, U[i*m+j, :] should satisfy
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            # This is: U[i*m+j, :] @ (V[j*p+k, :] * W[k*n+i, :]) = 1 for all k
            # Let M[k, r] = V[j*p+k, r] * W[k*n+i, r]
            # Then U[i*m+j, :] @ M = 1 (vector of ones of length p)
            
            # This is a linear system: M^T @ U[i*m+j, :]^T = 1
            # Or: (M^T M) @ U[i*m+j, :]^T = M^T @ 1
            
            # Do this for all i,j
            for i in range(n):
                for j in range(m):
                    # Build M: p x target_rank
                    M = np.zeros((p, target_rank))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    
                    # Solve M^T M u = M^T 1
                    A_mat = M.T @ M
                    b_vec = M.T @ np.ones(p)
                    try:
                        U[i*m+j, :] = np.linalg.solve(A_mat + 1e-10 * np.eye(target_rank), b_vec)
                    except np.linalg.LinAlgError:
                        U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
            
            # Update V: for each j,k, V[j*p+k, :] should satisfy
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
            # Let M2[i, r] = U[i*m+j, r] * W[k*n+i, r]
            # Then V[j*p+k, :] @ M2 = 1 (vector of ones of length n)
            
            for j in range(m):
                for k in range(p):
                    M2 = np.zeros((n, target_rank))
                    for i in range(n):
                        M2[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    
                    A_mat = M2.T @ M2
                    b_vec = M2.T @ np.ones(n)
                    try:
                        V[j*p+k, :] = np.linalg.solve(A_mat + 1e-10 * np.eye(target_rank), b_vec)
                    except np.linalg.LinAlgError:
                        V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
            
            # Update W: for each k,i, W[k*n+i, :] should satisfy
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
            # Let M3[j, r] = U[i*m+j, r] * V[j*p+k, r]
            # Then W[k*n+i, :] @ M3 = 1 (vector of ones of length m)
            
            for k in range(p):
                for i in range(n):
                    M3 = np.zeros((m, target_rank))
                    for j in range(m):
                        M3[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    
                    A_mat = M3.T @ M3
                    b_vec = M3.T @ np.ones(m)
                    try:
                        W[k*n+i, :] = np.linalg.solve(A_mat + 1e-10 * np.eye(target_rank), b_vec)
                    except np.linalg.LinAlgError:
                        W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
            
            # Check convergence
            if it % 20 == 19:
                if _verify_decomposition(U, V, W, tol=1e-4):
                    converged = True
                    break
        
        if converged:
            best_rank = target_rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            break
    
    return best_U, best_V, best_W, best_rank

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Default: schoolbook decomposition
    U0, V0, W0 = _build_schoolbook()
    best_U, best_V, best_W = U0, V0, W0
    best_rank = n * m * p
    
    # Try structured search
    if time.time() < deadline - 10:
        U_try, V_try, W_try, rank_try = _try_structured_search(deadline)
        if rank_try < best_rank:
            # Verify
            if _verify_decomposition(U_try, V_try, W_try, tol=1e-6):
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = rank_try
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
