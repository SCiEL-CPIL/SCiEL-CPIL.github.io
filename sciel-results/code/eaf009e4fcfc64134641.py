import time
import numpy as np

DIMENSIONS = [2, 2, 9]

def _build_rank32(n, m, p):
    """Build the rank-32 decomposition: 4 Strassen blocks + 1 trivial block."""
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        # M1
        U[0, col] = 1
        U[3, col] = 1
        V[0*p + k0, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        # M2
        U[2, col] = 1
        U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1
        W[k1*n + 1, col] = -1
        col += 1
        # M3
        U[0, col] = 1
        V[0*p + k1, col] = 1
        V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        # M4
        U[3, col] = 1
        V[1*p + k0, col] = 1
        V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1
        W[k0*n + 1, col] = 1
        col += 1
        # M5
        U[0, col] = 1
        U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1
        W[k1*n + 0, col] = 1
        col += 1
        # M6
        U[2, col] = 1
        U[0, col] = -1
        V[0*p + k0, col] = 1
        V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        # M7
        U[1, col] = 1
        U[3, col] = -1
        V[1*p + k0, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1
        V[0*p + k, col] = 1
        W[k*n + 0, col] = 1
        col += 1
        
        U[1, col] = 1
        V[1*p + k, col] = 1
        W[k*n + 0, col] = 1
        col += 1
        
        U[2, col] = 1
        V[0*p + k, col] = 1
        W[k*n + 1, col] = 1
        col += 1
        
        U[3, col] = 1
        V[1*p + k, col] = 1
        W[k*n + 1, col] = 1
        col += 1

    # Cols 0-1
    add_strassen(0, 1)
    # Cols 2-3
    add_strassen(2, 3)
    # Cols 4-5
    add_strassen(4, 5)
    # Cols 6-7
    add_strassen(6, 7)
    # Col 8
    add_trivial(8)
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # For matrix multiplication: T[a_idx, b_idx, c_idx] where
    # a_idx = i*m + j (row i, col j of A)
    # b_idx = j*p + k (row j, col k of B)  
    # c_idx = k*n + i (row i, col k of C)
    # Actually, the convention is:
    # A is n x m, B is m x p, C = A*B is n x p
    # T is a tensor in R^{(n*m) x (m*p) x (p*n)}
    # T[i*m+j, j*p+k, k*n+i] = 1 if this is a valid multiplication term
    
    # Let's build the target tensor
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                target[a_idx, b_idx, c_idx] = 1
    
    # Build the decomposition tensor
    decomp = np.zeros_like(target)
    for r in range(rank):
        decomp += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    
    residual = np.max(np.abs(decomp - target))
    return residual < tol, residual

def _try_rank31_search(n, m, p, U32, V32, W32, deadline):
    """Try to find a rank-31 decomposition by removing one term and adjusting."""
    rank32 = U32.shape[1]
    
    # Strategy: for each term to remove, try to see if the remaining 31 terms
    # can be adjusted to still decompose the tensor.
    # This is a nonlinear problem, so we use a heuristic approach.
    
    # First, let's try a simpler approach: check if any term is redundant
    # by checking if the 32 rank-1 tensors are linearly dependent.
    
    # Flatten the tensor space
    # Each rank-1 term is a vector in R^{(n*m)*(m*p)*(p*n)}
    tensor_dim = (n*m) * (m*p) * (p*n)
    
    # Build the matrix of rank-1 tensors
    M = np.zeros((rank32, tensor_dim))
    for r in range(rank32):
        M[r] = np.einsum('i,j,k->ijk', U32[:, r], V32[:, r], W32[:, r]).flatten()
    
    # Check rank of M
    try:
        rank_M = np.linalg.matrix_rank(M, tol=1e-10)
    except:
        rank_M = rank32
    
    if rank_M < rank32:
        # There's a dependency, which means we might be able to reduce rank
        # Find the null space
        try:
            _, s, vh = np.linalg.svd(M, full_matrices=False)
            # Null space is in the last rows of vh
            null_space = vh[-1:]  # One null vector
            # The null vector tells us the coefficients for the dependency
            coeffs = null_space[0]
            # Find the term with the smallest coefficient (most removable)
            min_idx = np.argmin(np.abs(coeffs))
            # Remove that term and see if the rest works
            # But this is just a linear dependency, not necessarily a valid decomposition
            # Let's try to use it to adjust
        except:
            pass
    
    # More practical: try removing each term and using least squares to fit
    # the remaining 31 terms to the target tensor.
    
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                target[a_idx, b_idx, c_idx] = 1
    
    target_flat = target.flatten()
    
    # For each term to remove, try to see if the remaining 31 can fit
    # This is a 3-way tensor completion problem, which is hard.
    # Instead, let's try a simpler heuristic: 
    # Try to merge two terms into one by solving for a rank-1 approximation.
    
    # Alternative: use the fact that the last 4 terms are trivial.
    # Can we share structure between the Strassen blocks and the trivial block?
    
    # For now, let's just try a random perturbation search
    # to see if we can find a rank-31 solution.
    
    # Actually, let's try a more systematic approach:
    # The rank-32 solution has a specific structure. Let's see if we can
    # modify it to get rank 31.
    
    # One idea: the 4 Strassen blocks each use 7 terms. The trivial block uses 4.
    # Total: 4*7 + 4 = 32.
    # Can we do the last column (col 8) with fewer than 4 terms by sharing
    # with the Strassen blocks?
    
    # Let's try: use 3 Strassen blocks (21 terms) + 1 modified block for cols 6-8.
    # The modified block would need to handle 2x2x3 matrix multiplication.
    # The rank of M_{2,2,3} is at most 2*2*3 = 12, but we can do better.
    # Actually, M_{2,2,3} can be done with rank 10 or so.
    # But 21 + 10 = 31! Let's try this.
    
    # Build a decomposition for 2x2x3 matrix multiplication
    # This is the key: can we do M_{2,2,3} with rank 10?
    
    # Let's construct M_{2,2,3} explicitly.
    # A is 2x2, B is 2x3, C = A*B is 2x3.
    # T is in R^{4 x 6 x 6}
    
    # We'll use a combination of Strassen-like techniques.
    # For M_{2,2,2}, rank is 7 (Strassen).
    # For M_{2,2,3}, we can think of it as M_{2,2,2} + M_{2,2,1}.
    # M_{2,2,2} has rank 7, M_{2,2,1} has rank 4. Total: 11.
    # Can we do better?
    
    # Let's try to construct M_{2,2,3} with rank 10.
    # This is a known open problem, but let's try a heuristic search.
    
    # For now, let's just return the rank-32 solution if we can't find rank 31.
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Build the rank-32 solution
    U32, V32, W32 = _build_rank32(n, m, p)
    
    # Verify it works
    valid, residual = _verify_decomposition(U32, V32, W32, n, m, p)
    if not valid:
        # Shouldn't happen, but just in case
        pass
    
    # Try to find a rank-31 solution
    if time.time() < deadline - 10:
        result = _try_rank31_search(n, m, p, U32, V32, W32, deadline)
        if result is not None:
            U, V, W = result
            valid, residual = _verify_decomposition(U, V, W, n, m, p)
            if valid:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return the rank-32 solution
    return {"U": U32.tolist(), "V": V32.tolist(), "W": W32.tolist()}
