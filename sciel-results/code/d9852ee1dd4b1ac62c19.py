"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from numpy.random import default_rng

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-8):
    """
    Check that the decomposition exactly reproduces the matrix multiplication tensor.
    The tensor T[i,j,k] = delta_{i,k} is the canonical matmul tensor where
    (A*B)[i,k] = sum_j A[i,j]*B[j,k].
    
    In our factorization: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    We verify by checking that for random A, B: (A*B)[i,k] == sum_j A[i,j]*B[j,k]
    which is equivalent to sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = delta_{i,k} for all i,j,k.
    
    Actually, the standard matmul tensor has T[i,j,k] = 1 if the indices are "compatible" in the
    matmul sense. Let me re-derive:
    
    Matmul: C[i,k] = sum_j A[i,j] * B[j,k]
    Tensor form: T[i,j,k] such that C[i,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]
    This means T[i,j,k] = 1 if and only if the triple (i,j,k) contributes to C[i,k],
    i.e., T[i,j,k] = delta_{i,k} is wrong.
    
    Correct: T[i,j,k] = 1 for all i,j,k where the contraction gives C[i,k] = sum_j A[i,j]B[j,k].
    So T[i,j,k] = 1 if and only if the indices are "free" — actually T is just the identity
    in the j-index contraction: T[i,j,k] = 1 for all valid i,j,k? No.
    
    The matmul tensor T in R^{n x m x p} is defined by T[i,j,k] = 1 if and only if the
    (i,j) entry of A and (j,k) entry of B produce the (i,k) entry of C. Since every
    (i,j,k) triple with valid indices contributes, T[i,j,k] = 1 for ALL i,j,k.
    
    Wait, that's not right either. Let me think again.
    
    C = A * B where A is n x m and B is m x p.
    C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    As a tensor: C[i,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]
    For this to hold, we need T[i,j,k] = 1 for all i, j, k (since each term A[i,j]*B[j,k]
    appears exactly once in the sum for C[i,k]).
    
    Hmm, but that would make the tensor all-ones, which has rank 1. That can't be right
    for matrix multiplication.
    
    The issue is that the tensor must map (A, B) -> C where C is n x p. The tensor T is in
    R^{n x m x p} and the operation is a bilinear map R^{n x m} x R^{m x p} -> R^{n x p}.
    
    T[i,j,k] should be such that sum_{j} A[i,j] B[j,k] = C[i,k].
    In tensor notation with Einstein summation: C[i,k] = T[i,j,k] A[i,j] B[j,k]
    (summing over j).
    
    For this to work: T[i,j,k] = 1 for all i,j,k. Because then:
    C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k] = sum_j A[i,j] B[j,k]
    
    So yes, T[i,j,k] = 1 for all valid indices. The tensor is the all-ones tensor in R^{n x m x p}.
    
    But wait, that tensor has rank 1! The all-ones tensor J[i,j,k] = 1 can be written as
    u_i v_j w_k where u = ones(n), v = ones(m), w = ones(p). So rank = 1.
    
    This is clearly wrong. Matrix multiplication is NOT just the all-ones tensor.
    
    I think I'm confusing myself. Let me look at the seed code more carefully.
    
    In the seed: U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for each (i,j,k) triple.
    The tensor reconstructed is T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    = 1 (only one c contributes for each (i,j,k)).
    
    So the seed constructs the all-ones tensor, which is indeed rank 1... but the rank is
    n*m*p = 120. That's contradictory.
    
    OH WAIT. I think the convention is different. Let me re-read the task:
    "the output factor follows the official trace convention W[k*n+i,r]"
    
    And the shapes are U: (n*m, rank), V: (m*p, rank), W: (p*n, rank).
    
    The tensor is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For matrix multiplication, we need:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    In terms of the tensor: C[i,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] ... no, that's not
    the right contraction.
    
    The correct tensor formulation: The bilinear map (A,B) -> A*B can be represented by a
    tensor T in R^{(n*m) x (m*p) x (n*p)} such that:
    vec(A*B) = T : (vec(A), vec(B))
    where T[a, b, c] with a indexing A (size n*m), b indexing B (size m*p), c indexing C (size n*p).
    
    T[a, b, c] = 1 if and only if the (a-th) entry of A and (b-th) entry of B multiply to
    contribute to the (c-th) entry of C.
    
    With row-major indexing: a = i*m + j, b = j*p + k, c = i*p + k.
    T[a, b, c] = 1 iff there exist i,j,k such that a = i*m+j, b = j*p+k, c = i*p+k.
    Note: the j in a and the j in b must be the SAME j (that's the summation index).
    
    So T is not all-ones. T[i*m+j, j*p+k, i*p+k] = 1 for all i,j,k, and 0 otherwise.
    
    Now with the given factorization format:
    U has shape (n*m, rank) = (20, rank), indexed by i*m+j
    V has shape (m*p, rank) = (30, rank), indexed by j*p+k  
    W has shape (p*n, rank) = (24, rank), indexed by k*n+i
    
    The reconstructed tensor: T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    where a = i*m+j, b = j'*p+k', c = k''*n+i''
    
    For this to equal the matmul tensor, we need:
    T[i*m+j, j*p+k, k*n+i] = 1 (note: the c index is k*n+i, NOT i*p+k)
    
    Wait, the W index is k*n + i. So c = k*n + i means the C matrix is indexed as C[k, i]?
    That would be C transposed, or the convention is that C is stored column-major?
    
    Let me just trust the seed code. The seed sets:
    For each (i,j,k): U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
    
    So the reconstructed tensor has T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 elsewhere.
    
    This is the correct matmul tensor with the W index convention W[k*n+i] (i.e., C is indexed
    as C[k, i] or equivalently the output is the transpose of the usual C, or the convention
    is that C[i,k] is stored at position k*n+i).
    
    OK so now I understand the tensor. Let me verify my understanding is correct and then
    think about how to find a lower-rank decomposition.
    """
    # Reconstruct tensor T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # and check it equals the matmul tensor.
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    # Build the target tensor
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Compute the reconstructed tensor
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # This is a 3-way tensor product / Khatri-Rao based reconstruction
    # T = U[:,:,None,None] * V[None,:,None,None] * W[None,None,:,None] summed over r
    # More efficiently: T[a,b,c] = (U @ V.T) doesn't directly work for 3-way.
    # Use einsum: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    diff = np.abs(T_recon - T_target)
    return np.max(diff) < tol


def _try_rank(r, n, m, p, max_iters=200, n_restarts=5, seed=42, time_limit=100):
    """
    Try to find a rank-r decomposition using iterative refinement.
    Uses a random initialization and then alternating minimization.
    """
    start_time = time.time()
    rng = default_rng(seed)
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        U = rng.standard_normal((n * m, r))
        V = rng.standard_normal((m * p, r))
        W = rng.standard_normal((p * n, r))
        
        # Scale to unit norm columns to avoid degenerate scaling
        for mat in [U, V, W]:
            for c in range(r):
                norm = np.linalg.norm(mat[:, c])
                if norm > 1e-12:
                    mat[:, c] /= norm
        
        best_err = float('inf')
        converged = False
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Alternating least squares
            # Fix V, W: solve for U
            # T_target[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed V, W: T_target[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
            # Let M_bw = V[b,r] * W[c,r] reshaped... 
            # Actually: for each a, we need U[a,:] such that sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c] for all b,c
            # This is: U[a,:] @ (V * W)^T ... let me think.
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # Define for fixed a: vector t_a of length m*p*p where t_a[b,c] = T[a,b,c]
            # And matrix M_a where M_a[b,c] ... no.
            # T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
            # Let X_r[b,c] = V[b,r] * W[c,r] (outer product)
            # Then T[a] = sum_r U[a,r] * X_r (where T[a] is the (m*p x p*n) slice)
            # This is a linear system: T[a] = X @ U[a,:]^T where X is (m*p*p x r) with X[(b,c),r] = V[b,r]*W[c,r]
            
            # Build X: shape (m*p * p*n, r) where X[b*(p*n)+c, r] = V[b,r] * W[c,r]
            # This is the Khatri-Rao product of V and W (column-wise Kronecker)
            # X = kron(V, W) but with specific ordering... 
            # V is (m*p, r), W is (p*n, r)
            # X[(b,c), r] = V[b,r] * W[c,r]
            # This is: X = (V[:, :, None] * W[None, :, :]).reshape(m*p * p*n, r)
            
            X = (V[:, :, None] * W[None, :, :]).reshape(m * p * p * n, r)
            # X shape: (m*p * p*n, r) = (m*p^2*n, r)
            
            # T_target reshaped: shape (n*m, m*p*p*n) where T_flat[a, (b,c)] = T[a,b,c]
            T_flat = T_target_global.reshape(n * m, m * p * p * n)
            
            # Solve: T_flat @ U^T = X^T ... 
            # T_flat[a, idx] = sum_r U[a, r] * X[idx, r]
            # So T_flat = U @ X^T
            # Solve for U: T_flat = U @ X^T  =>  U = T_flat @ (X^T)^+ 
            # Using least squares: U @ X^T ≈ T_flat  =>  (X @ U^T)^T = T_flat
            # X @ U^T = T_flat^T, shape (m*p*p*n, n*m)
            # U^T = (X^T X)^{-1} X^T T_flat^T (if X has full column rank)
            # Or use np.linalg.lstsq(X, T_flat.T)
            
            try:
                U_new, _, _, _ = np.linalg.lstsq(X, T_flat.T, rcond=None)
                U = U_new.T  # shape (n*m, r)
            except np.linalg.LinAlgError:
                break
            
            # Fix U, W: solve for V
            # T[a,b,c] = sum_r V[b,r] * (U[a,r] * W[c,r])
            # For fixed b: T[:,b,:] = sum_r V[b,r] * (U[:,r] * W[:,r]) reshaped
            # Y[(a,c), r] = U[a,r] * W[c,r], shape (n*m * p*n, r)
            Y = (U[:, :, None] * W[None, :, :]).reshape(n * m * p * n, r)
            # T_target reshaped to (m*p, n*m * p*n): T_flat2[b, (a,c)] = T[a,b,c]
            # T_target is (n*m, m*p, p*n). We want T_flat2[b, a*(p*n)+c] = T[a,b,c]
            T_flat2 = T_target_global.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
            
            try:
                V_new, _, _, _ = np.linalg.lstsq(Y, T_flat2.T, rcond=None)
                V = V_new.T  # shape (m*p, r)
            except np.linalg.LinAlgError:
                break
            
            # Fix U, V: solve for W
            # T[a,b,c] = sum_r W[c,r] * (U[a,r] * V[b,r])
            # Z[(a,b), r] = U[a,r] * V[b,r], shape (n*m * m*p, r)
            Z = (U[:, :, None] * V[None, :, :]).reshape(n * m * m * p, r)
            # T_flat3[c, (a,b)] = T[a,b,c]
            T_flat3 = T_target_global.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
            
            try:
                W_new, _, _, _ = np.linalg.lstsq(Z, T_flat3.T, rcond=None)
                W = W_new.T  # shape (p*n, r)
            except np.linalg.LinAlgError:
                break
            
            # Check convergence
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T_recon - T_target_global))
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-10:
                converged = True
                break
        
        if converged or best_err < 1e-8:
            return best_U, best_V, best_W
    
    return None


# Precompute the target tensor
_n, _m, _p = DIMENSIONS
T_target_global = np.zeros((_n * _m, _m * _p, _p * _n))
for _i in range(_n):
    for _j in range(_m):
        for _k in range(_p):
            T_target_global[_i * _m + _j, _j * _p + _k, _k * _n + _i] = 1.0


def construct():
    global T_target_global
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin before 150s limit
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = (U_sb, V_sb, W_sb)
    best_rank = n * m * p  # 120
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(20, 30, 24) = 30
    # Upper bound from schoolbook: 120
    # Try ranks from a few above the lower bound up to schoolbook
    
    # For 4x5x6 matmul, known best is around 50-60 (strassen-like methods)
    # Let's try a range of ranks
    
    ranks_to_try = [30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 110, 119]
    
    for r in ranks_to_try:
        if time.time() - start_time > deadline:
            break
        
        time_left = deadline - (time.time() - start_time)
        if time_left < 5:
            break
        
        # Adjust iterations and restarts based on remaining time
        max_iters = min(150, int(time_left * 2))
        n_restarts = min(8, max(2, int(time_left / 15)))
        
        result = _try_rank(r, n, m, p, max_iters=max_iters, n_restarts=n_restarts, 
                          seed=42 + r, time_limit=time_left - 2)
        
        if result is not None:
            U_r, V_r, W_r = result
            # Verify
            if _verify(U_r, V_r, W_r, n, m, p, tol=1e-8):
                if r < best_rank:
                    best_rank = r
                    best_result = (U_r, V_r, W_r)
                break  # Found valid decomposition at this rank, try lower? No, we went low to high.
            # If verification fails, the result isn't valid, continue
    
    U_best, V_best, W_best = best_result
    
    # Convert to lists of lists with integer/half-integer entries if possible
    # The problem says "integer/half-integer entries" - let's round to nearest half-integer
    def to_half_integer_list(arr):
        arr = np.asarray(arr, dtype=float)
        # Round to nearest 0.5
        arr_rounded = np.round(arr * 2) / 2
        # Verify rounding didn't break things (shouldn't if solution is clean)
        return arr_rounded.tolist()
    
    # Check if rounding to half-integers preserves validity
    U_hi = to_half_integer_list(U_best)
    V_hi = to_half_integer_list(V_best)
    W_hi = to_half_integer_list(W_best)
    
    if _verify(U_hi, V_hi, W_hi, n, m, p, tol=1e-8):
        return {"U": U_hi, "V": V_hi, "W": W_hi}
    else:
        # Fall back to the non-rounded version (still valid)
        return {"U": np.asarray(U_best).tolist(), "V": np.asarray(V_best).tolist(), "W": np.asarray(W_best).tolist()}

# EVOLVE-BLOCK-END
