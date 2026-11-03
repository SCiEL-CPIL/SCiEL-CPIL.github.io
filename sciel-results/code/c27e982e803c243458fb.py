"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    
    The decomposition must satisfy:
    For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    And for all (i,j,k) where the product should be 0, it must be 0.
    Actually, the tensor T[i,j,k] = 1 if we're computing C[i,k] += A[i,j]*B[j,k]
    So T is indexed by (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    and T[i,j,k] = 1 for all valid (i,j,k) since each term A[i,j]*B[j,k] contributes.
    
    Wait, let me reconsider. The matrix multiplication tensor has entries:
    T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    And the decomposition is:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check all (i, j, k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _try_rank(rank, n, m, p, max_time=100.0):
    """Try to find a rank-`rank` decomposition using randomized SVD-based approach."""
    start_time = time.time()
    deadline = start_time + max_time
    
    # The tensor T has shape (n*m, m*p, p*n) but only specific entries are 1.
    # Actually, let's think of it differently.
    # T is a 3-tensor with dimensions (n*m) x (m*p) x (p*n)
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # Otherwise T[a, b, c] = 0
    
    # For matrix multiplication, the tensor is:
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i, j, k
    
    # We want to find U (n*m x rank), V (m*p x rank), W (p*n x rank) such that
    # T[a,b,c] = sum_c U[a,c] * V[b,c] * W[c_idx,c]
    
    # Strategy: Use the fact that for fixed j, the sub-tensor over (i, k) is rank-1
    # (it's outer product of e_i and e_k vectors). So the whole tensor can be written
    # as sum over j of (sum over i of e_i in first mode) tensor (e_j in second mode) tensor (sum over k of e_k in third mode)
    # But that gives rank m, which is not right because the indices don't separate cleanly.
    
    # Better approach: The standard schoolbook decomposition has rank n*m*p.
    # We can try to find a lower rank decomposition.
    
    # For 3x5x6, the known lower bound is at least max(n*m, m*p, p*n) / something...
    # Actually, the Strassen-like algorithms give better bounds for square matrices.
    # For rectangular, it's less clear.
    
    # Let's try a randomized approach:
    # 1. Generate random projections
    # 2. Use them to estimate the tensor rank
    # 3. Try to find a low-rank approximation
    
    # First, let's construct the full tensor as a 3D array
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find rank-r decomposition using iterative method
    # Initialize U, V, W randomly
    np.random.seed(42)
    
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Random initialization
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Iterative refinement: alternate between updating U, V, W
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            # Update U: for each row a, U[a, :] should satisfy
            # sum_c U[a,c] * V[b,c] * W[c_idx,c] = T[a,b,c] for all b, c_idx
            # This is a linear system if we fix V and W
            # For fixed b, c_idx: sum_c U[a,c] * (V[b,c] * W[c_idx,c]) = T[a,b,c]
            # Let M_a = matrix where M_a[b, c] = V[b,c] * W[c_idx,c]... this isn't quite right.
            
            # Better: for each a, the equation is:
            # sum_c U[a,c] * (V[:, c] * W[:, c]) = T[a, :, :] (vectorized)
            # This is a linear system: U[a, :] @ (V * W) = T[a, :, :].reshape(-1)
            # where (V * W) is the element-wise product reshaped... no.
            
            # Let's think again. For fixed a:
            # sum_c U[a,c] * V[b,c] * W[d,c] = T[a, b, d] for all b, d
            # This is: U[a, :] is a solution to a linear system where the "matrix" is formed
            # by the outer products of V[:, c] and W[:, c].
            
            # Specifically, define A_a as a (m*p * p*n) x rank matrix where
            # A_a[(b, d), c] = V[b, c] * W[d, c]
            # Then A_a @ U[a, :] = T[a, :, :].reshape(-1)
            
            # This is a least squares problem.
            
            # Build the matrix for row a
            # Actually, let's do it more efficiently.
            
            # For updating U[a, :]:
            # We need to solve: sum_c U[a,c] * V[b,c] * W[d,c] = T[a,b,d]
            # Let's form the matrix M where M[(b,d), c] = V[b,c] * W[d,c]
            # Then M @ U[a, :] = T[a, :, :].flatten()
            
            # But building M for each a is expensive. Let's use the structure.
            
            # Alternative: use the CP decomposition update formulas.
            # For CP decomposition, the update for U is:
            # U = T_1 * (W ⊙ V)^+ where T_1 is the mode-1 unfolding
            
            # Mode-1 unfolding of T: T1[a, (b, d)] = T[a, b, d]
            # W ⊙ V: Khatri-Rao product, (W ⊙ V)[(b, d), c] = W[b, c] * V[d, c]
            # Wait, I need to be careful with the indices.
            
            # Let me define:
            # T1: mode-1 unfolding, shape (n*m, m*p * p*n)
            # T1[a, b * p*n + d] = T[a, b, d]
            
            # For the CP decomposition T ≈ sum_c U[:, c] ⊗ V[:, c] ⊗ W[:, c]
            # The mode-1 unfolding is: T1 ≈ U @ (W ⊙ V)^T
            # where (W ⊙ V)[(b, d), c] = W[b, c] * V[d, c]
            
            # So to update U: U = T1 @ (W ⊙ V)^+ (pseudo-inverse)
            
            # Let's compute this.
            
            # Build W ⊙ V: shape (m*p * p*n, rank)
            # (W ⊙ V)[b * p*n + d, c] = W[d, c] * V[b, c]
            
            # This is expensive to build explicitly, but for small sizes it's fine.
            
            # Actually, for n=3, m=5, p=6:
            # n*m = 15, m*p = 30, p*n = 18
            # rank is at most 90 (schoolbook), so W ⊙ V is 540 x rank
            # That's manageable.
            
            # Compute W ⊙ V
            # (W ⊙ V)[b * p*n + d, c] = W[d, c] * V[b, c]
            WkV = np.zeros((m*p * p*n, rank))
            for b in range(m*p):
                for d in range(p*n):
                    WkV[b * p*n + d, :] = W[d, :] * V[b, :]
            
            # T1: mode-1 unfolding
            T1 = T.reshape(n*m, -1)
            
            # Solve U = T1 @ pinv(WkV)
            try:
                U_new = T1 @ np.linalg.pinv(WkV)
            except:
                break
            
            # Update V
            # T2: mode-2 unfolding, shape (m*p, n*m * p*n)
            # T2[b, a * p*n + d] = T[a, b, d]
            T2 = np.zeros((m*p, n*m * p*n))
            for a in range(n*m):
                for d in range(p*n):
                    T2[:, a * p*n + d] = T[a, :, d]
            
            # (W ⊙ U)[(a, d), c] = W[d, c] * U[a, c]
            WkU = np.zeros((n*m * p*n, rank))
            for a in range(n*m):
                for d in range(p*n):
                    WkU[a * p*n + d, :] = W[d, :] * U_new[a, :]
            
            try:
                V_new = T2 @ np.linalg.pinv(WkU)
            except:
                break
            
            # Update W
            # T3: mode-3 unfolding, shape (p*n, n*m * m*p)
            # T3[d, a * m*p + b] = T[a, b, d]
            T3 = np.zeros((p*n, n*m * m*p))
            for a in range(n*m):
                for b in range(m*p):
                    T3[:, a * m*p + b] = T[a, b, :]
            
            # (V ⊙ U_new)[(a, b), c] = V_new[b, c] * U_new[a, c]
            VkU = np.zeros((n*m * m*p, rank))
            for a in range(n*m):
                for b in range(m*p):
                    VkU[a * m*p + b, :] = V_new[b, :] * U_new[a, :]
            
            try:
                W_new = T3 @ np.linalg.pinv(VkU)
            except:
                break
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            # Compute the reconstruction error
            recon = np.zeros_like(T)
            for c in range(rank):
                recon += U[:, c][:, None, None] * V[:, c][None, :, None] * W[:, c][None, None, :]
            
            error = np.linalg.norm(recon - T)
            if error < best_error:
                best_error = error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if error < 1e-10:
                break
        
        if best_error < 1e-9:
            # Try to round to half-integers
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            if _verify(U_rounded, V_rounded, W_rounded, n, m, p):
                return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
            
            # If rounding doesn't work, try the unrounded version (but it needs to be exact)
            # The task requires integer/half-integer entries, so we need to round.
            # If we can't round exactly, this rank might not work with half-integers.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try to find lower rank decompositions
    # The theoretical lower bound for matrix multiplication rank is at least
    # max(n*m, m*p, p*n) but for rectangular it's more complex.
    # For 3x5x6, let's try ranks from some lower bound up to schoolbook.
    
    # A simple lower bound: rank >= max(n*m, m*p, p*n) / min(n, m, p) ... not quite.
    # Actually, the rank must be at least max(n*m, m*p, p*n) / max(n, m, p) ... no.
    # The rank is at least the maximum of the ranks of the mode-i unfoldings.
    # Mode-1 unfolding has rank at most m*p (for each a, the slice over (b,d) has rank <= m*p)
    # Actually, the rank of the mode-1 unfolding is at most min(n*m, m*p*p*n) but also at most the tensor rank.
    
    # Let's just try a range of ranks.
    # For 3x5x6, schoolbook is 90. Let's try from 30 up to 90.
    
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin
    
    for rank in range(30, schoolbook_rank + 1):
        if time.time() > deadline:
            break
        
        result = _try_rank(rank, n, m, p, max_time=min(5.0, deadline - time.time()))
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # If we couldn't find a lower rank, return schoolbook
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
