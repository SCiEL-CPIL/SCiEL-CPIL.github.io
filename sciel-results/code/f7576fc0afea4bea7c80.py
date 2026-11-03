"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import time

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _ranked_decomposition(U, V, W):
    """Compute the tensor sum_c U[:,c] outer V[:,c] outer W[:,c]."""
    r = U.shape[1]
    result = np.zeros(U.shape[0:1] + V.shape[0:1] + W.shape[0:1])
    for c in range(r):
        result += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
    return result

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that the decomposition exactly reconstructs T."""
    recon = _ranked_decomposition(U, V, W)
    return np.max(np.abs(recon - T)) < tol

def _construct_block_decomposition(n, m, p):
    """
    Construct a block-structured decomposition for matrix multiplication.
    
    For 3x4x8: Split the 4 columns of A into two groups of 2, and the 8 columns of B into two groups of 4.
    Use the structure that C = A*B can be computed via block matrix multiplication.
    
    The idea: A (3x4) = [A1 A2] where A1, A2 are 3x2
    B (4x8) = [B1; B2] where B1, B2 are 2x8
    C = A1*B1 + A2*B2
    
    Each sub-problem is 3x2x8, which has schoolbook rank 3*2*8 = 48.
    Total rank = 48 + 48 = 96 (same as schoolbook).
    
    But we can do better by sharing structure. Let's try a different approach:
    Use the fact that we can pack multiple operations.
    """
    n, m, p = 3, 4, 8
    
    # Try a decomposition based on the observation that we can group operations
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
    
    # Let's try to use a rank-1 decomposition for each (i,j,k) triple
    # but with some sharing via the structure of the indices.
    
    # Actually, let's try a constructive approach:
    # For each pair (i, k), we need to compute C[i,k] = sum_j A[i,j] * B[j,k]
    # This is an inner product of row i of A and column k of B.
    
    # The standard decomposition uses one rank-1 term per (i,j,k) triple.
    # Can we share terms?
    
    # Let's try a different block structure:
    # Split j into two groups: j in {0,1} and j in {2,3}
    # Split k into two groups: k in {0,1,2,3} and k in {4,5,6,7}
    
    # For the first block (j in {0,1}, k in {0,1,2,3}):
    # C[i, k] += A[i,0]*B[0,k] + A[i,1]*B[1,k] for k in {0,1,2,3}
    # This is a 3x2x4 multiplication, rank at most 3*2*4 = 24
    
    # For the second block (j in {0,1}, k in {4,5,6,7}):
    # C[i, k] += A[i,0]*B[0,k] + A[i,1]*B[1,k] for k in {4,5,6,7}
    # This is a 3x2x4 multiplication, rank at most 24
    
    # For the third block (j in {2,3}, k in {0,1,2,3}):
    # C[i, k] += A[i,2]*B[2,k] + A[i,3]*B[3,k] for k in {0,1,2,3}
    # This is a 3x2x4 multiplication, rank at most 24
    
    # For the fourth block (j in {2,3}, k in {4,5,6,7}):
    # C[i, k] += A[i,2]*B[2,k] + A[i,3]*B[3,k] for k in {4,5,6,7}
    # This is a 3x2x4 multiplication, rank at most 24
    
    # Total: 4 * 24 = 96 (still same as schoolbook)
    
    # Let's try to be smarter. What if we use a Strassen-like approach?
    # Or what if we exploit that some entries can be shared?
    
    # Actually, let's just return the schoolbook decomposition for now
    # and try to optimize it.
    
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    
    return U, V, W, rank

def _try_alternating_least_squares(n, m, p, target_rank, max_iters=500, seed=42, time_limit=60):
    """
    Try to find a decomposition with the given target rank using alternating least squares.
    """
    rng = np.random.RandomState(seed)
    T = _build_tensor(n, m, p)
    
    start_time = time.time()
    
    best_U, best_V, best_W = None, None, None
    best_loss = np.inf
    
    n_restarts = 3
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize
        U = rng.randn(n*m, target_rank) * 0.5
        V = rng.randn(m*p, target_rank) * 0.5
        W = rng.randn(p*n, target_rank) * 0.5
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Update U: for each column c, minimize ||T - sum_c U_c V_c W_c||^2 w.r.t. U[:,c]
            # This is a linear system: for each i, sum_c U[i,c] * (V_c outer W_c)[i,:] = T[i,:,:]
            # Actually, it's: for each i, sum_c U[i,c] * V_c[k] * W_c[l] = T[i,k,l]
            
            # More carefully: The reconstruction is sum_c U[:,c] V[:,c] W[:,c]
            # The error is T - sum_c U[:,c] V[:,c] W[:,c]
            # To update U[:,c0], we fix all other columns and solve for U[:,c0]
            
            # The contribution of column c0 is U[:,c0] * V[:,c0] * W[:,c0]
            # The residual is T - sum_{c != c0} U[:,c] V[:,c] W[:,c]
            # We want to minimize ||residual - U[:,c0] V[:,c0] W[:,c0]||^2
            
            # This is a tensor least squares problem.
            # For a fixed V[:,c0] and W[:,c0], we're solving for U[:,c0]
            # to minimize ||residual - U[:,c0] * V[:,c0] * W[:,c0]||^2
            
            # This can be solved by: for each i, U[i,c0] = <residual[i,:,:], V[:,c0] * W[:,c0]> / ||V[:,c0] * W[:,c0]||^2
            
            # Let's do this for each column in turn (Gauss-Seidel style)
            
            for c in range(target_rank):
                # Compute residual without column c
                residual = T.copy()
                for c2 in range(target_rank):
                    if c2 != c:
                        residual -= np.einsum('i,j,k->ijk', U[:,c2], V[:,c2], W[:,c2])
                
                # Solve for U[:,c]
                # For each i: U[i,c] = sum_{j,k} residual[i,j,k] * V[j,c] * W[k,c] / (sum_{j,k} V[j,c]^2 * W[k,c]^2)
                numerator = np.einsum('ijk,j,k->i', residual, V[:,c], W[:,c])
                denominator = np.sum(V[:,c]**2) * np.sum(W[:,c]**2)
                if denominator > 1e-12:
                    U[:,c] = numerator / denominator
                else:
                    U[:,c] = 0
            
            # Similarly update V and W
            for c in range(target_rank):
                residual = T.copy()
                for c2 in range(target_rank):
                    if c2 != c:
                        residual -= np.einsum('i,j,k->ijk', U[:,c2], V[:,c2], W[:,c2])
                
                numerator = np.einsum('ijk,i,k->j', residual, U[:,c], W[:,c])
                denominator = np.sum(U[:,c]**2) * np.sum(W[:,c]**2)
                if denominator > 1e-12:
                    V[:,c] = numerator / denominator
                else:
                    V[:,c] = 0
            
            for c in range(target_rank):
                residual = T.copy()
                for c2 in range(target_rank):
                    if c2 != c:
                        residual -= np.einsum('i,j,k->ijk', U[:,c2], V[:,c2], W[:,c2])
                
                numerator = np.einsum('ijk,i,j->k', residual, U[:,c], V[:,c])
                denominator = np.sum(U[:,c]**2) * np.sum(V[:,c]**2)
                if denominator > 1e-12:
                    W[:,c] = numerator / denominator
                else:
                    W[:,c] = 0
            
            # Compute loss
            recon = np.zeros_like(T)
            for c in range(target_rank):
                recon += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
            loss = np.sum((recon - T)**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-8:
                break
    
    if best_U is not None and best_loss < 1e-6:
        return best_U, best_V, best_W, best_loss
    return None, None, None, best_loss

def _snap_to_rationals(U, V, W, tol=1e-6):
    """Try to snap values to rationals with small denominators."""
    U_s = U.copy()
    V_s = V.copy()
    W_s = W.copy()
    
    for denom in [1, 2, 3, 4, 5, 6, 7, 8]:
        U_cand = np.round(U * denom) / denom
        V_cand = np.round(V * denom) / denom
        W_cand = np.round(W * denom) / denom
        
        err_U = np.max(np.abs(U_cand - U))
        err_V = np.max(np.abs(V_cand - V))
        err_W = np.max(np.abs(W_cand - W))
        
        if err_U < tol and err_V < tol and err_W < tol:
            return U_cand, V_cand, W_cand, denom
    
    return U, V, W, 1

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Build the target tensor
    T = _build_tensor(n, m, p)
    
    # First, try the block-structured constructive approach
    # which should give rank 96 (schoolbook)
    U_block, V_block, W_block, rank_block = _construct_block_decomposition(n, m, p)
    
    # Try to find better decompositions using ALS
    candidate_ranks = [72, 64, 56, 48, 40, 36, 32]
    
    best_result = None
    best_rank = rank_block
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 30:
            break
        
        U, V, W, loss = _try_alternating_least_squares(n, m, p, target_rank, 
                                                         max_iters=200, seed=42, 
                                                         time_limit=25)
        
        if U is not None and loss < 1e-6:
            # Try to snap to rationals
            U_s, V_s, W_s, denom = _snap_to_rationals(U, V, W, tol=1e-5)
            
            # Verify the snapped version
            recon = _ranked_decomposition(U_s, V_s, W_s)
            snap_loss = np.sum((recon - T)**2)
            
            if snap_loss < 1e-4:
                best_result = (U_s, V_s, W_s)
                best_rank = target_rank
                break
    
    if best_result is not None:
        U_s, V_s, W_s = best_result
        U_list = [[float(x) for x in U_s[i]] for i in range(n*m)]
        V_list = [[float(x) for x in V_s[i]] for i in range(m*p)]
        W_list = [[float(x) for x in W_s[i]] for i in range(p*n)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition
    U_list = [[float(x) for x in U_block[i]] for i in range(n*m)]
    V_list = [[float(x) for x in V_block[i]] for i in range(m*p)]
    W_list = [[float(x) for x in W_block[i]] for i in range(p*n)]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
