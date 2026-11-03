"""Program for rank-minimal exact tensor decomposition of 3x3x5 matrix multiplication."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np
from scipy.optimize import linprog

def _validate_decomposition(U, V, W, n, m, p, rank):
    """Validate that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check dimensions
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # The decomposition should satisfy:
    # For each output position (i_out, j_out), the bilinear form should give
    # sum over k of A[i_out, k] * B[k, j_out]
    # In tensor form: T[i,j,k; i',j',k'] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # where i,i' are row indices of A (0..n-1), j,j' are col/row indices (0..m-1),
    # k,k' are col indices of B / row indices of output (0..p-1)
    
    # Check all tensor components
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Output position is (i, k) in the n x p matrix
                # The bilinear form for this output should be sum_{l=0}^{m-1} A[i,l] * B[l,k]
                
                # Compute the bilinear form from the decomposition
                # U[i*m + l, r] * V[l*p + k, r] * W[k*n + i, r] summed over r
                bilinear_val = 0.0
                for l in range(m):
                    for r in range(rank):
                        bilinear_val += U[i*m + l, r] * V[l*p + k, r] * W[k*n + i, r]
                
                # This should equal delta_{j,k} (since B[l,k] has l as row, k as col)
                # Wait, let me reconsider the tensor structure
                # Actually for C = A*B where A is n x m, B is m x p:
                # C[i, k] = sum_{j=0}^{m-1} A[i, j] * B[j, k]
                # So the tensor T[i, j, k; i', j', k'] should be:
                # T[i, j, k; i', j', k'] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
                # But this doesn't seem right either.
                
                # Let me use the trace convention from the problem:
                # W[k*n + i, r] suggests output index is k*n + i
                # So output is indexed by (k, i) where k is 0..p-1, i is 0..n-1
                # This means output matrix is p x n? That's unusual.
                
                # Actually, let me just verify by checking that the decomposition
                # correctly computes matrix multiplication for random inputs.
                
    # Better approach: verify by checking the bilinear identity
    # For the decomposition to work, we need:
    # sum_r U[i*m + j, r] * V[j'*p + k, r] * W[k*n + i, r] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # Wait, that's not quite right either.
    
    # Let me think about this differently. The standard bilinear form for matrix multiplication:
    # C = A * B, where A is n x m, B is m x p, C is n x p
    # C[i, k] = sum_{j=0}^{m-1} A[i, j] * B[j, k]
    # 
    # In the decomposition, we have:
    # C[i, k] = sum_{r=0}^{rank-1} (sum_{j} U[i*m + j, r] * A[i, j]) * (sum_{j} V[j*p + k, r] * B[j, k])
    # Wait, that's not standard either.
    
    # The standard form is:
    # C[i, k] = sum_{r} (sum_{j} U[i, j, r] * A[i, j]) * (sum_{j} V[j, k, r] * B[j, k])
    # 
    # But the problem says U has shape (n*m, rank), so U is flattened.
    # Let me assume:
    # U[i*m + j, r] is the coefficient for A[i, j] in the r-th bilinear form
    # V[j*p + k, r] is the coefficient for B[j, k] in the r-th bilinear form  
    # W[k*n + i, r] is the coefficient for C[i, k] in the r-th bilinear form
    # 
    # Then: C[i, k] = sum_{r} W[k*n + i, r] * (sum_{j} U[i*m + j, r] * A[i, j]) * (sum_{j} V[j*p + k, r] * B[j, k])
    
    # This should equal sum_{j} A[i, j] * B[j, k] for all i, k
    
    # Test with random matrices
    for _ in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_decomposed = np.zeros((n, p))
        for r in range(rank):
            u_prod = np.zeros((n,))
            v_prod = np.zeros((p,))
            for i in range(n):
                for j in range(m):
                    u_prod[i] += U[i*m + j, r] * A[i, j]
            for j in range(m):
                for k in range(p):
                    v_prod[k] += V[j*p + k, r] * B[j, k]
            for i in range(n):
                for k in range(p):
                    C_decomposed[i, k] += W[k*n + i, r] * u_prod[i] * v_prod[k]
        
        if not np.allclose(C_decomposed, C_expected, atol=1e-10):
            return False
    
    return True

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return U.tolist(), V.tolist(), W.tolist(), rank

def _try_random_rank_reduction(n, m, p, target_rank, max_iterations=1000, deadline=None):
    """Try to find a decomposition with target_rank using random search."""
    best_rank = n * m * p
    best_decomp = None
    
    if deadline is not None and time.time() > deadline:
        return best_rank, best_decomp
    
    for iteration in range(max_iterations):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Randomly initialize with small values
        U = np.random.choice([-1, 0, 1], size=(n*m, target_rank))
        V = np.random.choice([-1, 0, 1], size=(m*p, target_rank))
        W = np.random.choice([-1, 0, 1], size=(p*n, target_rank))
        
        # Check if this is a valid decomposition
        if _validate_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, target_rank):
            best_rank = target_rank
            best_decomp = (U.tolist(), V.tolist(), W.tolist(), target_rank)
            break
    
    return best_rank, best_decomp

def _greedy_rank_reduction(n, m, p, start_rank, max_iterations=500, deadline=None):
    """Try to reduce rank by removing columns one at a time."""
    # Start with schoolbook
    U, V, W, current_rank = _schoolbook_decomposition(n, m, p)
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    best_rank = current_rank
    best_decomp = (U.tolist(), V.tolist(), W.tolist(), current_rank)
    
    for iteration in range(max_iterations):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        if current_rank <= best_rank:
            break
        
        # Try removing each column
        found_reduction = False
        for col_to_remove in range(current_rank):
            if deadline is not None and time.time() > deadline - 5:
                break
            
            # Create new matrices without this column
            mask = np.ones(current_rank, dtype=bool)
            mask[col_to_remove] = False
            
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            
            # Check if still valid
            if _validate_decomposition(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p, current_rank - 1):
                U, V, W = U_new, V_new, W_new
                current_rank -= 1
                best_rank = current_rank
                best_decomp = (U.tolist(), V.tolist(), W.tolist(), current_rank)
                found_reduction = True
                break
        
        if not found_reduction:
            break
    
    return best_rank, best_decomp

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 @ 3x5 matrix multiplication.
    Uses schoolbook as baseline and tries to reduce rank.
    """
    n, m, p = 3, 3, 5
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Start with schoolbook
    U, V, W, current_rank = _schoolbook_decomposition(n, m, p)
    best_rank = current_rank
    best_decomp = (U, V, W, current_rank)
    
    # Try greedy rank reduction
    if time.time() < deadline - 10:
        rank, decomp = _greedy_rank_reduction(n, m, p, current_rank, max_iterations=200, deadline=deadline)
        if rank < best_rank:
            best_rank = rank
            best_decomp = decomp
    
    # Try random search for lower ranks
    if time.time() < deadline - 5:
        for target_rank in range(best_rank - 1, max(best_rank - 10, 1), -1):
            if time.time() > deadline - 5:
                break
            rank, decomp = _try_random_rank_reduction(n, m, p, target_rank, max_iterations=100, deadline=deadline)
            if rank < best_rank:
                best_rank = rank
                best_decomp = decomp
            if best_rank <= n * m * p // 2:
                break
    
    # Return the best decomposition found
    U, V, W, rank = best_decomp
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
