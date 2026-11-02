import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 7]
TIME_LIMIT = 120  # seconds
SAFETY_MARGIN = 5  # seconds

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Initialize Schoolbook
    U = np.zeros((n * m, rank_schoolbook), dtype=float)
    V = np.zeros((m * p, rank_schoolbook), dtype=float)
    W = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_schoolbook
    
    # Build the target tensor
    target = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
    
    def verify(U_mat, V_mat, W_mat):
        """Verify that the decomposition exactly matches the target tensor."""
        if U_mat.shape[1] != V_mat.shape[1] or V_mat.shape[1] != W_mat.shape[1]:
            return False
        r = U_mat.shape[1]
        reconstructed = np.zeros((n * m, m * p, p * n), dtype=float)
        for t in range(r):
            reconstructed += np.einsum('a,b,c->abc', U_mat[:, t], V_mat[:, t], W_mat[:, t])
        return np.allclose(reconstructed, target, atol=1e-10)
    
    # Try to find lower rank decompositions by removing columns
    # and compensating with other rank-1 terms
    def try_remove_columns(U_mat, V_mat, W_mat, max_remove=5):
        """Try to remove columns and find compensating rank-1 terms."""
        r = U_mat.shape[1]
        current_U, current_V, current_W = U_mat.copy(), V_mat.copy(), W_mat.copy()
        current_rank = r
        
        for attempt in range(max_remove):
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            found = False
            # Try removing each column
            for col_idx in range(current_rank):
                if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                    break
                    
                # Remove this column
                test_U = np.delete(current_U, col_idx, axis=1)
                test_V = np.delete(current_V, col_idx, axis=1)
                test_W = np.delete(current_W, col_idx, axis=1)
                
                # Compute what's missing
                missing = target.copy()
                for t in range(test_U.shape[1]):
                    missing -= np.einsum('a,b,c->abc', test_U[:, t], test_V[:, t], test_W[:, t])
                
                # Check if missing can be expressed as a single rank-1 term
                # with small integer coefficients
                if _is_sparse_rank1(missing):
                    # Find the rank-1 decomposition of missing
                    u, v, w = _decompose_sparse_rank1(missing)
                    if u is not None:
                        new_U = np.column_stack([test_U, u]) if test_U.size > 0 else u.reshape(-1, 1)
                        new_V = np.column_stack([test_V, v]) if test_V.size > 0 else v.reshape(-1, 1)
                        new_W = np.column_stack([test_W, w]) if test_W.size > 0 else w.reshape(-1, 1)
                        
                        if verify(new_U, new_V, new_W):
                            current_U, current_V, current_W = new_U, new_V, new_W
                            current_rank = new_U.shape[1]
                            found = True
                            break
            
            if not found:
                break
        
        return current_U, current_V, current_W, current_rank
    
    def _is_sparse_rank1(tensor):
        """Check if tensor is a sparse rank-1 tensor with small integer/half-integer entries."""
        # A rank-1 tensor has form u ⊗ v ⊗ w
        # Check if all non-zero entries have the same sign pattern
        nonzeros = tensor[tensor != 0]
        if len(nonzeros) == 0:
            return True
        if len(nonzeros) > 100:  # Too many non-zeros for sparse
            return False
        
        # Check if values are small integers or half-integers
        for val in nonzeros:
            if not np.isclose(val * 2, np.round(val * 2), atol=1e-10):
                return False
            if abs(val) > 2:
                return False
        return True
    
    def _decompose_sparse_rank1(tensor):
        """Decompose a sparse rank-1 tensor into u, v, w vectors."""
        # Find the support
        nz_idx = np.argwhere(tensor != 0)
        if len(nz_idx) == 0:
            return None, None, None
        
        # Get the unique values
        vals = tensor[nz_idx]
        if len(set(np.round(vals, 10))) != 1:
            return None, None, None
        
        val = vals[0]
        
        # For a rank-1 tensor u⊗v⊗w with all non-zero entries equal to val,
        # we need u[i]*v[j]*w[k] = val for all (i,j,k) in support
        # This means u[i], v[j], w[k] must be constant on their supports
        
        # Try to find u, v, w
        indices_a = set(nz_idx[:, 0])
        indices_b = set(nz_idx[:, 1])
        indices_c = set(nz_idx[:, 2])
        
        # Simple case: try u[i] = val, v[j] = 1, w[k] = 1
        u = np.zeros(n * m)
        v = np.zeros(m * p)
        w = np.zeros(p * n)
        
        for idx in indices_a:
            u[idx] = val
        for idx in indices_b:
            v[idx] = 1
        for idx in indices_c:
            w[idx] = 1
        
        # Verify
        reconstructed = np.einsum('a,b,c->abc', u, v, w)
        if np.allclose(reconstructed, tensor, atol=1e-10):
            return u, v, w
        
        return None, None, None
    
    # Try the elimination approach
    U_elim, V_elim, W_elim, rank_elim = try_remove_columns(U, V, W, max_remove=3)
    
    if rank_elim < best_rank and verify(U_elim, V_elim, W_elim):
        best_U, best_V, best_W = U_elim, V_elim, W_elim
        best_rank = rank_elim
    
    # Return the best result found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
