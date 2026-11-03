"""Rewritten with structural algebraic approach for 3x4x8 matrix multiplication tensor rank."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product, combinations
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    T = _build_tensor(n, m, p)
    rank = U.shape[1]
    recon = np.zeros_like(T)
    for c in range(rank):
        recon += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
    err = np.max(np.abs(recon - T))
    return err < tol, err

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
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
    return U, V, W

def _try_merge_terms(U, V, W, n, m, p, tol=1e-9):
    """
    Try to reduce rank by finding linearly dependent rank terms.
    If two rank terms c1 and c2 have proportional U, V, W columns,
    they can be merged. More generally, if the span of rank terms
    has dimension less than the number of terms, we can reduce.
    
    This is a heuristic: check if removing any single term and 
    re-projecting still gives an exact decomposition.
    """
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W
    
    # Try greedy removal: for each term, check if the remaining terms
    # can still span the tensor
    # This is expensive, so we do a simplified version:
    # Check if any two terms are linearly dependent
    
    # First, check for exact duplicates or proportional terms
    to_remove = set()
    for c in range(rank):
        if c in to_remove:
            continue
        for d in range(c+1, rank):
            if d in to_remove:
                continue
            # Check if (U[:,c], V[:,c], W[:,c]) is proportional to (U[:,d], V[:,d], W[:,d])
            # This means U[:,c] = a*U[:,d], V[:,c] = a*V[:,d], W[:,c] = a*W[:,d] for some scalar a
            # Find a from non-zero entries
            a_candidates = []
            for idx in range(U.shape[0]):
                if abs(U[idx, d]) > tol:
                    a_candidates.append(U[idx, c] / U[idx, d])
            if not a_candidates:
                continue
            a = a_candidates[0]
            # Check consistency
            if (np.max(np.abs(U[:, c] - a * U[:, d])) < tol and
                np.max(np.abs(V[:, c] - a * V[:, d])) < tol and
                np.max(np.abs(W[:, c] - a * W[:, d])) < tol):
                to_remove.add(d)
    
    if to_remove:
        keep = [c for c in range(rank) if c not in to_remove]
        U_new = U[:, keep]
        V_new = V[:, keep]
        W_new = W[:, keep]
        return U_new, V_new, W_new
    
    return U, V, W

def _rank_reduction_via_slicing(n, m, p, U, V, W, max_iterations=100, tol=1e-9):
    """
    Try to reduce rank by exploiting the structure.
    Use the fact that for matrix multiplication, certain slices have low rank.
    
    For the tensor T[i*m+j, j*p+k, k*n+i], fixing i and k gives:
    T[i*m+j, j*p+k, k*n+i] = 1 only when the indices are consistent.
    
    A different approach: use the border rank / approximate rank ideas,
    but we need exact rank.
    
    Let's try: if we can find a set of rank terms that are linearly
    dependent in a way that allows us to express the tensor with fewer terms.
    """
    # This is a placeholder for more sophisticated reduction
    # For now, just try the merge
    return _try_merge_terms(U, V, W, n, m, p, tol)

def _structured_decomposition_3x4x8(n, m, p):
    """
    Try to find a structured decomposition exploiting the 3x4x8 dimensions.
    
    Idea: Split the 4x4 inner product into blocks.
    A is 3x4, B is 4x8.
    
    We can write A = [A1, A2] where A1 is 3x2, A2 is 3x2
    B = [B1; B2] where B1 is 2x8, B2 is 2x8
    C = A1*B1 + A2*B2
    
    Each term A1*B1 is 3x2 * 2x8, which has schoolbook rank 3*2*8 = 48
    Similarly for A2*B2.
    Total: 96, same as schoolbook.
    
    But we can do better by sharing computation.
    
    Alternative: use the fact that 2x2 matrix multiplication has rank 7 (Strassen).
    But our blocks are 3x2 and 2x8, not square.
    
    Let me try a different split:
    A is 3x4, split into 2x2 blocks of size (3/2 is not integer)...
    
    Better: think of it as sum over j of A[:,j] * B[j,:]
    A[:,j] is 3x1, B[j,:] is 1x8
    Each outer product has rank 1, so total rank 4*3*8 = 96.
    
    To reduce rank, we need to find that some of these outer products
    can be combined.
    
    Key insight: The tensor T has a specific algebraic structure.
    Let's use the concept of "tensor rank" for matrix multiplication.
    
    For m x n * n x p, the rank is known to be at most m*n*p.
    For small cases, better bounds are known.
    
    For 3x4x8, let's try to construct a decomposition of rank < 96.
    """
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    
    # Try to reduce
    U, V, W = _try_merge_terms(U, V, W, n, m, p)
    
    return U, V, W

def _randomized_rank_search(n, m, p, target_rank, n_restarts=5, max_iters=2000, tol=1e-9):
    """
    Try to find a rank-`target_rank` decomposition using iterative refinement.
    Use a different optimization: alternating least squares on the tensor.
    """
    T = _build_tensor(n, m, p)
    rng = np.random.RandomState(42)
    
    best_U, best_V, best_W = None, None, None
    best_err = np.inf
    
    for restart in range(n_restarts):
        # Initialize with random vectors
        scale = 1.0 / np.sqrt(target_rank)
        U = rng.randn(n*m, target_rank) * scale
        V = rng.randn(m*p, target_rank) * scale
        W = rng.randn(p*n, target_rank) * scale
        
        for it in range(max_iters):
            # Alternating least squares
            # Fix V, W, solve for U:
            # T[i,:,:] = sum_c U[i,c] * V[:,c] outer W[:,c]
            # For each i, T[i,:,:] = sum_c U[i,c] * (V[:,c] outer W[:,c])
            # Let M_c = V[:,c] outer W[:,c], shape (m*p, p*n)
            # Then T[i,:,:] = sum_c U[i,c] * M_c
            # This is a linear system: T_flat[i] = U[i,:] @ M_flat
            # where M_flat has columns M_c flattened
            
            # Build M matrix: shape (m*p * p*n, target_rank)
            M = np.zeros((m*p * p*n, target_rank))
            for c in range(target_rank):
                M[c::target_rank] = np.outer(V[:,c], W[:,c]).flatten()
            
            # For each i, solve: T[i,:,:].flatten() = U[i,:] @ M
            # This is an overdetermined system, use least squares
            T_flat = T.reshape(n*m, -1)
            # U[i,:] = T_flat[i] @ M^T (M M^T)^{-1}
            # Use np.linalg.lstsq
            try:
                U_new, _, _, _ = np.linalg.lstsq(M, T_flat.T, rcond=None)
                U = U_new.T
            except:
                pass
            
            # Similarly for V and W
            # Fix U, W, solve for V:
            # T[:,j,:] = sum_c V[j,c] * (U[:,c] outer W[:,c])
            M_v = np.zeros((n*m * p*n, target_rank))
            for c in range(target_rank):
                M_v[c::target_rank] = np.outer(U[:,c], W[:,c]).flatten()
            T_flat_v = T.reshape(m*p, -1)
            try:
                V_new, _, _, _ = np.linalg.lstsq(M_v, T_flat_v.T, rcond=None)
                V = V_new.T
            except:
                pass
            
            # Fix U, V, solve for W:
            M_w = np.zeros((n*m * m*p, target_rank))
            for c in range(target_rank):
                M_w[c::target_rank] = np.outer(U[:,c], V[:,c]).flatten()
            T_flat_w = T.reshape(p*n, -1)
            try:
                W_new, _, _, _ = np.linalg.lstsq(M_w, T_flat_w.T, rcond=None)
                W = W_new.T
            except:
                pass
            
            # Check error
            if it % 100 == 0 or it == max_iters - 1:
                recon = np.zeros_like(T)
                for c in range(target_rank):
                    recon += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
                err = np.max(np.abs(recon - T))
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if err < tol:
                    break
        
        if best_err < tol:
            break
    
    if best_U is not None and best_err < tol:
        return best_U, best_V, best_W, best_err
    return None, None, None, best_err

def _snap_to_half_integers(U, V, W, tol=1e-6):
    """Snap values to half-integers (multiples of 0.5)."""
    U_s = np.round(U * 2) / 2
    V_s = np.round(V * 2) / 2
    W_s = np.round(W * 2) / 2
    return U_s, V_s, W_s

def _snap_to_integers(U, V, W, tol=1e-6):
    """Snap values to integers."""
    U_s = np.round(U)
    V_s = np.round(V)
    W_s = np.round(W)
    return U_s, V_s, W_s

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Build the target tensor
    T = _build_tensor(n, m, p)
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p  # 96
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank
    # First, try simple merging
    U, V, W = _try_merge_terms(U, V, W, n, m, p)
    if U.shape[1] < best_rank:
        best_rank = U.shape[1]
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try randomized search for lower ranks
    # Known lower bound: rank >= max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # Try ranks from best_rank-1 down to 32
    candidate_ranks = list(range(min(best_rank - 1, 60), 31, -1))
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 15:
            break
        
        U_c, V_c, W_c, err = _randomized_rank_search(
            n, m, p, target_rank, 
            n_restarts=3, 
            max_iters=500, 
            tol=1e-9
        )
        
        if U_c is not None and err < 1e-9:
            # Try to snap to nice values
            U_s, V_s, W_s = _snap_to_half_integers(U_c, V_c, W_c)
            valid, err_s = _verify_decomposition(U_s, V_s, W_s, n, m, p, tol=1e-9)
            if valid:
                best_rank = target_rank
                best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
                break
            else:
                # Use the non-snapped version
                best_rank = target_rank
                best_U, best_V, best_W = U_c.copy(), V_c.copy(), W_c.copy()
                break
    
    # Convert to lists
    U_list = [[float(x) for x in best_U[i]] for i in range(n*m)]
    V_list = [[float(x) for x in best_V[i]] for i in range(m*p)]
    W_list = [[float(x) for x in best_W[i]] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
