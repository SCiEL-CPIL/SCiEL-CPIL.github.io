"""Matrix multiplication 2x3x4: rank-minimal exact tensor decomposition."""
import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i, j, k] = 1 for all i,j,k
    in the appropriate ranges, indexed as a 3D array of shape (n*m, m*p, p*n)
    where the trace convention is W[k*n+i, r].
    
    The tensor T has entries T[a, b, c] where:
      a = i*m + j  (row of U, i in [0,n), j in [0,m))
      b = j*p + k  (row of V, j in [0,m), k in [0,p))
      c = k*n + i  (row of W, k in [0,p), i in [0,n))
    
    T[a, b, c] = 1 iff a, b, c are consistent (i.e., the j in a equals the j in b,
    and the k in b equals the k in c, and the i in c equals the i in a).
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _reconstruct(U, V, W):
    """Reconstruct the tensor from factor matrices.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    rank = U.shape[1]
    # Use einsum for efficiency
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    return np.einsum('ar,br,cr->abc', U, V, W)


def _error(U, V, W, T):
    """Compute Frobenius norm of the error."""
    R = _reconstruct(U, V, W)
    return np.linalg.norm(R - T)


def _snap_to_half_integer(U, V, W, tol=1e-6):
    """Try to snap entries to nearest half-integer. Returns True if successful."""
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    # Check if the snapped version still gives a valid decomposition
    # We need to verify against the target tensor
    # But we don't have T here, so we check consistency
    # Actually, let's just return the snapped values; validation happens in construct
    return U_snap, V_snap, W_snap


def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    Returns True if valid."""
    rank = U.shape[1]
    # Check that for each (i, j, k), the sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for inconsistent indices, it should be 0.
    
    # Build target
    T = _build_target_tensor(n, m, p)
    R = _reconstruct(U, V, W)
    
    # Check max absolute error
    max_err = np.max(np.abs(R - T))
    return max_err < tol


def _als_step(U, V, W, T, n, m, p):
    """One step of ALS: fix two, solve for the third.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    T: (n*m, m*p, p*n)
    """
    n_m = n * m
    m_p = m * p
    p_n = p * n
    rank = U.shape[1]
    
    # Solve for U: U[a, :] = argmin sum_{b,c} (T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
    # For each a: U[a, :] = (sum_{b,c} T[a,b,c] V[b,r] W[c,r]) / (sum_{b,c} V[b,r]^2 W[c,r]^2)
    # More precisely, for fixed V, W:
    # U[a, r] = sum_{b,c} T[a,b,c] V[b,r] W[c,r] / sum_{b,c} V[b,r]^2 W[c,r]^2
    
    # Efficient: for each r, U[:, r] = (T * (V[:, r] * W[:, r])) / (||V[:, r] * W[:, r]||^2)
    # where T * (V[:, r] * W[:, r]) means element-wise multiply then sum over b, c
    
    # U[a, r] = sum_{b,c} T[a,b,c] * V[b,r] * W[c,r] / sum_{b,c} V[b,r]^2 * W[c,r]^2
    # = sum_b (sum_c T[a,b,c] W[c,r]) V[b,r] / sum_b V[b,r]^2 sum_c W[c,r]^2
    
    # Let's compute: for each r,
    # num[a, r] = sum_{b,c} T[a,b,c] V[b,r] W[c,r]
    # den[r] = sum_{b,c} V[b,r]^2 W[c,r]^2
    
    # num[a,r] = einsum('abc,br,cr->ar', T, V, W)
    # den[r] = einsum('br,cr->r', V*V, W*W)
    
    num_U = np.einsum('abc,br,cr->ar', T, V, W)
    den_U = np.einsum('br,cr->r', V**2, W**2)
    den_U[den_U == 0] = 1e-12
    U_new = num_U / den_U[None, :]
    
    # Similarly for V:
    # V[b, r] = sum_{a,c} T[a,b,c] U[a,r] W[c,r] / sum_{a,c} U[a,r]^2 W[c,r]^2
    num_V = np.einsum('abc,ar,cr->br', T, U_new, W)
    den_V = np.einsum('ar,cr->r', U_new**2, W**2)
    den_V[den_V == 0] = 1e-12
    V_new = num_V / den_V[None, :]
    
    # Similarly for W:
    # W[c, r] = sum_{a,b} T[a,b,c] U[a,r] V[b,r] / sum_{a,b} U[a,r]^2 V[b,r]^2
    num_W = np.einsum('abc,ar,br->cr', T, U_new, V_new)
    den_W = np.einsum('ar,br->r', U_new**2, V_new**2)
    den_W[den_W == 0] = 1e-12
    W_new = num_W / den_W[None, :]
    
    return U_new, V_new, W_new


def _als_optimize(U, V, W, T, n, m, p, max_iters=2000, tol=1e-12, deadline=None):
    """Run ALS optimization. Returns (U, V, W, converged)."""
    U_curr = U.copy()
    V_curr = V.copy()
    W_curr = W.copy()
    
    prev_err = _error(U_curr, V_curr, W_curr, T)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        U_curr, V_curr, W_curr = _als_step(U_curr, V_curr, W_curr, T, n, m, p)
        
        err = _error(U_curr, V_curr, W_curr, T)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    converged = _verify_decomposition(U_curr, V_curr, W_curr, n, m, p, tol=1e-8)
    return U_curr, V_curr, W_curr, converged


def _random_init(n, m, p, rank, seed=None):
    """Generate random initialization for factor matrices."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    return U, V, W


def _structured_init(n, m, p, rank):
    """Generate a structured initialization based on the schoolbook decomposition
    but with some perturbation to allow ALS to find a better solution."""
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    U = np.zeros((n_m, rank))
    V = np.zeros((m_p, rank))
    W = np.zeros((p_n, rank))
    
    # Start with schoolbook-like pattern but spread across rank
    # For each (i, j, k), assign to a random column
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = (i * m + j) * p + k  # This gives n*m*p columns, but we only have rank
                if c < rank:
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
    
    # Add some random perturbation to remaining columns
    for c in range(n * m * p, rank):
        U[:, c] = np.random.randn(n_m) * 0.1
        V[:, c] = np.random.randn(m_p) * 0.1
        W[:, c] = np.random.randn(p_n) * 0.1
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    T = _build_target_tensor(n, m, p)
    
    # Known lower bound: max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # But for matrix multiplication, the tensor rank is known to be less than n*m*p.
    # For 2x3x4, let's try ranks from 12 upward.
    
    best_result = None
    best_rank = n * m * p  # Schoolbook
    
    # First, ensure we have the schoolbook solution as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_result = (U_school, V_school, W_school)
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Based on knowledge, ranks 18-21 have been found. Let's try from 12 upward.
    for rank in range(12, rank_school + 1):
        if time.time() > deadline:
            break
        
        found = False
        
        # Multiple random restarts for each rank
        n_restarts = 50 if rank <= 20 else 30
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            seed = restart * 7919 + rank * 31
            U, V, W = _random_init(n, m, p, rank, seed=seed)
            
            # Run ALS
            U, V, W, converged = _als_optimize(U, V, W, T, n, m, p, 
                                                  max_iters=1000, tol=1e-14,
                                                  deadline=deadline)
            
            if converged:
                # Snap to half-integer
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-10):
                    best_result = (U_snap, V_snap, W_snap)
                    best_rank = rank
                    found = True
                    break
                else:
                    # Try without snapping
                    if _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
                        best_result = (U, V, W)
                        best_rank = rank
                        found = True
                        break
        
        if found:
            # Found a solution at this rank; try one lower rank
            continue
        else:
            # Didn't find at this rank, but maybe higher ranks will work
            # Continue trying
            pass
    
    # Also try structured initialization for a few ranks
    for rank in range(12, min(best_rank + 1, 25)):
        if time.time() > deadline:
            break
        if rank >= best_rank:
            break
        
        U, V, W = _structured_init(n, m, p, rank)
        U, V, W, converged = _als_optimize(U, V, W, T, n, m, p,
                                              max_iters=1000, tol=1e-14,
                                              deadline=deadline)
        if converged:
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-10):
                best_result = (U_snap, V_snap, W_snap)
                best_rank = rank
    
    U_final, V_final, W_final = best_result
    
    # Convert to lists of lists as expected by the interface
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
