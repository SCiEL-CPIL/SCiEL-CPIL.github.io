"""Combined approach: structured rank-44 base + search for lower rank via term elimination."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _build_target_tensor(n, m, p):
    """Return the 3D tensor T[i,j,k] = 1."""
    return np.ones((n, m, p), dtype=np.float64)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if U,V,W exactly decompose the tensor (within tol)."""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)

def _construct_structured_decomposition():
    """Construct a rank-44 decomposition using Strassen for 2x2x2 blocks and schoolbook for 1x2x2 blocks."""
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    def add_strassen(rA, cA, rB, cB):
        nonlocal col
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+1), col] = 1
        W[(cB+1)*n + (rA+1), col] = -1
        col += 1
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = -1
        W[(cB+1)*n + (rA+0), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = -1
        W[(cB+0)*n + (rA+0), col] = 1
        W[(cB+0)*n + (rA+1), col] = 1
        col += 1
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1
        W[(cB+0)*n + (rA+0), col] = -1
        col += 1
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+0), col] = -1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        U[(rA+0)*m + (cA+1), col] = 1
        U[(rA+1)*m + (cA+1), col] = -1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        col += 1

    def add_schoolbook_1x2x2(rA, cA, rB, cB):
        nonlocal col
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        col += 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        col += 1
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1
        col += 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1
        col += 1

    add_strassen(0, 0, 0, 0)
    add_strassen(0, 2, 2, 0)
    add_strassen(0, 0, 0, 2)
    add_strassen(0, 2, 2, 2)
    add_schoolbook_1x2x2(2, 0, 0, 0)
    add_schoolbook_1x2x2(2, 2, 2, 0)
    add_schoolbook_1x2x2(2, 0, 0, 2)
    add_schoolbook_1x2x2(2, 2, 2, 2)
    
    return U, V, W

def _try_eliminate_terms(U, V, W, n, m, p, max_eliminate=4, time_budget=60):
    """Try to eliminate terms by solving for dependencies."""
    start_time = time.time()
    rank = U.shape[1]
    U, V, W = U.copy(), V.copy(), W.copy()
    
    # Try to find a term that can be expressed as a combination of others
    # For each term r, check if removing it can be compensated by adjusting others
    for r_to_remove in range(rank):
        if time.time() - start_time > time_budget:
            break
        if rank - 1 < 30:  # Don't go too low
            break
            
        # Remove term r_to_remove
        U_new = np.delete(U, r_to_remove, axis=1)
        V_new = np.delete(V, r_to_remove, axis=1)
        W_new = np.delete(W, r_to_remove, axis=1)
        
        # The deficit is the outer product of the removed term
        deficit = np.einsum('ij,jk,ki->ijk', 
                           U[:, r_to_remove].reshape(n, m),
                           V[:, r_to_remove].reshape(m, p),
                           W[:, r_to_remove].reshape(p, n))
        
        # Try to absorb the deficit into the remaining terms
        # This is a linear system: find delta_r such that sum_r delta_r * (U_r ⊗ V_r ⊗ W_r) = deficit
        # This is complex, so instead try a simpler approach: 
        # Check if the deficit is in the span of the remaining terms' outer products
        
        # Flatten the deficit
        deficit_flat = deficit.flatten()
        
        # Build the matrix of remaining terms' outer products
        # Each term r contributes U_r ⊗ V_r ⊗ W_r flattened
        # This is a (n*m*p) x (rank-1) matrix
        n_entries = n * m * p
        A_mat = np.zeros((n_entries, rank - 1))
        for new_r in range(rank - 1):
            orig_r = new_r if new_r < r_to_remove else new_r + 1
            outer = np.einsum('ij,jk,ki->ijk',
                             U[:, orig_r].reshape(n, m),
                             V[:, orig_r].reshape(m, p),
                             W[:, orig_r].reshape(p, n))
            A_mat[:, new_r] = outer.flatten()
        
        # Solve A_mat @ delta = deficit_flat
        try:
            delta, residuals, rank_mat, sv = np.linalg.lstsq(A_mat, deficit_flat, rcond=None)
            # Check if the solution is good
            if np.allclose(A_mat @ delta, deficit_flat, atol=1e-8):
                # Check if delta entries are "nice" (small integers or half-integers)
                if np.allclose(delta * 2, np.round(delta * 2), atol=1e-6):
                    # Apply the delta
                    for new_r in range(rank - 1):
                        orig_r = new_r if new_r < r_to_remove else new_r + 1
                        U_new[:, new_r] += delta[new_r] * U[:, orig_r]
                        V_new[:, new_r] += delta[new_r] * V[:, orig_r]
                        W_new[:, new_r] += delta[new_r] * W[:, orig_r]
                    
                    # Verify
                    if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-6):
                        return U_new, V_new, W_new
        except:
            pass
    
    return U, V, W

def _try_rank_search(U, V, W, n, m, p, target_rank, time_budget=30):
    """Try to find a decomposition of the given target rank."""
    start_time = time.time()
    current_rank = U.shape[1]
    
    if target_rank >= current_rank:
        return U, V, W
    
    # Try to eliminate (current_rank - target_rank) terms
    U_cur, V_cur, W_cur = U.copy(), V.copy(), W.copy()
    
    for _ in range(current_rank - target_rank):
        if time.time() - start_time > time_budget:
            break
        U_cur, V_cur, W_cur = _try_eliminate_terms(U_cur, V_cur, W_cur, n, m, p, time_budget=10)
        if U_cur.shape[1] == current_rank:
            break
        current_rank = U_cur.shape[1]
    
    if U_cur.shape[1] <= target_rank:
        return U_cur, V_cur, W_cur
    return U, V, W

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Start with the structured rank-44 decomposition
    U44, V44, W44 = _construct_structured_decomposition()
    best_U, best_V, best_W = U44.copy(), V44.copy(), W44.copy()
    best_rank = 44
    
    # Try to reduce rank
    for target_rank in [43, 42, 41, 40]:
        if time.time() > deadline - 30:
            break
        U_try, V_try, W_try = _try_rank_search(U44, V44, W44, n, m, p, target_rank, time_budget=20)
        if U_try.shape[1] < best_rank:
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = U_try.shape[1]
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
