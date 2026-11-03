"""Matrix multiplication 4x5 by 5x5: rank-minimal exact tensor decomposition.

Approach:
  * Build the multiplication tensor T of shape (n, m, p) with T[i, j, k] = 1.
  * Use structured rank-1 peeling with systematic direction sampling.
  * For each rank-1 component, try structured directions (basis vectors, 
    simple integer combinations) before falling back to random directions.
  * Use the tensor's algebraic structure to guide the peeling process.
  * Deadline guard: return the best valid solution before the wall-clock budget.

Return:  {'U': list[list[int/float]], 'V': ..., 'W': ...}
          shapes (n*m, R), (m*p, R), (p*n, R)
"""

import time
import numpy as np
from itertools import product

# Budget (evaluator kills at 150 s; keep ~40 s margin for the harness).
DEADLINE_S = 110.0

DIMENSIONS = [4, 5, 5]


def _build_tensor_3way(n, m, p):
    """Build the 3-way tensor T[i, j, k] = 1."""
    T = np.ones((n, m, p), dtype=np.float64)
    return T


def _validate_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Validate that the rank-R decomposition exactly reconstructs the multiplication tensor.
    
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all (i,j,k)
    """
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _schoolbook(n, m, p):
    """Return the schoolbook rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


def _generate_structured_directions(dim, max_simple=5):
    """Generate structured directions: basis vectors and simple integer combinations."""
    dirs = []
    # Basis vectors
    for i in range(dim):
        e = np.zeros(dim)
        e[i] = 1.0
        dirs.append(e)
    
    # Simple integer combinations with small coefficients
    for coeff in product([-max_simple, -1, 0, 1, max_simple], repeat=dim):
        if all(c == 0 for c in coeff):
            continue
        # Normalize to avoid very large values
        norm = np.sqrt(sum(c*c for c in coeff))
        if norm < 1e-10:
            continue
        v = np.array(coeff, dtype=np.float64) / norm
        # Only add if not too similar to existing directions
        is_new = True
        for existing in dirs:
            if np.dot(v, existing) > 0.95:
                is_new = False
                break
        if is_new:
            dirs.append(v)
    
    return dirs


def _try_peel_rank1(residual, n, m, p, structured_dirs_u, structured_dirs_v, 
                    max_random=20, tol=1e-10):
    """Try to peel a rank-1 component from residual tensor.
    
    Returns (u, v, w, new_residual) if successful, else None.
    """
    best_reduction = 0.0
    best_result = None
    
    old_norm = np.sum(np.abs(residual))
    
    # Try structured directions first
    for u in structured_dirs_u:
        for v in structured_dirs_v:
            denom = np.sum(u**2) * np.sum(v**2)
            if abs(denom) < 1e-12:
                continue
            
            # Compute optimal w
            A = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    A[i, k] = np.dot(residual[i, :, k], v)
            
            numerator = np.dot(u, A)
            w = numerator / denom
            
            peeled = np.einsum('i,j,k->ijk', u, v, w)
            new_residual = residual - peeled
            new_norm = np.sum(np.abs(new_residual))
            
            reduction = old_norm - new_norm
            if reduction > best_reduction:
                best_reduction = reduction
                best_result = (u.copy(), v.copy(), w.copy(), new_residual)
    
    # Try random directions if structured didn't give good result
    if best_reduction < old_norm * 0.5:
        for _ in range(max_random):
            u = np.random.randn(n)
            u = u / np.linalg.norm(u)
            v = np.random.randn(m)
            v = v / np.linalg.norm(v)
            
            denom = np.sum(u**2) * np.sum(v**2)
            if abs(denom) < 1e-12:
                continue
            
            A = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    A[i, k] = np.dot(residual[i, :, k], v)
            
            numerator = np.dot(u, A)
            w = numerator / denom
            
            peeled = np.einsum('i,j,k->ijk', u, v, w)
            new_residual = residual - peeled
            new_norm = np.sum(np.abs(new_residual))
            
            reduction = old_norm - new_norm
            if reduction > best_reduction:
                best_reduction = reduction
                best_result = (u.copy(), v.copy(), w.copy(), new_residual)
    
    # Require significant reduction
    if best_result is not None and best_reduction > old_norm * 0.3:
        return best_result
    
    return None


def _try_rank_decomposition(T, n, m, p, target_rank, max_restarts=30, tol=1e-8):
    """Try to find a rank-target_rank decomposition of T.
    
    Uses iterative peeling with structured + random direction sampling.
    """
    t_start = time.time()
    
    # Pre-generate structured directions
    structured_u = _generate_structured_directions(n, max_simple=3)
    structured_v = _generate_structured_directions(m, max_simple=3)
    
    for restart in range(max_restarts):
        if time.time() - t_start > DEADLINE_S - 25:
            break
        
        residual = T.copy()
        U_cols = []
        V_cols = []
        W_cols = []
        success = True
        
        for r in range(target_rank):
            if time.time() - t_start > DEADLINE_S - 20:
                success = False
                break
            
            result = _try_peel_rank1(residual, n, m, p, 
                                      structured_u, structured_v,
                                      max_random=15, tol=tol)
            
            if result is None:
                success = False
                break
            
            u, v, w, new_residual = result
            residual = new_residual
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
        
        if not success or len(U_cols) != target_rank:
            continue
        
        # Build full factor matrices
        U_full = np.zeros((n * m, target_rank))
        V_full = np.zeros((m * p, target_rank))
        W_full = np.zeros((p * n, target_rank))
        
        for r in range(target_rank):
            u_r = U_cols[r]
            v_r = V_cols[r]
            w_r = W_cols[r]
            
            for i in range(n):
                for j in range(m):
                    U_full[i * m + j, r] = u_r[i]
            
            for j in range(m):
                for k in range(p):
                    V_full[j * p + k, r] = v_r[j]
            
            for k in range(p):
                for i in range(n):
                    W_full[k * n + i, r] = w_r[k]
        
        if _validate_decomposition(U_full, V_full, W_full, n, m, p, tol):
            return (U_full, V_full, W_full)
    
    return None


def _try_sparse_linear_system(n, m, p, target_rank, t_start):
    """Try to solve for a rank-target_rank decomposition using sparse linear system.
    
    For a given rank R, we have R * (n + m + p) unknowns (the entries of u_r, v_r, w_r).
    The constraints are: for all (i,j,k), sum_r u_r[i]*v_r[j]*w_r[k] = 1.
    This is a system of n*m*p nonlinear equations, but we can linearize by fixing
    two of the three factor matrices.
    
    Strategy: Fix U and V, then solve for W linearly.
    """
    if time.time() - t_start > DEADLINE_S - 15:
        return None
    
    # Try a structured initialization
    # Use the first R basis-like directions
    U_init = np.zeros((n * m, target_rank))
    V_init = np.zeros((m * p, target_rank))
    
    # Initialize with structured patterns
    for r in range(target_rank):
        # Spread the rank across different (i,j) and (j,k) pairs
        idx_u = r % (n * m)
        idx_v = r % (m * p)
        U_init[idx_u, r] = 1.0
        V_init[idx_v, r] = 1.0
    
    # For fixed U and V, solve for W:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    # This is linear in W!
    
    # Build the linear system A * W_flat = b
    # W_flat is of length p*n (all entries of W)
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # But W[k*n+i, r] depends on r, so we need to be careful.
    
    # Actually, W is (p*n, R), so W_flat would be of length p*n*R
    # But the equation couples different r's through the product U*V*W
    
    # This is still nonlinear in general. Let's try a different approach.
    
    # Alternative: Use the fact that for small dimensions, we can try to find
    # a decomposition where many entries are zero (sparse).
    
    # For now, this is a placeholder for a more sophisticated approach.
    return None


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
    # Start with the guaranteed schoolbook decomposition.
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build the 3-way tensor
    T = _build_tensor_3way(n, m, p)
    
    # Try to find lower-rank decompositions
    # For 4x5x5, schoolbook is 100. The theoretical lower bound is around 4*5*5/2 = 50
    # Try ranks from 95 down to 55
    
    for target_rank in range(best_rank - 5, max(best_rank - 45, 50), -1):
        if time.time() - t0 > DEADLINE_S - 25:
            break
        
        # Try the peeling approach
        result = _try_rank_decomposition(T, n, m, p, target_rank, max_restarts=25, tol=1e-8)
        
        if result is not None:
            U_new, V_new, W_new = result
            best_rank = target_rank
            best_result = {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
            break  # Found a valid decomposition at this rank
    
    # Final polish: try to verify and clean up
    def _to_list(M):
        out = []
        for row in M:
            lrow = []
            for x in row:
                xv = float(x)
                if abs(xv - round(xv)) < 1e-6:
                    xv = int(round(xv))
                elif abs(xv * 2 - round(xv * 2)) < 1e-6:
                    xv = int(round(xv * 2)) / 2
                lrow.append(xv)
            out.append(lrow)
        return out
    
    return {
        "U": _to_list(np.array(best_result["U"])),
        "V": _to_list(np.array(best_result["V"])),
        "W": _to_list(np.array(best_result["W"]))
    }
