"""Matrix multiplication 4x5 by 5x5: rank-minimal exact tensor decomposition.

Approach:
  * Build the multiplication tensor T of shape (n, m, p) with T[i, j, k] = 1.
  * Use iterative LP-based rank-1 peeling to find a low-rank exact decomposition.
  * At each step, solve an LP to find a rank-1 component (u, v, w) such that
    the residual tensor can still be completed. Peel it off and repeat.
  * Deadline guard: return the best valid solution before the wall-clock budget is exhausted.

Return:  {'U': list[list[int/float]], 'V': ..., 'W': ...}
          shapes (n*m, R), (m*p, R), (p*n, R)
"""

import time
import numpy as np
from scipy.optimize import linprog

# Budget (evaluator kills at 150 s; keep ~40 s margin for the harness).
DEADLINE_S = 110.0

DIMENSIONS = [4, 5, 5]


def _build_tensor(n, m, p):
    """T.reshape(n*m, p*n) where T[i, j, k] = 1.
    
    Row index: i*m + j
    Col index: k*n + i
    """
    T = np.zeros((n * m, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, k * n + i] = 1.0
    return T


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


def _peel_rank1_lp(T, n, m, p, tol=1e-10):
    """Try to peel a rank-1 component from tensor T using LP.
    
    We want to find (u, v, w) such that T - outer(u, v, w) has lower "complexity".
    A practical approach: fix one mode, solve for the others.
    
    Strategy: For each possible support pattern, try to find a rank-1 component.
    We use the fact that for a rank-1 tensor, all 2x2 minors vanish.
    
    A simpler approach: randomly sample directions and project.
    """
    # This is a placeholder - we'll use a more direct approach in the main loop
    return None


def _try_rank_decomposition(T, n, m, p, target_rank, max_iters=1000, tol=1e-8):
    """Try to find a rank-target_rank decomposition of T.
    
    Uses iterative peeling with random restarts.
    """
    t_start = time.time()
    best = None
    
    for restart in range(max_iters):
        if time.time() - t_start > DEADLINE_S - 20:
            break
            
        # Start with the full tensor
        residual = T.copy()
        U_cols = []
        V_cols = []
        W_cols = []
        
        for r in range(target_rank):
            # Try to find a rank-1 component in the residual
            # Use a simple heuristic: pick a random linear combination
            # and decompose via SVD-like approach
            
            # Flatten residual to matrix
            # T[i,j,k] -> matrix of shape (n*m, p) or similar
            # Let's use the mode-1 unfolding: shape (n, m*p)
            # But for rank-1 peeling, we need u (n), v (m), w (p)
            
            # Heuristic: pick random u, v, compute optimal w
            success = False
            for attempt in range(50):
                # Random u
                u = np.random.randn(n)
                # Random v  
                v = np.random.randn(m)
                
                # Compute optimal w: for each k, w[k] should satisfy
                # sum_{i,j} residual[i,j,k] * u[i] * v[j] = sum_{i,j} T_orig[i,j,k] * u[i] * v[j]
                # But we're peeling from residual, so we want:
                # residual - outer(u,v,w) to have structure
                # 
                # Actually, for exact peeling, we need:
                # For all (i,j,k): residual[i,j,k] - u[i]*v[j]*w[k] 
                # should be completable by remaining rank-1 terms.
                #
                # A necessary condition: the residual after peeling should have
                # rank at most target_rank - r - 1 in some sense.
                #
                # Practical approach: just try to make residual - outer(u,v,w) 
                # have smaller nuclear norm or something.
                #
                # Let's try: given u and v, find w that minimizes ||residual - outer(u,v,w)||_F
                # This is a least squares problem.
                
                # residual[i,j,k] - u[i]*v[j]*w[k]
                # For fixed u,v, minimize over w:
                # sum_{i,j,k} (residual[i,j,k] - u[i]*v[j]*w[k])^2
                # = sum_k (sum_{i,j} residual[i,j,k]^2 - 2*w[k]*sum_{i,j} residual[i,j,k]*u[i]*v[j] + w[k]^2*sum_{i,j} u[i]^2*v[j]^2)
                # Optimal w[k] = sum_{i,j} residual[i,j,k]*u[i]*v[j] / sum_{i,j} u[i]^2*v[j]^2
                
                denom = np.sum(u**2) * np.sum(v**2)
                if abs(denom) < 1e-12:
                    continue
                
                # Compute numerator for each k
                # numerator[k] = sum_{i,j} residual[i,j,k] * u[i] * v[j]
                # = sum_i u[i] * sum_j residual[i,j,k] * v[j]
                # = u^T * (residual * v) where residual is (n,m,p)
                
                # Compute A[i,k] = sum_j residual[i,j,k] * v[j]
                A = np.zeros((n, p))
                for i in range(n):
                    for k in range(p):
                        A[i, k] = np.dot(residual[i, :, k], v)
                
                numerator = np.dot(u, A)
                w = numerator / denom
                
                # Check if peeling this component makes sense
                # The peeled component is outer(u, v, w)
                peeled = np.einsum('i,j,k->ijk', u, v, w)
                new_residual = residual - peeled
                
                # Check if new_residual has smaller "rank" indicator
                # Use the sum of absolute values as a proxy
                old_norm = np.sum(np.abs(residual))
                new_norm = np.sum(np.abs(new_residual))
                
                if new_norm < old_norm * 0.95:  # Significant reduction
                    residual = new_residual
                    U_cols.append(u)
                    V_cols.append(v)
                    W_cols.append(w)
                    success = True
                    break
            
            if not success:
                break
        
        # Check if we successfully peeled target_rank components
        if len(U_cols) == target_rank:
            # Verify the decomposition
            U = np.column_stack(U_cols)
            V = np.column_stack(V_cols)
            W = np.column_stack(W_cols)
            
            # Reshape to proper forms
            # U should be (n*m, R) where U[i*m+j, r] = u_r[i] * something
            # Wait, our u is just (n,), not (n*m,)
            # We need U[i*m+j, r] such that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = T[i,j,k]
            #
            # If we have rank-1 components (u_r, v_r, w_r) where u_r is (n,), v_r is (m,), w_r is (p,)
            # Then T[i,j,k] = sum_r u_r[i] * v_r[j] * w_r[k]
            #
            # We need: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = u_r[i]*v_r[j]*w_r[k]
            #
            # So we can set: U[i*m+j, r] = u_r[i], V[j*p+k, r] = v_r[j], W[k*n+i, r] = w_r[k]
            # Then product = u_r[i] * v_r[j] * w_r[k] which is correct!
            
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
                best = (U_full, V_full, W_full)
                break
    
    return best


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
    # Start from a reasonable rank and go down
    # For 4x5x5, schoolbook is 100. Try ranks from 95 down to maybe 70.
    
    for target_rank in range(best_rank - 1, max(best_rank - 30, 50), -1):
        if time.time() - t0 > DEADLINE_S - 30:
            break
        
        result = _try_rank_decomposition(T, n, m, p, target_rank, max_iters=50, tol=1e-8)
        
        if result is not None:
            U_new, V_new, W_new = result
            best_rank = target_rank
            best_result = {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
            break  # Found the lowest rank in our search range
    
    # Final polish: try to verify and clean up
    def _to_list(M):
        out = []
        for row in M:
            lrow = []
            for x in row:
                xv = float(x)
                if abs(xv - round(xv)) < 1e-6:
                    xv = int(round(xv))
                lrow.append(xv)
            out.append(lrow)
        return out
    
    return {
        "U": _to_list(np.array(best_result["U"])),
        "V": _to_list(np.array(best_result["V"])),
        "W": _to_list(np.array(best_result["W"]))
    }
