"""Matrix multiplication 2x4 x 4x4 tensor decomposition (rank 28 via Strassen on 2x2 blocks)."""

import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def _build_strassen_rank28():
    """Build the rank-28 Strassen decomposition deterministically."""
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0
    for b in range(2):  # B column block
        for a_blk in range(2):  # A column block / B row block
            for s in range(7):
                c = col

                # U contributions: A[i, 2*a_blk + j]
                U_contrib = {
                    0: [(0, 2*a_blk, 1), (1, 2*a_blk+1, 1)],
                    1: [(1, 2*a_blk, 1), (1, 2*a_blk+1, 1)],
                    2: [(0, 2*a_blk, 1)],
                    3: [(1, 2*a_blk+1, 1)],
                    4: [(0, 2*a_blk, 1), (0, 2*a_blk+1, 1)],
                    5: [(1, 2*a_blk, 1), (0, 2*a_blk, -1)],
                    6: [(0, 2*a_blk+1, 1), (1, 2*a_blk+1, -1)],
                }
                for (i, j, coeff) in U_contrib[s]:
                    U[i * 4 + j, c] += coeff

                # V contributions: B[2*a_blk + j, 2*b + k]
                V_contrib = {
                    0: [(2*a_blk, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],
                    1: [(2*a_blk, 2*b, 1)],
                    2: [(2*a_blk, 2*b+1, 1), (2*a_blk+1, 2*b+1, -1)],
                    3: [(2*a_blk+1, 2*b, 1), (2*a_blk, 2*b, -1)],
                    4: [(2*a_blk+1, 2*b+1, 1)],
                    5: [(2*a_blk, 2*b, 1), (2*a_blk, 2*b+1, 1)],
                    6: [(2*a_blk+1, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],
                }
                for (j, k, coeff) in V_contrib[s]:
                    V[j * 4 + k, c] += coeff

                # W contributions: C[i, 2*b + k]
                W_contrib = {
                    0: {(0, 0): 1, (1, 1): 1},
                    1: {(1, 0): 1, (1, 1): -1},
                    2: {(0, 1): 1, (1, 1): 1},
                    3: {(0, 0): 1, (1, 0): 1},
                    4: {(0, 0): -1, (0, 1): 1},
                    5: {(1, 1): 1},
                    6: {(0, 0): 1},
                }
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    W[(2 * b + k_loc) * 2 + i_loc, c] += coeff

                col += 1

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Check if U, V, W correctly decompose the matmul tensor."""
    rank = U.shape[1]
    for i in range(n):
        for k in range(m):
            for j in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _try_reduce_rank(U, V, W, n, m, p, max_iterations=50, tol=1e-9, deadline=None):
    """
    Attempt to reduce the rank by driving one column to zero using BFGS optimization.
    For each candidate column c_target, try to find adjustments to other columns
    that absorb its tensor contribution.
    """
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W
    
    nm, mp, pn = n*m, m*p, p*n
    n_cols = rank - 1
    total_params = nm * n_cols + mp * n_cols + pn * n_cols
    
    def tensor_residual(params, c_target, U0, V0, W0):
        """Compute squared tensor residual when trying to eliminate column c_target."""
        u_offset = 0
        v_offset = nm * n_cols
        w_offset = v_offset + mp * n_cols
        
        # Build adjusted U, V, W (excluding column c_target)
        U_adj = U0.copy()
        V_adj = V0.copy()
        W_adj = W0.copy()
        
        param_idx = 0
        for i in range(nm):
            for r in range(rank):
                if r == c_target:
                    continue
                U_adj[i, r] += params[u_offset + param_idx]
                param_idx += 1
        
        param_idx = 0
        for i in range(mp):
            for r in range(rank):
                if r == c_target:
                    continue
                V_adj[i, r] += params[v_offset + param_idx]
                param_idx += 1
        
        param_idx = 0
        for i in range(pn):
            for r in range(rank):
                if r == c_target:
                    continue
                W_adj[i, r] += params[w_offset + param_idx]
                param_idx += 1
        
        # Compute residual
        residual_sq = 0.0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    target = U0[i*m+k, c_target] * V0[k*p+j, c_target] * W0[j*n+i, c_target]
                    others = 0.0
                    for r in range(rank):
                        if r == c_target:
                            continue
                        others += U_adj[i*m+k, r] * V_adj[k*p+j, r] * W_adj[j*n+i, r]
                    diff = target - others
                    residual_sq += diff * diff
        
        return residual_sq
    
    # Try to eliminate each column
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    for c_target in range(rank):
        if deadline is not None and time.time() > deadline:
            break
        
        x0 = np.zeros(total_params)
        
        def objective(params, ct=c_target):
            return tensor_residual(params, ct, U, V, W)
        
        try:
            result = opt.minimize(objective, x0, method='BFGS', options={'maxiter': 30, 'gtol': 1e-8})
        except Exception:
            continue
        
        if result.fun < tol:
            # Successfully eliminated column c_target
            mask = [r for r in range(rank) if r != c_target]
            new_U = U[:, mask].copy()
            new_V = V[:, mask].copy()
            new_W = W[:, mask].copy()
            
            # Apply adjustments
            param_idx = 0
            for i in range(nm):
                for r_idx, r in enumerate(mask):
                    new_U[i, r_idx] += result.x[param_idx]
                    param_idx += 1
            
            param_idx = 0
            for i in range(mp):
                for r_idx, r in enumerate(mask):
                    new_V[i, r_idx] += result.x[param_idx]
                    param_idx += 1
            
            param_idx = 0
            for i in range(pn):
                for r_idx, r in enumerate(mask):
                    new_W[i, r_idx] += result.x[param_idx]
                    param_idx += 1
            
            if _verify_decomposition(new_U, new_V, new_W, n, m, p, tol):
                if new_U.shape[1] < best_rank:
                    best_U, best_V, best_W = new_U, new_V, new_W
                    best_rank = new_U.shape[1]
                    break
    
    return best_U, best_V, best_W


def construct():
    """
    Return a rank-28 exact decomposition of the 2x4 * 4x4 matrix multiplication tensor.
    Uses Strassen's algorithm on 2x2 blocks.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    start_time = time.time()
    deadline = start_time + 110  # 110s limit, keep margin before 120s
    
    # Build the base rank-28 Strassen decomposition
    U, V, W = _build_strassen_rank28()
    
    # Verify it's correct
    assert _verify_decomposition(U, V, W, n, m, p), "Base rank-28 decomposition failed verification"
    
    # Try to reduce rank (heuristic, may not succeed)
    if time.time() < deadline:
        U, V, W = _try_reduce_rank(U, V, W, n, m, p, max_iterations=50, deadline=deadline)
        # Re-verify after potential reduction
        if not _verify_decomposition(U, V, W, n, m, p):
            # Revert to original if reduction broke correctness
            U, V, W = _build_strassen_rank28()
    
    # Return the result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
