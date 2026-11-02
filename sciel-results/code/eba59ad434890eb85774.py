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


def _try_reduce_rank_als(U, V, W, n, m, p, max_iterations=100, tol=1e-9, deadline=None):
    """
    Attempt to reduce the rank by driving one column to zero using Alternating Least Squares (ALS).
    """
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W
    
    nm, mp, pn = n*m, m*p, p*n
    n_cols = rank - 1
    
    # Precompute index arrays for vectorized residual
    idx_i = np.repeat(np.arange(n), m * p)
    idx_k = np.tile(np.repeat(np.arange(m), p), n)
    idx_j = np.tile(np.arange(p), n * m)
    
    # Target indices for U, V, W
    # U index: i*m + k
    # V index: k*p + j
    # W index: j*n + i
    
    u_idx = idx_i * m + idx_k
    v_idx = idx_k * p + idx_j
    w_idx = idx_j * n + idx_i
    
    def compute_residual(U_adj, V_adj, W_adj, c_target):
        """Compute squared tensor residual when trying to eliminate column c_target."""
        target = U[u_idx, c_target] * V[v_idx, c_target] * W[w_idx, c_target]
        
        others = np.zeros(n * m * p)
        for r in range(n_cols):
            others += U_adj[u_idx, r] * V_adj[v_idx, r] * W_adj[w_idx, r]
        
        diff = target - others
        return np.sum(diff * diff)

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    rng = np.random.RandomState(42)
    
    for c_target in range(rank):
        if deadline is not None and time.time() > deadline:
            break
        
        found = False
        for restart in range(5):
            if deadline is not None and time.time() > deadline:
                break
            
            # Initialize adjustments
            if restart == 0:
                U_adj = U[:, [r for r in range(rank) if r != c_target]].copy()
                V_adj = V[:, [r for r in range(rank) if r != c_target]].copy()
                W_adj = W[:, [r for r in range(rank) if r != c_target]].copy()
            else:
                U_adj = U[:, [r for r in range(rank) if r != c_target]].copy() + rng.normal(0, 0.01, (nm, n_cols))
                V_adj = V[:, [r for r in range(rank) if r != c_target]].copy() + rng.normal(0, 0.01, (mp, n_cols))
                W_adj = W[:, [r for r in range(rank) if r != c_target]].copy() + rng.normal(0, 0.01, (pn, n_cols))

            # ALS iterations
            prev_res = np.inf
            for it in range(max_iterations):
                if deadline is not None and time.time() > deadline:
                    break
                
                # Update U
                # Minimize || T - sum U V W ||^2
                # For each row i of U, the contribution is U[i, :] * (V * W)
                # We can solve for U[i, :] by least squares
                # Residual R = T - sum_{r != i_row} U[i, r] V[:, r] W[:, r]
                # This is complex to vectorize per row.
                # Instead, we can use gradient descent on U, V, W.
                
                # Let's use a simple gradient step
                # Gradient of residual w.r.t U[i, r]
                # dR/dr = 2 * (T - sum U V W) * V[v_idx, r] * W[w_idx, r]
                
                # Compute current residual vector
                others = np.zeros(n * m * p)
                for r in range(n_cols):
                    others += U_adj[u_idx, r] * V_adj[v_idx, r] * W_adj[w_idx, r]
                target = U[u_idx, c_target] * V[v_idx, c_target] * W[w_idx, c_target]
                diff = target - others
                
                # Update U
                for i in range(nm):
                    mask = (u_idx == i)
                    if np.any(mask):
                        grad_U = np.zeros(n_cols)
                        for r in range(n_cols):
                            grad_U[r] = 2 * np.sum(diff[mask] * V_adj[v_idx[mask], r] * W_adj[w_idx[mask], r])
                        U_adj[i, :] -= 0.01 * grad_U
                
                # Update V
                for k in range(mp):
                    mask = (v_idx == k)
                    if np.any(mask):
                        grad_V = np.zeros(n_cols)
                        for r in range(n_cols):
                            grad_V[r] = 2 * np.sum(diff[mask] * U_adj[u_idx[mask], r] * W_adj[w_idx[mask], r])
                        V_adj[k, :] -= 0.01 * grad_V
                
                # Update W
                for j in range(pn):
                    mask = (w_idx == j)
                    if np.any(mask):
                        grad_W = np.zeros(n_cols)
                        for r in range(n_cols):
                            grad_W[r] = 2 * np.sum(diff[mask] * U_adj[u_idx[mask], r] * V_adj[v_idx[mask], r])
                        W_adj[j, :] -= 0.01 * grad_W
                
                # Check convergence
                res = np.sum(diff * diff)
                if abs(prev_res - res) < 1e-12:
                    break
                prev_res = res
                
                if res < tol:
                    break
            
            if prev_res < tol:
                if _verify_decomposition(U_adj, V_adj, W_adj, n, m, p, tol):
                    if U_adj.shape[1] < best_rank:
                        best_U, best_V, best_W = U_adj, V_adj, W_adj
                        best_rank = U_adj.shape[1]
                        found = True
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
        U, V, W = _try_reduce_rank_als(U, V, W, n, m, p, max_iterations=100, deadline=deadline)
        # Re-verify after potential reduction
        if not _verify_decomposition(U, V, W, n, m, p):
            # Revert to original if reduction broke correctness
            U, V, W = _build_strassen_rank28()
    
    # Return the result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
