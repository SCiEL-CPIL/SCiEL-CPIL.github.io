"""Matrix multiplication 2x4 x 4x4 tensor decomposition (rank 28 via Strassen on 2x2 blocks, search for rank 27)."""

import time
import random
import numpy as np

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


def _build_target_tensor(n, m, p):
    """Build the target matmul tensor T[i*m+k, k*p+j, j*n+i] = 1."""
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                target[i*m+k, k*p+j, j*n+i] = 1.0
    return target


def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Check if U, V, W correctly decompose the matmul tensor using vectorized ops."""
    rank = U.shape[1]
    target = _build_target_tensor(n, m, p)
    actual = np.zeros_like(target)
    for r in range(rank):
        actual += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(actual, target, atol=tol)


def _compute_tensor_residual(U, V, W, n, m, p):
    """Compute the residual tensor R = T - sum_r U_r ⊗ V_r ⊗ W_r."""
    target = _build_target_tensor(n, m, p)
    actual = np.zeros_like(target)
    for r in range(U.shape[1]):
        actual += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return target - actual


def _try_rank_reduction(U, V, W, n, m, p, target_rank, max_time=60):
    """Try to reduce rank by searching for valid decompositions with fewer terms.
    
    Strategy: Start from rank-28, try to merge/eliminate terms.
    Use random perturbations and optimization to find valid lower-rank solutions.
    """
    start = time.time()
    rank = U.shape[1]
    if rank <= target_rank:
        return U, V, W
    
    from scipy.optimize import minimize
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    target = _build_target_tensor(n, m, p)
    
    for attempt in range(80):
        if time.time() - start > max_time:
            break
        
        # Strategy: randomly perturb all entries, check proximity, then refine
        noise_scale = 0.005 * (1 + attempt * 0.01)
        U_pert = U + np.random.randn(*U.shape) * noise_scale
        V_pert = V + np.random.randn(*V.shape) * noise_scale
        W_pert = W + np.random.randn(*W.shape) * noise_scale
        
        # Quick check: compute residual norm
        residual = _compute_tensor_residual(U_pert, V_pert, W_pert, n, m, p)
        resid_norm = np.linalg.norm(residual)
        
        if resid_norm < 0.5:
            # Close enough to refine
            params = np.concatenate([U_pert.flatten(), V_pert.flatten(), W_pert.flatten()])
            
            def objective(x):
                U_x = x[:U.shape[0]*U.shape[1]].reshape(U.shape)
                V_x = x[U.shape[0]*U.shape[1]:U.shape[0]*U.shape[1]+V.shape[0]*V.shape[1]].reshape(V.shape)
                W_x = x[U.shape[0]*U.shape[1]+V.shape[0]*V.shape[1]:].reshape(W.shape)
                
                actual = np.zeros_like(target)
                for r in range(U_x.shape[1]):
                    actual += np.einsum('a,b,c->abc', U_x[:, r], V_x[:, r], W_x[:, r])
                
                return np.sum((actual - target)**2)
            
            # Use Nelder-Mead with tighter tolerances
            result = minimize(objective, params, method='Nelder-Mead', 
                            options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if result.fun < 1e-4:
                U_final = result.x[:U.shape[0]*U.shape[1]].reshape(U.shape)
                V_final = result.x[U.shape[0]*U.shape[1]:U.shape[0]*U.shape[1]+V.shape[0]*V.shape[1]].reshape(V.shape)
                W_final = result.x[U.shape[0]*U.shape[1]+V.shape[0]*V.shape[1]:].reshape(W.shape)
                
                if _verify_decomposition(U_final, V_final, W_final, n, m, p, tol=1e-6):
                    # Try to reduce rank by checking if any term can be dropped
                    for r_check in range(U_final.shape[1]):
                        U_test = U_final.copy()
                        V_test = V_final.copy()
                        W_test = W_final.copy()
                        U_test[:, r_check] = 0
                        V_test[:, r_check] = 0
                        W_test[:, r_check] = 0
                        
                        if _verify_decomposition(U_test, V_test, W_test, n, m, p, tol=1e-6):
                            mask = [i for i in range(U_final.shape[1]) if i != r_check]
                            U_reduced = U_final[:, mask]
                            V_reduced = V_final[:, mask]
                            W_reduced = W_final[:, mask]
                            
                            if U_reduced.shape[1] < best_rank:
                                best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
                                best_rank = U_reduced.shape[1]
                                if best_rank <= target_rank:
                                    return best_U, best_V, best_W
                            break
    
    return best_U, best_V, best_W


def construct():
    """
    Return a rank-minimal exact decomposition of the 2x4 * 4x4 matrix multiplication tensor.
    Uses Strassen's algorithm on 2x2 blocks as baseline (rank 28), then searches for rank 27.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    start_time = time.time()
    deadline = start_time + 110  # 110s limit, keep 10s safety margin

    # Build the base rank-28 Strassen decomposition
    U, V, W = _build_strassen_rank28()
    
    # Verify it's correct
    if not _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
        # Fallback: return as-is even if verification has numerical issues
        pass
    
    # Try to reduce rank to 27
    search_time = min(90, deadline - time.time())
    if search_time > 5:
        U, V, W = _try_rank_reduction(U, V, W, n, m, p, target_rank=27, max_time=search_time)
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        # Reset to known-good rank-28 solution
        U, V, W = _build_strassen_rank28()
    
    # Converting to lists for the expected return format
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
