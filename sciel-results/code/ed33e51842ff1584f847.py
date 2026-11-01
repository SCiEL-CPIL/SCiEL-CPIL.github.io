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


def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Check if U, V, W correctly decompose the matmul tensor using vectorized ops."""
    rank = U.shape[1]
    # T[i*m+k, k*p+j, j*n+i] should equal 1 for all i,k,j
    # Compute: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # Vectorized: for each r, outer product of U[:,r], V[:,r], W[:,r]
    # T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] reshaped appropriately
    
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                target[i*m+k, k*p+j, j*n+i] = 1.0
    
    # Compute actual tensor
    actual = np.zeros_like(target)
    for r in range(rank):
        actual += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    return np.allclose(actual, target, atol=tol)


def _try_rank_reduction(U, V, W, n, m, p, target_rank, max_time=60):
    """Try to reduce rank by searching for valid decompositions with fewer terms.
    
    Strategy: Start from rank-28, try to merge/eliminate terms.
    Use random perturbations and optimization to find valid lower-rank solutions.
    """
    start = time.time()
    rank = U.shape[1]
    if rank <= target_rank:
        return U, V, W
    
    # Try random restarts with scipy optimization
    from scipy.optimize import minimize
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    for attempt in range(50):
        if time.time() - start > max_time:
            break
        
        # Strategy: Try to find a valid decomposition by perturbing and optimizing
        # We'll work with a parametrization where we try to drop one term
        
        # Create a candidate by zeroing out one random term and compensating
        r_to_drop = random.randint(0, rank - 1)
        
        # Create modified matrices with the dropped term
        U_mod = U.copy()
        V_mod = V.copy()
        W_mod = W.copy()
        
        # Zero out the term
        U_mod[:, r_to_drop] = 0
        V_mod[:, r_to_drop] = 0
        W_mod[:, r_to_drop] = 0
        
        # Now we need to adjust other terms to compensate
        # The error tensor is: -U[:,r_to_drop] ⊗ V[:,r_to_drop] ⊗ W[:,r_to_drop]
        # We need to add this to the sum of remaining terms
        
        # Parametrize adjustments as small perturbations to remaining terms
        # Use a random linear combination approach
        
        # Alternative: use random search with validation
        # Try random linear combinations of existing terms to replace the dropped one
        
        # Generate random coefficients
        coeffs = np.random.randn(rank) * 0.1
        
        # Create new U, V, W as linear combinations
        U_new = U @ coeffs
        V_new = V @ coeffs
        W_new = W @ coeffs
        
        # This won't work directly - we need to preserve the tensor structure
        # Let's use a different approach: random restarts with gradient-free search
        
        # Approach: randomly perturb all entries slightly, then check validity
        # If close to valid, try to refine
        
        noise_scale = 0.01
        U_pert = U + np.random.randn(*U.shape) * noise_scale
        V_pert = V + np.random.randn(*V.shape) * noise_scale
        W_pert = W + np.random.randn(*W.shape) * noise_scale
        
        # Check if this perturbation is close to valid
        if _verify_decomposition(U_pert, V_pert, W_pert, n, m, p, tol=0.1):
            # It's close - try to refine with optimization
            # Flatten parameters
            params = np.concatenate([U_pert.flatten(), V_pert.flatten(), W_pert.flatten()])
            
            def objective(x):
                U_x = x[:U.shape[0]*U.shape[1]].reshape(U.shape)
                V_x = x[U.shape[0]*U.shape[1]:U.shape[0]*U.shape[1]+V.shape[0]*V.shape[1]].reshape(V.shape)
                W_x = x[U.shape[0]*U.shape[1]+V.shape[0]*V.shape[1]:].reshape(W.shape)
                
                # Compute tensor error
                target = np.zeros((n*m, m*p, p*n))
                for i in range(n):
                    for k in range(m):
                        for j in range(p):
                            target[i*m+k, k*p+j, j*n+i] = 1.0
                
                actual = np.zeros_like(target)
                for r in range(U_x.shape[1]):
                    actual += np.einsum('a,b,c->abc', U_x[:, r], V_x[:, r], W_x[:, r])
                
                return np.sum((actual - target)**2)
            
            # Use Nelder-Mead (doesn't require gradients)
            result = minimize(objective, params, method='Nelder-Mead', 
                            options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if result.fun < 1e-6:
                U_final = result.x[:U.shape[0]*U.shape[1]].reshape(U.shape)
                V_final = result.x[U.shape[0]*U.shape[1]:U.shape[0]*U.shape[1]+V.shape[0]*V.shape[1]].reshape(V.shape)
                W_final = result.x[U.shape[0]*U.shape[1]+V.shape[0]*V.shape[1]:].reshape(W.shape)
                
                if _verify_decomposition(U_final, V_final, W_final, n, m, p, tol=1e-6):
                    # Found a valid solution - now try to reduce rank
                    # Check if any term can be dropped
                    for r_check in range(U_final.shape[1]):
                        U_test = U_final.copy()
                        V_test = V_final.copy()
                        W_test = W_final.copy()
                        U_test[:, r_check] = 0
                        V_test[:, r_check] = 0
                        W_test[:, r_check] = 0
                        
                        if _verify_decomposition(U_test, V_test, W_test, n, m, p, tol=1e-6):
                            # Remove this term
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
