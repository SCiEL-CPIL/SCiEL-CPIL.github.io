"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time
import itertools

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition_fast(U, V, W, n, m, p, rank):
    """Vectorized verification: check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    # For each (i,j,k), compute the sum over r
    # U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U_np.reshape(n, m, rank)
    V3 = V_np.reshape(m, p, rank)
    W3 = W_np.reshape(p, n, rank)
    # Product: U3[i,j,:] * V3[j,k,:] * W3[k,i,:] summed over last axis
    # Use einsum: 'ijr,jkr,ki r->ijk'
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(result, 1.0, atol=1e-9)

def _try_rank_reduction(n, m, p, max_time=100):
    """Try to find a rank-minimal decomposition below n*m*p using structured search."""
    start_time = time.time()
    base_rank = n * m * p
    deadline = start_time + max_time - 5  # safety margin
    
    # Schoolbook initialization
    U = [[0] * base_rank for _ in range(n * m)]
    V = [[0] * base_rank for _ in range(m * p)]
    W = [[0] * base_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = base_rank
    
    # Build the multiplication tensor as a flat vector for efficient operations
    # T[(i,j,k)] = 1 for all valid triples
    # We'll work with the tensor in a form where each term r contributes:
    #   term_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # and we need sum_r term_r = 1 for all (i,j,k)
    
    # Strategy: Start from schoolbook and try to merge/eliminate terms
    # For small dimensions, we can try algebraic reductions
    
    # Approach: Try to find a decomposition of rank < base_rank by
    # solving a system of polynomial equations. For rank r, we need:
    #   sum_{c=0}^{r-1} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1  for all (i,j,k)
    
    # This is a system of n*m*p equations in 3*r*(n*m + m*p + p*n) unknowns.
    # For small reductions, we can use random initialization + gradient descent.
    
    # More efficient: use the structure that for prime dimensions,
    # known bounds give specific ranks. For (3,5,7), the tensor rank is
    # at most n*m*p = 105, and potentially lower.
    
    # Try gradient-based optimization with multiple restarts
    # Key improvement: use vectorized operations and check convergence more frequently
    
    for target_rank in range(base_rank - 1, max(20, base_rank - 50), -1):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            np.random.seed(12345 + target_rank * 1000 + attempt * 7)
            
            # Initialize with small random values
            U_rand = np.random.randn(n*m, target_rank) * 0.5
            V_rand = np.random.randn(m*p, target_rank) * 0.5
            W_rand = np.random.randn(p*n, target_rank) * 0.5
            
            # Adaptive learning rate gradient descent
            lr = 0.05
            prev_loss = float('inf')
            
            for step in range(200):
                if time.time() > deadline:
                    break
                
                # Compute approximation and error
                U3 = U_rand.reshape(n, m, target_rank)
                V3 = V_rand.reshape(m, p, target_rank)
                W3 = W_rand.reshape(p, n, target_rank)
                
                approx = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                error = 1.0 - approx  # We want approx = 1
                
                loss = np.sum(error**2)
                
                # Early termination if converged
                if loss < 1e-10:
                    # Verify with exact integer check
                    U_rounded = np.round(U_rand)
                    V_rounded = np.round(V_rand)
                    W_rounded = np.round(W_rand)
                    
                    if _verify_decomposition_fast(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p, target_rank):
                        best_U = U_rounded.tolist()
                        best_V = V_rounded.tolist()
                        best_W = W_rounded.tolist()
                        best_rank = target_rank
                        return best_U, best_V, best_W, best_rank
                    break
                
                # Compute gradients
                grad_U = np.einsum('ijk,jkr,kir->ijr', error, V3, W3)
                grad_V = np.einsum('ijk,ijr,kir->jkr', error, U3, W3)
                grad_W = np.einsum('ijk,ijr,jkr->kir', error, U3, V3)
                
                # Reshape gradients back
                grad_U = grad_U.reshape(n*m, target_rank)
                grad_V = grad_V.reshape(m*p, target_rank)
                grad_W = grad_W.reshape(p*n, target_rank)
                
                # Update
                U_rand -= lr * grad_U
                V_rand -= lr * grad_V
                W_rand -= lr * grad_W
                
                # Adaptive learning rate
                if loss < prev_loss:
                    lr = min(lr * 1.1, 0.5)
                else:
                    lr = max(lr * 0.8, 0.001)
                prev_loss = loss
                
                # Check for near-convergence periodically
                if step % 20 == 0 and loss < 1e-6:
                    U_rounded = np.round(U_rand)
                    V_rounded = np.round(V_rand)
                    W_rounded = np.round(W_rand)
                    if _verify_decomposition_fast(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p, target_rank):
                        best_U = U_rounded.tolist()
                        best_V = V_rounded.tolist()
                        best_W = W_rounded.tolist()
                        best_rank = target_rank
                        return best_U, best_V, best_W, best_rank
                    break
    
    # If no improvement found, return schoolbook
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # First get schoolbook
    base_rank = n * m * p
    U = [[0] * base_rank for _ in range(n * m)]
    V = [[0] * base_rank for _ in range(m * p)]
    W = [[0] * base_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    # Try to improve with time budget
    try:
        U_best, V_best, W_best, rank_best = _try_rank_reduction(n, m, p, max_time=130)
        if rank_best < base_rank:
            return {"U": U_best, "V": V_best, "W": W_best}
    except Exception:
        pass
    
    # Fallback to schoolbook
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
