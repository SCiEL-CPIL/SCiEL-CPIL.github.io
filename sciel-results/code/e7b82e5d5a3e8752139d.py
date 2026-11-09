"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    # Check tensor equality: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal delta(i,j,k) for standard MM
    # Actually, for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,k,j] = delta for the multiplication operation
    # In the trace convention W[k*n+i, r], we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this corresponds to a valid (i,j,k) term
    # Actually the standard form is: C[i,k] = sum_j sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] * A[i,j] * B[j,k]
    # For exact decomposition of the multiplication tensor, we need the bilinear form to reproduce the product.
    
    # Let's verify by constructing the full multiplication tensor and comparing
    # The multiplication tensor T has T[i,k,j] = 1 for each (i,j,k) triple
    # In the decomposition: T[i,k,j] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if val != 1:
                    return False
    return True

def _try_rank_reduction(n, m, p, max_time=100):
    """Try to find a rank-minimal decomposition below n*m*p."""
    start_time = time.time()
    base_rank = n * m * p
    
    # Start with schoolbook
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
    
    # Convert to numpy for linear algebra operations
    def to_np(U_list, V_list, W_list, rank):
        U_np = np.array(U_list, dtype=float)
        V_np = np.array(V_list, dtype=float)
        W_np = np.array(W_list, dtype=float)
        return U_np, V_np, W_np
    
    def from_np(U_np, V_np, W_np, rank):
        U_list = [[int(round(U_np[i][r])) for r in range(rank)] for i in range(n*m)]
        V_list = [[int(round(V_np[i][r])) for r in range(rank)] for i in range(m*p)]
        W_list = [[int(round(W_np[i][r])) for r in range(rank)] for i in range(p*n)]
        return U_list, V_list, W_list
    
    # Strategy: try to eliminate terms by finding linear dependencies
    # For each term, check if it can be expressed as a combination of others
    # This is a hard problem, so we use heuristic approaches
    
    # Approach 1: Random perturbation and SVD-based compression
    # The multiplication tensor is T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k)
    # We can try to find low-rank approximations
    
    # Build the 3D tensor
    tensor = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i*m+j, j*p+k, k*n+i] = 1
    
    # Try different rank values
    for target_rank in range(base_rank - 1, max(10, base_rank - 30), -1):
        if time.time() - start_time > max_time - 5:
            break
        
        # Try random initialization with target rank
        for attempt in range(5):
            if time.time() - start_time > max_time - 5:
                break
            
            # Random initialization
            np.random.seed(42 + target_rank * 100 + attempt)
            U_rand = np.random.randn(n*m, target_rank)
            V_rand = np.random.randn(m*p, target_rank)
            W_rand = np.random.randn(p*n, target_rank)
            
            # Iterative optimization: minimize ||tensor - sum_r U_r outer V_r outer W_r||
            # Use gradient descent on the non-convex problem
            lr = 0.01
            for step in range(50):
                if time.time() - start_time > max_time - 5:
                    break
                
                # Compute current approximation
                approx = np.zeros((n*m, m*p, p*n))
                for r in range(target_rank):
                    approx += np.einsum('i,j,k->ijk', U_rand[:,r], V_rand[:,r], W_rand[:,r])
                
                # Compute gradient
                error = tensor - approx
                grad_U = np.zeros_like(U_rand)
                grad_V = np.zeros_like(V_rand)
                grad_W = np.zeros_like(W_rand)
                
                for r in range(target_rank):
                    # Gradient wrt U[:,r]
                    grad_U[:,r] = np.einsum('ijk,jk->i', error, 
                                            np.einsum('j,k->jk', V_rand[:,r], W_rand[:,r]))
                    # Gradient wrt V[:,r]
                    grad_V[:,r] = np.einsum('ijk,ik->j', error, 
                                            np.einsum('i,k->ik', U_rand[:,r], W_rand[:,r]))
                    # Gradient wrt W[:,r]
                    grad_W[:,r] = np.einsum('ijk,ij->k', error, 
                                            np.einsum('i,j->ij', U_rand[:,r], V_rand[:,r]))
                
                # Update
                U_rand -= lr * grad_U
                V_rand -= lr * grad_V
                W_rand -= lr * grad_W
                
                # Check convergence
                if step % 10 == 0:
                    err_norm = np.linalg.norm(error)
                    if err_norm < 1e-6:
                        # Check if we can round to integers
                        U_rounded = np.round(U_rand)
                        V_rounded = np.round(V_rand)
                        W_rounded = np.round(W_rand)
                        
                        U_list, V_list, W_list = from_np(U_rounded, V_rounded, W_rounded, target_rank)
                        if _verify_decomposition(U_list, V_list, W_list, n, m, p, target_rank):
                            best_U, best_V, best_W = U_list, V_list, W_list
                            best_rank = target_rank
                            return best_U, best_V, best_W, best_rank
                        break
    
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
    
    # Try to improve
    try:
        U_best, V_best, W_best, rank_best = _try_rank_reduction(n, m, p, max_time=110)
        if rank_best < base_rank:
            return {"U": U_best, "V": V_best, "W": W_best}
    except Exception:
        pass
    
    # Fallback to schoolbook
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
