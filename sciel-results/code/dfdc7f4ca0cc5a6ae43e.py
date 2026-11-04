"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W form a valid decomposition of the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a] * V[b] * W[c])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def check_and_return(U, V, W, tol=1e-8):
        """Check decomposition validity and return dict if valid."""
        if check_decomposition(U, V, W, tol):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None
    
    # Strategy 1: Try to construct a low-rank decomposition using structured approaches
    # For 2x3 x 3x4 matrix multiplication, we can try ranks from 19 down to lower bounds
    
    # Theoretical lower bounds:
    # - Rank >= max(n*p, n*m, m*p) is NOT a valid lower bound for bilinear rank
    # - But we know rank(n*m*p) = 24 for naive
    # - Strassen gives 7 for 2x2, so we can potentially do better than naive
    
    # Try structured decompositions at various ranks
    best_result = None
    best_rank = nm * mp * pn  # Start with naive rank
    
    # Naive decomposition as fallback
    rank_naive = n * m * p
    U_naive = np.zeros((nm, rank_naive))
    V_naive = np.zeros((mp, rank_naive))
    W_naive = np.zeros((pn, rank_naive))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U_naive[i * m + j, r] = 1
                V_naive[j * p + k, r] = 1
                W_naive[k * n + i, r] = 1
    
    best_result = check_and_return(U_naive, V_naive, W_naive)
    best_rank = rank_naive
    
    # Try to find lower rank decompositions
    # Use a combination of structured initialization and optimization
    
    # Lower bound estimate: for 2x3x3, rank is known to be 13 (not 18 naive)
    # For 2x3x4, we expect rank < 24
    # Try ranks from 19 down to 15
    
    for target_rank in range(19, 14, -1):
        if time.time() > deadline - 25:
            break
        
        found = False
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
            
            U_try = np.zeros((nm, target_rank))
            V_try = np.zeros((mp, target_rank))
            W_try = np.zeros((pn, target_rank))
            
            if attempt == 0:
                # Random initialization with moderate scale
                np.random.seed(1000 + attempt)
                U_try = np.random.randn(nm, target_rank) * 0.5
                V_try = np.random.randn(mp, target_rank) * 0.5
                W_try = np.random.randn(pn, target_rank) * 0.5
            elif attempt == 1:
                # Sparse random initialization
                np.random.seed(2000 + attempt)
                U_try = np.random.randn(nm, target_rank)
                V_try = np.random.randn(mp, target_rank)
                W_try = np.random.randn(pn, target_rank)
                # Sparsify with different densities
                U_try *= (np.random.rand(nm, target_rank) > 0.6)
                V_try *= (np.random.rand(mp, target_rank) > 0.6)
                W_try *= (np.random.rand(pn, target_rank) > 0.6)
            elif attempt == 2:
                # Use patterned initialization based on Strassen-like structure
                # For 2x2x2, Strassen uses 7 products
                # We'll create a block structure
                np.random.seed(3000 + attempt)
                for r in range(target_rank):
                    # Assign random but structured entries
                    U_try[:, r] = np.random.randn(nm) * 0.3
                    V_try[:, r] = np.random.randn(mp) * 0.3
                    W_try[:, r] = np.random.randn(pn) * 0.3
            elif attempt == 3:
                # Initialize from naive and randomly project
                np.random.seed(4000 + attempt)
                if target_rank <= rank_naive:
                    U_try = U_naive[:, :target_rank].copy()
                    V_try = V_naive[:, :target_rank].copy()
                    W_try = W_naive[:, :target_rank].copy()
                    # Add small random perturbation
                    U_try += np.random.randn(nm, target_rank) * 0.01
                    V_try += np.random.randn(mp, target_rank) * 0.01
                    W_try += np.random.randn(pn, target_rank) * 0.01
            else:
                # Hybrid: start sparse, then densify
                np.random.seed(5000 + attempt)
                U_try = np.random.randn(nm, target_rank) * 0.2
                V_try = np.random.randn(mp, target_rank) * 0.2
                W_try = np.random.randn(pn, target_rank) * 0.2
                # Make some entries larger
                for r in range(0, target_rank, 3):
                    U_try[:, r] *= 2.0
                    V_try[:, r] *= 2.0
                    W_try[:, r] *= 2.0
            
            # Optimize using gradient descent with adaptive learning rate
            lr = 0.02
            best_err = np.inf
            stuck_count = 0
            
            for step in range(300):
                if time.time() > deadline - 10:
                    break
                
                # Compute gradients
                grad_U = np.zeros_like(U_try)
                grad_V = np.zeros_like(V_try)
                grad_W = np.zeros_like(W_try)
                total_err = 0.0
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            val = np.sum(U_try[a] * V_try[b] * W_try[c])
                            err = val - 1.0
                            total_err += err * err
                            grad_U[a] += err * (V_try[b] * W_try[c])
                            grad_V[b] += err * (U_try[a] * W_try[c])
                            grad_W[c] += err * (U_try[a] * V_try[b])
                
                # Adaptive learning rate
                grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
                if grad_norm > 1e-10:
                    effective_lr = lr * min(1.0, 0.1 / grad_norm)
                else:
                    effective_lr = lr
                
                U_try -= effective_lr * grad_U
                V_try -= effective_lr * grad_V
                W_try -= effective_lr * grad_W
                
                if total_err < best_err:
                    best_err = total_err
                    stuck_count = 0
                else:
                    stuck_count += 1
                
                if total_err < 1e-12:
                    break
                if stuck_count > 50:
                    break
            
            # Check if valid
            result = check_and_return(U_try, V_try, W_try)
            if result is not None:
                best_rank = target_rank
                best_result = result
                found = True
                break
        
        if found:
            continue
    
    # If no lower rank found, return the naive decomposition
    if best_result is None:
        best_result = check_and_return(U_naive, V_naive, W_naive)
    
    return best_result
# EVOLVE-BLOCK-END
