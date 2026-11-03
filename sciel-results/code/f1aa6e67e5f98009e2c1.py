import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using NumPy broadcasting."""
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    rank = U.shape[1]
    
    U_tensor = U.reshape(n, m, rank)
    V_tensor = V.reshape(m, p, rank)
    W_tensor = W.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_tensor, V_tensor, W_tensor)
    
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _generate_random_sparse(n, m, p, rank, density=0.1, rng=None):
    if rng is None:
        rng = np.random
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Vectorized random generation
    for c in range(rank):
        U_mask = rng.random(n * m) < density
        U[U_mask, c] = 1.0
        V_mask = rng.random(m * p) < density
        V[V_mask, c] = 1.0
        W_mask = rng.random(p * n) < density
        W[W_mask, c] = 1.0
        
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_school

    # Try to find a lower rank decomposition
    # We try ranks slightly below schoolbook.
    # Given the time constraint, we limit the number of attempts.
    # For 4x5x5, rank 100 is the schoolbook.
    # It is unlikely to find a rank < 100 quickly with random search.
    # However, we will attempt a few quick checks for rank 99, 98, etc.
    
    # We use a fixed seed for reproducibility in case of rare hits, but random seed is better for exploration.
    # Since the problem is likely to remain at 100, we prioritize speed.
    
    # Try rank 99 with a few random attempts
    for target_rank in [99, 98, 97]:
        if time.time() > deadline - 20:
            break
            
        # Number of attempts based on remaining time
        attempts = 50
        
        for _ in range(attempts):
            if time.time() > deadline - 10:
                break
                
            U_try, V_try, W_try = _generate_random_sparse(n, m, p, target_rank, density=0.2)
            
            if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = target_rank
                break # Found a valid one for this rank, try lower? No, just return best.
                # Actually, if we find rank 99, we should check 98? 
                # But it's highly unlikely. Let's just return the first improvement.
        
        if best_rank < rank_school:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
