"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 6]
DEADLINE = 135.0  # seconds, safety margin before 150s kill

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W, "rank": rank}


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact."""
    rank = len(U[0])
    # Build numpy arrays
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The multiplication tensor: T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For each (i,j,k), check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U_np[i * m + j, c] * V_np[j * p + k, c] * W_np[k * n + i, c]
                if abs(actual - expected) > tol:
                    return False
                # Also check orthogonality: for different (i',j',k') with at least one index different,
                # the product should be 0 if it's a proper decomposition (not strictly required for correctness)
    # Also check that for non-matching indices, the sum is 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if (i, j, k) == (i2, j2, k2):
                                continue
                            actual = 0.0
                            for c in range(rank):
                                actual += U_np[i * m + j, c] * V_np[j2 * p + k2, c] * W_np[k2 * n + i2, c]
                            if abs(actual) > tol:
                                return False
    return True


def _try_rank_structured(n, m, p, target_rank, max_iters=100, tol=1e-8):
    """
    Try to find a decomposition using structured initialization based on
    the observation that (3,3,6) can be viewed as two (3,3,3) blocks.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # Initialize using the two-block structure: k=0,1,2 is block 0, k=3,4,5 is block 1
    # For each block, use a random (3,3,3)-like initialization
    U = np.zeros((nm, target_rank))
    V = np.zeros((mp, target_rank))
    W = np.zeros((pn, target_rank))
    
    # Split target_rank into two halves for the two blocks
    half_rank = target_rank // 2
    rem = target_rank % 2
    
    for block in range(2):
        start = block * half_rank + min(block, rem)
        end = start + half_rank + (1 if block < rem else 0)
        block_size = end - start
        
        if block_size <= 0:
            continue
            
        # Random initialization for this block
        U_block = np.random.randn(nm, block_size)
        V_block = np.random.randn(mp, block_size)
        W_block = np.random.randn(pn, block_size)
        
        # Only activate relevant indices
        # For block 0: k in {0,1,2}, for block 1: k in {3,4,5}
        k_indices = [3*block + b for b in range(3)]
        
        # Zero out V rows not in this block's k range
        V_block_active = V_block.copy()
        V_block_zero = np.zeros_like(V_block)
        for k in k_indices:
            for j in range(m):
                V_block_zero[j*p+k, :] = V_block[j*p+k, :]
        V_block = V_block_zero
        
        # Zero out W rows not in this block's k range
        W_block_active = W_block.copy()
        W_block_zero = np.zeros_like(W_block)
        for k in k_indices:
            for i in range(n):
                W_block_zero[k*n+i, :] = W_block[k*n+i, :]
        W_block = W_block_zero
        
        U[start:end] = U_block
        V[start:end] = V_block
        W[start:end] = W_block
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 5:
            break
        
        # Compute full loss efficiently
        loss = 0.0
        count = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                                actual = np.sum(U[i*m+j, :] * V[j2*p+k2, :] * W[k2*n+i2, :])
                                loss += (actual - expected) ** 2
                                count += 1
        loss /= count
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < tol:
            break
        
        # Gradient-free update with adaptive step size
        lr = 0.05 / (1.0 + best_loss)
        U += lr * np.random.randn(nm, target_rank) * (1.0 + best_loss)
        V += lr * np.random.randn(mp, target_rank) * (1.0 + best_loss)
        W += lr * np.random.randn(pn, target_rank) * (1.0 + best_loss)
    
    if best_loss < 1e-4:
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        if _verify(U_list, V_list, W_list, n, m, p, tol=1e-4):
            return {"U": U_list, "V": V_list, "W": W_list, "rank": target_rank}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # For (3,3,6): the tensor can be viewed as two (3,3,3) blocks (k=0,1,2 and k=3,4,5)
    # Lower bound: max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # Upper bound: schoolbook = 54, or 2*rank(3,3,3) if we use block structure
    # Known: rank(3,3,3) <= 23, so rank(3,3,6) <= 46 via block decomposition
    
    # Try ranks from the theoretical lower bound upward
    # Use structured initialization for better convergence
    lower_bound = max(n*m, m*p, p*n)
    
    # First try a few ranks near the lower bound with structured approach
    for target_rank in range(lower_bound, min(schoolbook_rank, lower_bound + 30)):
        if time.time() > DEADLINE - 10:
            break
        result = _try_rank_structured(n, m, p, target_rank, max_iters=30)
        if result is not None:
            return {"U": result["U"], "V": result["V"], "W": result["W"]}
    
    # Fall back to schoolbook
    sb = _schoolbook(n, m, p)
    return {"U": sb["U"], "V": sb["V"], "W": sb["W"]}

# EVOLVE-BLOCK-END
