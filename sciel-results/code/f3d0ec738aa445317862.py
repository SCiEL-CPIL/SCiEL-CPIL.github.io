"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    best_U, best_V, best_W = U, V, W
    best_rank = rank_schoolbook
    
    # Attempt to find a lower rank decomposition
    # Time limit 10 seconds
    start_time = time.time()
    time_limit = 10.0
    
    # Try ranks from max(n*m, m*p, p*n) + 1 up to best_rank - 1
    # Lower bound is max(9, 15, 15) = 15.
    # We try a few random searches.
    
    # Note: Exact integer/half-integer entries are required.
    # Random search over small integers {-1, 0, 1}
    
    def verify(U_np, V_np, W_np, rank):
        # Check if T = sum U_r V_r W_r matches MatMul Tensor
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # Target T_target[a, b, c] is 1 if a=i*m+j, b=j*p+k, c=k*n+i
        # Else 0.
        
        # Efficient check: 
        # For each r, we have outer product.
        # We can sum them up.
        # T = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        # This is expensive for large rank.
        # Alternative: Check on random inputs.
        # If it works on random inputs, it's likely correct (probabilistic).
        # But we need exactness.
        # For small ranks, einsum is fast enough.
        
        T_computed = np.einsum('ar,br,cr->abc', U_np, V_np, W_np, optimize=True)
        
        # Build Target Tensor
        T_target = np.zeros((n * m, m * p, p * n), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_target[a, b, c] = 1
                    
        return np.allclose(T_computed, T_target)

    # Search
    for rank in range(16, best_rank):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random initializations
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
                
            # Random integer matrices in {-1, 0, 1}
            U_np = np.random.choice([-1, 0, 1], size=(n * m, rank))
            V_np = np.random.choice([-1, 0, 1], size=(m * p, rank))
            W_np = np.random.choice([-1, 0, 1], size=(p * n, rank))
            
            # Optimization: If we find one, we can stop.
            # But random search is unlikely to hit exact solution.
            # We can try to solve for W given U, V?
            # T_target = U V^T W^T? No.
            # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r].
            # For fixed a, b, this is a linear system for W[:, r]?
            # No, W is the variable.
            # Let M_ab = sum_r U[a, r] V[b, r]. This is an nm x mp matrix of inner products.
            # Then T[a, b, :] = sum_r U[a, r] V[b, r] W[:, r]? No.
            # T[a, b, c] = sum_r (U[a, r] V[b, r]) W[c, r].
            # This is not a simple matrix multiplication.
            
            # We'll just stick to the verification.
            if verify(U_np, V_np, W_np, rank):
                best_U = U_np.tolist()
                best_V = V_np.tolist()
                best_W = W_np.tolist()
                best_rank = rank
                break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
