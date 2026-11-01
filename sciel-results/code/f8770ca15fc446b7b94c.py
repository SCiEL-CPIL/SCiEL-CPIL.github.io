import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify that U, V, W form a valid tensor decomposition."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # The tensor for matrix multiplication is T[i,j,k,l] = delta_{i,l} * delta_{j,k}
    # In the factorized form: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We check that this equals 1 if the indices are consistent (which they always are in this form)
    # Actually, the multiplication tensor is: T[i,j,k] = 1 for all (i,j,k) where we're computing
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = delta_{i,i} * delta_{j,j} * delta_{k,k} = 1 for all valid indices
    # Wait, that's not right. Let me think again.
    
    # The matrix multiplication tensor T is defined by:
    # T[i, j, k] = 1 if and only if we need A[i,j] * B[j,k] to contribute to C[i,k]
    # So T[i, j, k] = 1 for all i, j, k (since every A[i,j]*B[j,k] contributes to C[i,k])
    # No wait, that's the structure tensor. Let me re-read the task.
    
    # "exactly decompose the matrix multiplication tensor"
    # The multiplication tensor T has entries T[i,j,k] where:
    # T[i,j,k] = 1 if A[i,j] and B[j,k] multiply to contribute to C[i,k]
    # So T[i,j,k] = 1 for all valid i,j,k
    
    # In the factorized form:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For this to equal 1 for all (i,j,k), we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Let's verify this
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline with margin
    
    # Start with the trivial decomposition
    rank = n * m * p
    U0 = np.zeros((n * m, rank))
    V0 = np.zeros((m * p, rank))
    W0 = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U0.tolist(), V0.tolist(), W0.tolist()
    
    # Try to find a lower rank decomposition using random linear combinations
    # Strategy: try ranks from 24 down to some minimum, and see if we can find a valid decomposition
    
    # For matrix multiplication, the known lower bounds suggest rank >= 23 for 2x3x4
    # Let's try ranks 24, 25, 26 and see if we can find valid decompositions
    
    for target_rank in range(24, rank):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random attempts for this rank
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Generate random U, V, W with the target rank
            # We'll use a heuristic: start from the trivial decomposition and try to merge factors
            
            # Approach: randomly select subsets of the 24 trivial factors and try to combine them
            # This is a heuristic approach - try to find linear dependencies
            
            # Alternative: use a random projection approach
            # Generate random U of shape (n*m, target_rank)
            # Then solve for V and W given U
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This is a bilinear equation in V and W given U
            
            # Simpler heuristic: try random integer/half-integer entries
            U_rand = np.zeros((n * m, target_rank))
            V_rand = np.zeros((m * p, target_rank))
            W_rand = np.zeros((p * n, target_rank))
            
            # Fill with random values from {-1, -0.5, 0, 0.5, 1}
            vals = [-1, -0.5, 0, 0.5, 1]
            for r in range(target_rank):
                for idx in range(n * m):
                    U_rand[idx, r] = vals[np.random.randint(0, len(vals))]
                for idx in range(m * p):
                    V_rand[idx, r] = vals[np.random.randint(0, len(vals))]
                for idx in range(p * n):
                    W_rand[idx, r] = vals[np.random.randint(0, len(vals))]
            
            if _verify(U_rand, V_rand, W_rand):
                best_rank = target_rank
                best_U, best_V, best_W = U_rand.tolist(), V_rand.tolist(), W_rand.tolist()
                break
        
        # If we found a valid decomposition at this rank, we can stop (we're going from low to high)
        if best_rank <= target_rank:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
