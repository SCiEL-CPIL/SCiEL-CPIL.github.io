"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify that the decomposition correctly computes matrix multiplication."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        
        # Test with random matrices
        rng = np.random.RandomState(42)
        for trial in range(10):
            A = rng.rand(n, m)
            B = rng.rand(m, p)
            C_expected = A @ B
            
            C_computed = np.zeros((n, p))
            for r in range(rank):
                u_val = np.dot(U_np[:, r], A.flatten())
                v_val = np.dot(V_np[:, r], B.flatten())
                # W index is k*n + i, so we need to map back to C[i, k]
                w_vec = W_np[:, r]
                for idx in range(p * n):
                    k = idx // n
                    i = idx % n
                    C_computed[i, k] += u_val * v_val * w_vec[idx]
            
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True
    
    def try_rank_decomposition(target_rank, max_time=30):
        """Try to find a decomposition of the given rank."""
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        # Initialize with schoolbook-like structure but allow for sharing
        # Use random initialization with some structure
        rng = np.random.RandomState(123)
        
        # Try to find a decomposition by solving the tensor equation
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For matmul, T should satisfy: for each (i,k), sum_j T[i,j,k] = 1 and T[i,j,k] is 1 if j is the middle index
        
        # Actually, the tensor for matmul has T[i,j,k] = 1 for all valid (i,j,k)
        # because each term A[i,j]*B[j,k] contributes to C[i,k]
        
        # Let's set up the constraint: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # And we want to minimize rank
        
        # Use gradient descent or a structured approach
        # For now, try a structured approach based on block decomposition
        
        if time.time() - start_time > deadline:
            return None
        
        return None  # Placeholder - we'll use a more direct approach below
    
    # Try to use known low-rank decompositions
    # For 3x3 x 3x5, we can decompose as (3x3 x 3x3) + (3x3 x 3x2)
    
    # First, let's try to find a good decomposition using the structure
    # The key is that the middle dimension is 3, which allows for sharing
    
    # Try rank 30, 35, 40, etc.
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to improve by finding linear dependencies in the schoolbook
    # or by using a more clever decomposition
    
    # For now, return schoolbook as a safe baseline
    # In future iterations, we can try to find better decompositions
    
    return best_result
# EVOLVE-BLOCK-END
