"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication: C = A @ B where A is n x m, B is m x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor for matrix multiplication has T[i,j,k] = 1 if we consider the standard basis
    # Actually, the tensor T is defined such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal delta_{i,j,k} in some sense
    
    # Let's think about this more carefully. The matrix multiplication tensor T_{n,m,p} is a 3-tensor
    # where T[i,j,k] = 1 if the output element (i,k) depends on input elements (i,j) and (j,k).
    # So T[i,j,k] = 1 for all valid i,j,k (it's the all-ones tensor in the appropriate space).
    # Wait, no. The tensor is defined by:
    # <T, a ⊗ b ⊗ c> = sum_{i,j,k} T[i,j,k] a[i,j] b[j,k] c[i,k]
    # For matrix multiplication, we want this to equal sum_{i,k} (sum_j a[i,j] b[j,k]) c[i,k]
    # So T[i,j,k] = 1 for all i, j, k.
    
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k
    
    # Check for all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Try to find a lower rank decomposition using randomized algorithms
    # This is a hard problem, so we'll try a few heuristic approaches
    
    # Approach: Try to find a decomposition with rank < 45
    # We can use the fact that for small matrices, there might be known optimal decompositions
    
    # For 3x3x5, the schoolbook rank is 45. Let's try to find something better.
    # One approach is to use bilinear algorithms or known results from tensor decomposition literature.
    
    # For now, let's try a simple randomized search with verification
    # This is unlikely to find a better solution quickly, but let's try
    
    if time.time() < deadline - 10:
        # Try a few random attempts to see if we can find a lower rank
        # This is a very naive approach and unlikely to work, but it's a start
        np.random.seed(42)
        
        for target_rank in range(44, 30, -1):  # Try ranks from 44 down to 31
            if time.time() > deadline - 5:
                break
            
            # Try a few random initializations
            for attempt in range(10):
                if time.time() > deadline - 5:
                    break
                
                # Random initialization
                U_rand = np.random.randn(n * m, target_rank) * 0.1
                V_rand = np.random.randn(m * p, target_rank) * 0.1
                W_rand = np.random.randn(p * n, target_rank) * 0.1
                
                # Convert to lists for verification
                U_list = U_rand.tolist()
                V_list = V_rand.tolist()
                W_list = W_rand.tolist()
                
                if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    break
    
    return best
# EVOLVE-BLOCK-END
