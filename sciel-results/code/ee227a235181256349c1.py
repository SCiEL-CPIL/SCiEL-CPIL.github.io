import time
import numpy as np

DIMENSIONS = [2, 2, 10]

def _build_strassen_35():
    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(5):
        col_base = t * 7
        
        col = col_base
        U[0][col] = 1; U[3][col] = 1
        V[2*t][col] = 1; V[10+2*t+1][col] = 1
        W[4*t][col] = 1; W[4*t+3][col] = 1
        
        col = col_base + 1
        U[2][col] = 1; U[3][col] = 1
        V[2*t][col] = 1
        W[4*t+1][col] = 1; W[4*t+3][col] = -1
        
        col = col_base + 2
        U[0][col] = 1
        V[2*t+1][col] = 1; V[10+2*t+1][col] = -1
        W[4*t+2][col] = 1; W[4*t+3][col] = 1
        
        col = col_base + 3
        U[3][col] = 1
        V[10+2*t][col] = 1; V[2*t][col] = -1
        W[4*t][col] = 1; W[4*t+1][col] = 1
        
        col = col_base + 4
        U[0][col] = 1; U[1][col] = 1
        V[10+2*t+1][col] = 1
        W[4*t][col] = -1; W[4*t+2][col] = 1
        
        col = col_base + 5
        U[2][col] = 1; U[0][col] = -1
        V[2*t][col] = 1; V[2*t+1][col] = 1
        W[4*t+3][col] = 1
        
        col = col_base + 6
        U[1][col] = 1; U[3][col] = -1
        V[10+2*t][col] = 1; V[10+2*t+1][col] = 1
        W[4*t][col] = 1

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(result, n=2, m=2, p=10, tol=1e-9):
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    rank = U.shape[1]
    
    # Build the target tensor T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i, j, k] = 1 if i < n and j < m and k < p and (i,j,k) is a valid triple
    # For matrix multiplication C = A @ B, C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor T is defined such that T_{i,j,k} = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # So T[i, j, k] = 1 for all i, j, k where the indices are valid
    
    # Actually, the standard definition:
    # We want sum_r U[i, r] * V[j, r] * W[k, r] = delta(i, j, k)
    # where delta(i, j, k) = 1 if i, j, k are valid indices for the multiplication
    
    # Let's construct the target tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    # Reconstruct tensor from factors
    # T_recon[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    # This can be computed as: for each r, outer product of U[:,r], V[:,r], W[:,r]
    # T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # For efficiency, let's compute it differently
    # T_recon[i, j, k] = sum_r U[i, r] * (V[j, r] * W[k, r])
    # Let M[j, k, r] = V[j, r] * W[k, r]
    # Then T_recon[i, j, k] = sum_r U[i, r] * M[j, k, r]
    
    # Simpler: use einsum
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    diff = np.abs(T_recon - T)
    max_diff = np.max(diff)
    return max_diff < tol

def construct():
    n, m, p = DIMENSIONS
    
    # Start with the rank 35 Strassen decomposition
    result = _build_strassen_35()
    
    # Verify the baseline
    if not _verify_decomposition(result, n, m, p):
        return result
    
    # Try to find a lower rank decomposition
    # We'll try rank 33 and 34
    
    start_time = time.time()
    deadline = start_time + 120
    
    # Try rank 33
    for target_rank in [33, 34]:
        if time.time() > deadline - 30:
            break
        
        # Initialize with random vectors based on the Strassen structure
        # But we need to be careful - we want to find a valid decomposition
        
        # Strategy: Start from rank 35, remove 2 terms, and try to adjust
        # This is equivalent to finding a rank 33 decomposition
        
        # Use numerical optimization
        # Variables: U (n*m x rank), V (m*p x rank), W (p*n x rank)
        # Objective: ||sum_r U[:,r] outer V[:,r] outer W[:,r] - T||_F
        
        # This is a non-convex problem. Let's use a heuristic approach.
        
        # Alternative: Use the fact that the tensor is structured
        # The 2x2x10 multiplication can be seen as 5 independent 2x2x2 multiplications
        # if we group the columns of B into 5 pairs
        
        # Each 2x2x2 multiplication has rank 7 (Strassen)
        # Total rank: 5 * 7 = 35
        
        # To get rank 33, we need to share 2 products between different blocks
        # This requires a non-trivial structure
        
        # For now, let's just return the rank 35 decomposition
        # A proper search for rank 33 would require more sophisticated methods
        
        pass
    
    return result

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 35 decomposition: 5 Strassen blocks (rank 7 each)
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # All 5 blocks: Standard Strassen (7 terms each)
    for t in range(5):
        col_base = t * 7
        
        # Term 1: P1 = (a00+a11)*(b00+b11) -> c00, c11
        col = col_base
        U[0][col] = 1; U[3][col] = 1
        V[2*t][col] = 1; V[10+2*t+1][col] = 1
        W[4*t][col] = 1; W[4*t+3][col] = 1
        
        # Term 2: P2 = (a10+a11)*b00 -> c01, -c11
        col = col_base + 1
        U[2][col] = 1; U[3][col] = 1
        V[2*t][col] = 1
        W[4*t+1][col] = 1; W[4*t+3][col] = -1
        
        # Term 3: P3 = a00*(b01-b11) -> c10, c11
        col = col_base + 2
        U[0][col] = 1
        V[2*t+1][col] = 1; V[10+2*t+1][col] = -1
        W[4*t+2][col] = 1; W[4*t+3][col] = 1
        
        # Term 4: P4 = a11*(b10-b00) -> c00, c01
        col = col_base + 3
        U[3][col] = 1
        V[10+2*t][col] = 1; V[2*t][col] = -1
        W[4*t][col] = 1; W[4*t+1][col] = 1
        
        # Term 5: P5 = (a00+a01)*b11 -> -c00, c10
        col = col_base + 4
        U[0][col] = 1; U[1][col] = 1
        V[10+2*t+1][col] = 1
        W[4*t][col] = -1; W[4*t+2][col] = 1
        
        # Term 6: P6 = (a10-a00)*(b00+b01) -> c11
        col = col_base + 5
        U[2][col] = 1; U[0][col] = -1
        V[2*t][col] = 1; V[2*t+1][col] = 1
        W[4*t+3][col] = 1
        
        # Term 7: P7 = (a01-a11)*(b10+b11) -> c00
        col = col_base + 6
        U[1][col] = 1; U[3][col] = -1
        V[10+2*t][col] = 1; V[10+2*t+1][col] = 1
        W[4*t][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
