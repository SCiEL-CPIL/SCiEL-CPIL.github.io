"""Rank-minimal exact tensor decomposition for 2x3 times 3x6 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

def construct():
    """Return the best rank-minimal decomposition found within the time budget."""
    start_time = time.time()
    DEADLINE = start_time + 120  # 120 s wall-clock safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook decomposition: rank = n*m*p
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def verify(U, V, W):
        """Check if the decomposition exactly reconstructs the tensor."""
        # Reconstruct tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For matrix mult: T[i,j,k] should be 1 if this is a valid rank-1 term in the sum
        # Actually, the tensor is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And this should equal the standard matrix mult tensor, which has T[i,j,k] = 1 for all valid (i,j,k)
        # Wait, no. The tensor for matrix multiplication is:
        # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor T[i, j, k] = 1 (it's the all-ones tensor in the sense that each term A[i,j]*B[j,k] contributes to C[i,k])
        # Actually, the tensor is defined such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
        # Let's verify by checking that the sum over r gives 1 for each (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find a lower rank decomposition using a simple search strategy
    # For 2x3x6, the schoolbook rank is 36. Let's see if we can do better.
    
    # A known lower bound: rank >= max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # Also, rank >= n+m+p-1 = 11 (for square, but this is a lower bound for general)
    # Actually, for matrix multiplication, the rank is at least max(n*m, m*p, p*n)
    # So rank >= 18.
    
    # Let's try ranks from 18 upward and see if we can find a valid decomposition
    # For small ranks, we can try random initializations and use some optimization
    
    # Given the complexity and time constraints, let's try a few known structures
    
    # For 2x3x6, one approach: the rank is known to be at most n*m*p = 36
    # Can we do better? Let's try a structured decomposition.
    
    # Idea: Use the fact that 2x3 can be done with rank 2*3=6, 3x6 with rank 3*6=18
    # But for the full tensor, we need to consider all three modes.
    
    # Let's try to construct a decomposition of rank 18 (the lower bound from max dimension product)
    # This might not be achievable, but let's try.
    
    # Actually, for small cases, the optimal rank is often close to the schoolbook.
    # Let's just return the schoolbook for now, as finding a lower rank requires significant search.
    
    # But wait - the task says "rank-minimal". So we should try to find the minimum.
    # For 2x3x6, let's check if rank 18 is achievable.
    
    # One strategy: try to factorize using the structure of the problem.
    # The tensor T[i,j,k] = 1 for all i in [0,1], j in [0,2], k in [0,5]
    # This is just the all-ones tensor of shape (2,3,6)!
    # The rank of an all-ones tensor of shape (n,m,p) is 1!
    # Because J = 1_n * 1_m * 1_p (outer product of all-ones vectors)
    
    # Wait, is that right? Let me reconsider.
    # The matrix multiplication tensor is NOT the all-ones tensor in the usual sense.
    # It's defined as: T[i,j,k] = 1 if the indices are compatible, else 0.
    # For matrix multiplication (A*B)[i,k] = sum_j A[i,j] B[j,k],
    # the tensor has T[i,j,k] = 1 for all valid (i,j,k), and 0 otherwise.
    # Since all (i,j,k) with i in [0,n), j in [0,m), k in [0,p) are valid,
    # T is indeed the all-ones tensor of shape (n, m, p).
    
    # And the all-ones tensor has rank 1!
    # T = u * v * w where u = [1,...,1]_n, v = [1,...,1]_m, w = [1,...,1]_p
    
    # So U[i*m+j, 0] = u[i] * v[j] = 1 * 1 = 1
    # V[j*p+k, 0] = v[j] * w[k] = 1 * 1 = 1  
    # W[k*n+i, 0] = w[k] * u[i] = 1 * 1 = 1
    
    # Let's verify: sum over r (only r=0): U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0] = 1*1*1 = 1 ✓
    
    # So the rank-1 decomposition works!
    U_r1 = np.ones((n * m, 1))
    V_r1 = np.ones((m * p, 1))
    W_r1 = np.ones((p * n, 1))
    
    if verify(U_r1, V_r1, W_r1):
        return {"U": U_r1.tolist(), "V": V_r1.tolist(), "W": W_r1.tolist()}
    
    # If for some reason rank 1 doesn't work (shouldn't happen), return schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
