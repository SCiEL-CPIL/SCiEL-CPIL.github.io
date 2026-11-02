"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    The tensor T[i,j,k] = 1 for all valid i,j,k (since multiplication always involves A[i,j]*B[j,k]).
    We seek rank r such that T = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    """
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = 140  # seconds, with margin before 150s limit
    
    # Build the target tensor: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This is the matrix multiplication tensor for C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Start with schoolbook (rank 63) as fallback
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
    
    # Try to find lower-rank decompositions
    # Strategy: Use the structure that for each j, the (i,k) block is all-ones.
    # We can try to share terms across j values.
    
    # Approach: Try ranks from lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But we need more due to the structure.
    
    # Let's try a constructive approach:
    # For each j, we have a 3x7 all-ones matrix in the (i,k) plane.
    # The tensor can be written as sum_j (e_i * e_j * e_k) for all i,j,k
    # = sum_j (sum_i e_i) * e_j * (sum_k e_k)
    # But this doesn't directly give a low-rank decomposition.
    
    # Better: Use the fact that the tensor is separable in j.
    # T[i,j,k] = sum_j delta_{j,j} * 1 * 1
    # Each term is (1 for all i) * (1 for this j) * (1 for all k)
    # But we need to express 1 (for all i) and 1 (for all k) as sums of rank-1 terms.
    
    # Actually, let's just try random initialization and optimization for various ranks.
    
    def verify_decomposition(U, V, W, tol=1e-6):
        """Check if U, V, W exactly decompose the tensor."""
        # Reconstruct tensor: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        r = U.shape[1]
        T_recon = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    T_recon[i, j, k] = val
        # Target is all ones
        return np.allclose(T_recon, 1.0, atol=tol)
    
    # Try a structured decomposition
    # Key insight: The tensor T[i,j,k] = 1 can be decomposed as:
    # For each j, T[i,j,k] = (sum over c in C_j) u_c[i] * v_c[j] * w_c[k]
    # where for fixed j, sum_c u_c[i] * w_c[k] = 1 for all i,k
    
    # For a fixed j, the 3x7 all-ones matrix has rank 1, so we can write:
    # 1_{i,k} = (1,1,1)^T * (1,1,1,1,1,1,1)
    # This uses 1 term per j, giving rank 3. But we need the middle factor to be v_c[j].
    
    # Actually, the decomposition is:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Note W depends on (k,i), not just k.
    
    # Let me reconsider the index structure:
    # U has indices (i,j) - row of A
    # V has indices (j,k) - column of B  
    # W has indices (k,i) - this is unusual, it's (output col, output row)
    
    # So W[k*n + i, c] means for output position (i,k), the factor depends on k and i.
    
    # For the schoolbook: W[k*n + i, c] = 1 when c corresponds to (i,j,k)
    
    # To reduce rank, I need to find c values where the same U[i,j,c], V[j,k,c], W[k,i,c]
    # pattern can cover multiple (i,j,k) triples.
    
    # Let's try a small-rank search with random initialization and gradient-free optimization.
    
    import random
    random.seed(42)
    np.random.seed(42)
    
    # Try ranks from 25 to 62
    for target_rank in range(25, 63):
        if time.time() - start_time > deadline:
            break
            
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() - start_time > deadline:
                break
                
            # Initialize with random values
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Simple iterative refinement: for each c, try to optimize
            # This is complex, so let's just check if a simple structured solution works.
            
            # Structured approach: 
            # Split the 7 columns into groups and use shared factors.
            
            # Alternative: Use the known 3x3 matrix mult rank-23 decomposition
            # and extend it.
            
            pass
    
    # Let's try a specific structured decomposition:
    # For 3x3 * 3x7, we can think of B as having 7 columns.
    # Use a decomposition that handles 3 columns at a time with rank-23 for 3x3,
    # but that's 23+23+9 = 55.
    
    # Better: Use a unified approach.
    # Let me try rank = 42 = 3 * 14, or rank = 45 = 3 * 15.
    
    # Actually, let's try a very simple structured decomposition:
    # For each j (0,1,2), we need to cover all (i,k) pairs with i in {0,1,2}, k in {0,...,6}.
    # The (i,k) sub-tensor for fixed j is all-ones, which is rank 1.
    # So for each j, we can use 1 term: U[i,j,c_j] = 1 for all i, V[j,k,c_j] = 1 for all k,
    # but W[k*n+i, c_j] must be 1 for all k,i.
    # This gives rank 3! But wait, W[k*n+i, c_j] = 1 for all k,i means the same W vector
    # is used for all (k,i) pairs, which is fine.
    
    # Let me verify: If I set:
    # For c = 0,1,2 corresponding to j = 0,1,2:
    #   U[i*m + j, c] = 1 if j == c, else 0  (so U[i*3+j, c] = delta_{j,c})
    #   V[j*p + k, c] = 1 if j == c, else 0  (so V[j*7+k, c] = delta_{j,c})
    #   W[k*n + i, c] = 1 for all k,i
    
    # Then T[i,j,k] = sum_c U[i*3+j, c] * V[j*7+k, c] * W[k*3+i, c]
    # = sum_c delta_{j,c} * delta_{j,c} * 1
    # = delta_{j,j} = 1
    
    # This works! Rank = 3.
    
    # But wait, is this valid? Let me double-check the index conventions.
    # U has shape (n*m, rank) = (9, rank), indexed by (i*m + j)
    # V has shape (m*p, rank) = (21, rank), indexed by (j*p + k)
    # W has shape (p*n, rank) = (21, rank), indexed by (k*n + i)
    
    # For c = 0 (j=0):
    #   U[i*3 + 0, 0] = 1 for all i, U[i*3 + 1, 0] = 0, U[i*3 + 2, 0] = 0
    #   V[0*7 + k, 0] = 1 for all k, V[1*7 + k, 0] = 0, V[2*7 + k, 0] = 0
    #   W[k*3 + i, 0] = 1 for all k,i
    
    # For c = 1 (j=1):
    #   U[i*3 + 1, 1] = 1 for all i, others 0
    #   V[1*7 + k, 1] = 1 for all k, others 0
    #   W[k*3 + i, 1] = 1 for all k,i
    
    # For c = 2 (j=2):
    #   U[i*3 + 2, 2] = 1 for all i, others 0
    #   V[2*7 + k, 2] = 1 for all k, others 0
    #   W[k*3 + i, 2] = 1 for all k,i
    
    # Then for T[i,j,k]:
    # sum_c U[i*3+j, c] * V[j*7+k, c] * W[k*3+i, c]
    # Only c=j contributes (since U[i*3+j, c] = delta_{j,c} and V[j*7+k, c] = delta_{j,c})
    # = U[i*3+j, j] * V[j*7+k, j] * W[k*3+i, j]
    # = 1 * 1 * 1 = 1
    
    # This is correct! Rank = 3.
    
    # But this seems too good. Let me re-examine the problem statement.
    # "Matrix multiplication 3,3,7" - multiplying a 3x3 matrix by a 3x7 matrix.
    # The tensor should represent C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] should be 1 for all valid i,j,k, which is what we have.
    
    # The decomposition T = sum_c U_c * V_c * W_c where:
    # (U_c * V_c * W_c)[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # With our construction, this equals 1 for all i,j,k. So rank 3 is valid.
    
    # But wait, can we do even better? Rank 1?
    # For rank 1: T[i,j,k] = U[i*m+j] * V[j*p+k] * W[k*n+i]
    # This would require U[i*m+j] * V[j*p+k] * W[k*n+i] = 1 for all i,j,k.
    # Setting j=0: U[i*3] * V[k] * W[k*3+i] = 1 for all i,k
    # Setting j=1: U[i*3+1] * V[7+k] * W[k*3+i] = 1 for all i,k
    # Dividing: U[i*3] * V[k] = U[i*3+1] * V[7+k] for all i,k
    # This is hard to satisfy. Likely rank 1 is not possible.
    
    # So rank 3 is likely optimal or near-optimal.
    
    # Let me verify rank 3 is correct and return it.
    
    rank = 3
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c in range(rank):
        j = c
        # U[i*m + j, c] = 1 for all i
        for i in range(n):
            U[i * m + j, c] = 1
        # V[j*p + k, c] = 1 for all k
        for k in range(p):
            V[j * p + k, c] = 1
        # W[k*n + i, c] = 1 for all k, i
        for k in range(p):
            for i in range(n):
                W[k * n + i, c] = 1
    
    # Verify
    if not verify_decomposition(U, V, W, tol=1e-6):
        # Fallback to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
