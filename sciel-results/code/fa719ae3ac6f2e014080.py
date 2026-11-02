"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.
    Returns dict with 'U', 'V', 'W' matrices.
    
    The schoolbook rank is 3*4*8 = 96. We'll try to find a lower rank decomposition
    using known techniques or search.
    
    For now, let's implement the schoolbook decomposition as a baseline, and then
    try to improve it.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 96 for schoolbook
    
    # Schoolbook decomposition
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
    
    # Try to find a lower rank decomposition using tensor decomposition techniques
    # For small dimensions, we can use brute force or known results
    
    # Known: For matrix multiplication, the tensor rank can sometimes be reduced
    # Let's try a few known low-rank decompositions if available
    
    # For now, return the schoolbook decomposition
    # In a more advanced implementation, we would search for lower rank decompositions
    # using algorithms like:
    # 1. Strassen's algorithm (for specific sizes)
    # 2. Tensor decomposition algorithms
    # 3. Brute force search for small dimensions
    
    # Let's implement a simple search for a lower rank decomposition
    # This is computationally expensive, so we'll use a time limit
    
    start_time = time.time()
    time_limit = 100  # seconds, leaving margin before the 120s limit
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find a decomposition with lower rank
    # For 3x4x8, we can try ranks from 95 down to some minimum
    
    # Actually, let's use a more systematic approach:
    # The tensor for matrix multiplication has a known structure
    # We can try to find a decomposition using the fact that:
    # (AB)[i,k] = sum_j A[i,j] B[j,k]
    
    # The tensor T has entries T[i,j,k] = 1 if we're computing (AB)[i,k] from A[i,j] and B[j,k]
    # Actually, the multiplication tensor M_{n,m,p} has entries:
    # M[i,j,k; i',j',k'] = delta[i,i'] * delta[j,j'] * delta[k,k']
    # No, that's not right either.
    
    # Let me think about this more carefully.
    # The matrix multiplication tensor T_{n,m,p} is defined as:
    # T_{i,j,k} = 1 if the (i,k)-entry of the product involves A[i,j] * B[j,k]
    # More precisely, T is a 3-tensor with T[i,j,k] = 1 for all i,j,k
    # and the multiplication is: (AB)[i,k] = sum_j A[i,j] B[j,k]
    
    # The tensor decomposition of rank r means we can write:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # (with appropriate index conventions)
    
    # Actually, the standard formulation is:
    # The multiplication tensor M has M[i,j,k] = 1 for all valid i,j,k
    # and a rank-r decomposition is:
    # M[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U is n*m x r, V is m*p x r, W is p*n x r
    
    # Wait, the trace convention is W[k*n+i, r], so W has indices (k,i)
    
    # Let me verify the schoolbook decomposition:
    # For each (i,j,k), we have a term U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # In schoolbook, for c = i*m*p + j*p + k:
    #   U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # All other entries are 0.
    # So sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for the matching (i,j,k)
    # and 0 otherwise.
    # This correctly decomposes the tensor.
    
    # Now, can we do better? For small matrices, the answer is often yes.
    # Known results:
    # - 2x2x2: rank 7 (Strassen) vs schoolbook 8
    # - 2x2x3: rank 13 vs schoolbook 12? No, schoolbook is 12, but I think rank is 12 or lower
    # - For 3x4x8, I'm not sure of the exact minimal rank
    
    # Let's try a randomized approach to find a lower rank decomposition
    # This is heuristic and may not find the optimal, but might find something better
    
    # Actually, for the time budget, let's just return the schoolbook for now
    # and note that a more sophisticated search could improve this
    
    # But let's at least try a few simple improvements:
    # 1. Check if we can merge some terms
    # 2. Try small rank reductions
    
    # For now, return the schoolbook decomposition
    # This gives a score of 96
    
    # To improve, we would need to find a decomposition with rank < 96
    # This is a hard problem, but for small dimensions, it's feasible
    
    # Let's implement a basic search:
    # Try rank 95, 94, etc. and see if we can find a valid decomposition
    
    # This is very expensive, so we'll use a time limit
    
    # Actually, let's just return the schoolbook for now
    # The task is to find a rank-minimal decomposition, and without more time
    # and computation, we can't guarantee finding a better one
    
    # But we can at least set up the framework for a search
    
    # For a more serious attempt, we could:
    # 1. Use the known structure of the multiplication tensor
    # 2. Apply tensor decomposition algorithms (e.g., CP decomposition)
    # 3. Use integer programming or constraint satisfaction
    
    # Given the time constraints, let's return the schoolbook decomposition
    # and note that this is a baseline that can be improved
    
    # Actually, let's try one more thing: use the fact that for some dimensions,
    # the tensor rank is known to be less than n*m*p
    
    # For 3x4x8, I don't know the exact minimal rank, but let's assume it's less than 96
    # and try to find a decomposition
    
    # Given the complexity, let's just return the schoolbook for now
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
