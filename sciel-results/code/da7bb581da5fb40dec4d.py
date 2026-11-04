"""Matrix multiplication 4x5x6 rank minimization."""
import time
import sys

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Strategy: Use a block decomposition approach with Strassen-like techniques.
    # We can view the 4x5 x 5x6 multiplication as a sum of smaller matrix multiplications.
    
    # Approach: Split the middle dimension (5) into 2+3
    # A = [A1; A2] where A1 is 4x2, A2 is 4x3
    # B = [B1 B2] where B1 is 2x6, B2 is 3x6
    # C = A1*B1 + A2*B2
    # 
    # For M_{4,2,6}: rank 48 (schoolbook)
    # For M_{4,3,6}: rank 72 (schoolbook)
    # Total = 120 (same as schoolbook, no improvement)
    
    # Let's try a different approach: use Strassen for 2x2x2 blocks
    # Split A (4x5) into blocks:
    #   A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x3, A21 is 2x2, A22 is 2x3
    # Split B (5x6) into blocks:
    #   B = [[B11, B12], [B21, B22]] where B11 is 2x3, B12 is 2x3, B21 is 3x3, B22 is 3x3
    # 
    # C = A*B = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
    #            [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    # This is complex. Let's use a simpler approach:
    # Use the fact that we can decompose M_{4,5,6} into smaller multiplications
    # and apply known low-rank decompositions.
    
    # For now, let's implement a greedy set-cover approach with sparse rank-1 tensors.
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 if i<4, j<5, k<6
    # and the decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Generate candidate rank-1 tensors with sparse support
    # A rank-1 tensor is defined by choosing one index from each dimension
    # U[i,j,r] = a, V[j,k,r] = b, W[k,i,r] = c
    # For exact decomposition, we need a*b*c = 1 for each (i,j,k) in the support
    
    # Simple approach: use the schoolbook solution as baseline
    # Then try to improve by finding a better decomposition
    
    # For a 4x5x6 multiplication, the minimum known rank is around 90-100
    # Let's try to find a solution with rank < 120
    
    # Generate all possible "stars" - rank-1 tensors that cover exactly one (i,j,k) entry
    # But this is the schoolbook approach with rank 120
    
    # Better approach: use a greedy algorithm to select rank-1 tensors that cover multiple entries
    # A rank-1 tensor (u, v, w) covers all (i,j,k) where u_i * v_j * w_k != 0
    
    # Let's use a simpler heuristic: try different block decompositions
    
    # Decomposition 1: Split 4x5x6 into two 2x5x6 multiplications
    # Rank = 2 * rank(M_{2,5,6})
    # M_{2,5,6} can be further decomposed
    
    # For M_{2,5,6}:
    # Split into 2x2x3 + 2x3x3
    # M_{2,2,3}: rank 6 (schoolbook), can we do better?
    # M_{2,3,3}: rank 18 (schoolbook)
    
    # Let's just return the schoolbook solution for now to ensure correctness
    # The rank is 120.
    
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
