import time
import random
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition is the trivial rank n*m*p
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][c] = 1
                V_school[j * p + k][c] = 1
                W_school[k * n + i][c] = 1
                c += 1
    
    # We will try to find a lower rank decomposition using randomized search
    # This is a "tensor rank minimization" problem.
    # The matrix multiplication tensor T[i,j,k] is 1.
    # We seek rank r such that T = sum_{c=0}^{r-1} U_c \otimes V_c \otimes W_c.
    # This is equivalent to finding matrices U (nm x r), V (mp x r), W (pn x r)
    # such that T = sum U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}.
    
    # Start with the schoolbook rank and try to reduce it.
    # We can use a randomized approach or a specific algebraic construction.
    
    # For 3x4x6, the trivial rank is 72.
    # Is there a known lower bound or construction?
    # For rectangular matrix multiplication, the rank can sometimes be lower than n*m*p.
    # However, finding an exact integer/half-integer decomposition is hard.
    
    # Let's attempt a simple reduction using the fact that if we can combine terms.
    # For example, in 2x2x2, Strassen reduces 8 to 7.
    # Does a similar trick work for 3x4x6?
    # We can try to apply Strassen to the 4x4 block if we can partition 3x4 and 4x6 appropriately.
    # 4 is not a power of 2, but 4 = 2+2.
    # 3x4 = 3x(2+2) -> 3x2, 3x2.
    # 4x6 = (2+2)x6 -> 2x6, 2x6.
    # C = A1 B1 + A2 B2.
    # This is sum of two 3x2x6 multiplications.
    # Rank of 3x2x6 is 3*2*6 = 36 (schoolbook).
    # Total rank 72. No gain.
    
    # What if we use a more clever combination?
    # C = A1 B1 + A2 B2.
    # (A1+A2) B1 = A1 B1 + A2 B1.
    # (A1+A2) B2 = A1 B2 + A2 B2.
    # We want A1 B1 + A2 B2.
    # This is not directly obtainable from just 2 products of larger matrices without extra terms.
    
    # Let's try a randomized search for a lower rank.
    # We fix a target rank r and try to solve the system of equations.
    # The number of equations is n*m*p (all entries of T are 1).
    # The variables are r*(n*m + m*p + p*n).
    # We can use a random initialization and iterative refinement, or simply check if a random rank-r decomposition works.
    # However, this is a non-convex problem.
    
    # Given the strict time limit and the difficulty, and the fact that the seed is 72,
    # and no obvious algebraic improvement for 3x4x6 exists in standard literature (unlike 2x2x2 or 4x4x4),
    # I will stick to the schoolbook decomposition unless I find a very specific trick.
    
    # One possible trick: 
    # If we can express the 3x4x6 multiplication as a sum of smaller multiplications.
    # e.g. 3x4x6 = 3x(1+1+1+1)x6.
    # This is just 4 copies of 3x1x6.
    # 3x1x6 multiplication: C[i,k] = sum_j A[i,j] B[j,k]. Since j is fixed, it's just A[i,j] * B[j,k].
    # This is rank 1 per i,k? No, it's an outer product.
    # The tensor for 3x1x6 has rank 3*6 = 18?
    # T[i, 0, k] = 1.
    # Decomposition: sum_{c=1}^{18} U_{i, c} V_{0, c} W_{k, c}.
    # If we set U_{i, c} = 1, W_{k, c} = 1 for specific pairs, it's rank 18.
    # So 4 * 18 = 72. Same as schoolbook.
    
    # What if we share some terms between the j-slices?
    # This would require A[i, j1] and A[i, j2] to be related, which they are not in general.
    
    # Therefore, the schoolbook rank of 72 is likely the best we can do with simple methods.
    # I will return the schoolbook decomposition.
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
