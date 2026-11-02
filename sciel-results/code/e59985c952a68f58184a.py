"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4x8 matrix multiplication.
    Strategy: Use the Kronecker product structure M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}.
    We know rank(M_{3,4,4}) = 20 (Winograd).
    M_{1,1,2} is just scalar multiplication by a 1x2 vector, which has rank 2.
    So total rank should be 20 * 2 = 40.
    
    However, let's verify if we can do better or if 40 is indeed achievable.
    The tensor for M_{n,m,p} is T[i,j,k] = e_{i,j} ⊗ e_{k,j} ⊗ e_{k,i} (depending on convention).
    Here we have A (3x4) * B (4x8) = C (3x8).
    C[i,k] = sum_j A[i,j] B[j,k].
    
    The Kronecker product approach:
    M_{3,4,8} can be viewed as (M_{3,4,4}) ⊗ (M_{1,1,2}).
    If we have a rank-r decomposition for M_{3,4,4} with factors U4, V4, W4,
    and a rank-s decomposition for M_{1,1,2} with factors U2, V2, W2,
    then the Kronecker product gives a rank-r*s decomposition for M_{3,4,8}.
    
    Let's construct the rank-20 decomposition for M_{3,4,4} first.
    Winograd's algorithm for 3x4x4 uses 20 multiplications.
    
    Actually, let's use a more direct approach. Since we need an exact decomposition with integer/half-integer entries,
    and we want to minimize rank, let's try to find a low-rank decomposition using random search or known identities.
    
    For now, let's implement the Kronecker product of a known rank-20 decomposition for 3x4x4 and rank-2 for 1x1x2.
    """
    deadline = time.time() + 120
    n, m, p = 3, 4, 8
    
    # Step 1: Get rank-20 decomposition for 3x4x4
    # Winograd's 3x4x4 algorithm
    # This is a well-known result. Let's construct it.
    # The 20 multiplications in Winograd's 3x4x4:
    # We can use the fact that it's based on 2x2x2 Strassen-like blocks.
    
    # Actually, let's use a simpler approach: schoolbook for 3x4x8 has rank 96.
    # We want to reduce this.
    
    # Let's try a Las Vegas approach: random search for low-rank decompositions.
    # But first, let's try the Kronecker product approach.
    
    # For M_{1,1,2}: C[0,0] = A[0,0] * B[0,0] + A[0,0] * B[0,1] ... wait, no.
    # M_{1,1,2}: A is 1x1, B is 1x2, C is 1x2.
    # C[0,k] = A[0,0] * B[0,k] for k=0,1.
    # This is just: C[0,0] = A[0,0]B[0,0], C[0,1] = A[0,0]B[0,1].
    # Rank 2 decomposition:
    # U2[0,0]=1, U2[0,1]=1 (A factor, shape 1x2)
    # V2[0,0]=1, V2[1,0]=0, V2[0,1]=0, V2[1,1]=1 (B factor, shape 2x2)
    # W2[0,0]=1, W2[1,0]=0, W2[0,1]=0, W2[1,1]=1 (C factor, shape 2x2)
    # Check: sum_c U2[0,c]*V2[j,c]*W2[k,c]
    # c=0: 1*1*1=1 for (j=0,k=0), 1*0*0=0 for others... wait, W2[k*n+i, c] = W2[k*1+0, c] = W2[k,c]
    # For c=0: U[0,0]=1, V[j,0], W[k,0]. V[0,0]=1, V[1,0]=0. W[0,0]=1, W[1,0]=0.
    # So term is 1*V[j,0]*W[k,0].
    # For (j=0,k=0): 1*1*1=1. Correct for C[0,0].
    # For (j=1,k=0): 1*0*1=0. Correct.
    # For (j=0,k=1): 1*1*0=0. Correct.
    # For (j=1,k=1): 1*0*0=0. Correct.
    # For c=1: U[0,1]=1, V[j,1], W[k,1]. V[0,1]=0, V[1,1]=1. W[0,1]=0, W[1,1]=1.
    # Term is 1*V[j,1]*W[k,1].
    # For (j=0,k=0): 1*0*0=0.
    # For (j=1,k=0): 1*1*0=0.
    # For (j=0,k=1): 1*0*1=0.
    # For (j=1,k=1): 1*1*1=1. Correct for C[0,1].
    
    # So M_{1,1,2} has rank 2.
    
    # Now for M_{3,4,4}, we need rank 20.
    # Let's use a known rank-20 decomposition.
    # Actually, let's just use schoolbook for now and see if we can improve.
    
    # Schoolbook for 3x4x8: rank = 3*4*8 = 96.
    
    # Let's try to find a better decomposition using random search.
    # We'll use the fact that the tensor is defined over rationals.
    
    # For now, return schoolbook decomposition.
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
# EVOLVE-BLOCK-END
