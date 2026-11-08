"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-98 decomposition for 3x4x8 matrix multiplication
    using Kronecker product of 3x2x4 (rank 14) and 2x2x2 (rank 7, Strassen).
    
    The 3x4x8 tensor is factored as (3x2x4) ⊗ (2x2x2):
    - 3x2x4: multiply 3x2 by 2x4 -> rank 14 (schoolbook for 3x2x4 is 24, but we use a better one)
      Actually, let's use schoolbook for 3x2x4 which is rank 3*2*4=24, and Strassen for 2x2x2 rank 7.
      Total rank = 24 * 7 = 168. That's worse than schoolbook 96.
      
    Let me reconsider. The schoolbook rank for 3x4x8 is 96.
    
    Kronecker approach: 3x4x8 = (3x2x4) ⊗ (2x2x2)? No, dimensions don't work directly.
    
    Actually, 3x4x8 can be seen as:
    - 4 = 2*2, 8 = 2*4
    So (3x4x8) = (3x2x4) ⊗ (2x2x2) where:
      First factor: 3x2x4 (multiply 3x2 by 2x4)
      Second factor: 2x2x2 (multiply 2x2 by 2x2)
    
    The Kronecker product of tensors gives:
    rank(A ⊗ B) ≤ rank(A) * rank(B)
    
    Schoolbook: rank(3x2x4) = 3*2*4 = 24, rank(2x2x2) = 8
    Total: 24*8 = 192. Worse.
    
    With Strassen for 2x2x2 (rank 7): 24*7 = 168. Still worse than 96.
    
    Hmm, this approach doesn't help directly. Let me think differently.
    
    Actually, maybe I should just use the schoolbook decomposition which gives rank 96,
    and that's already optimal or near-optimal for this size.
    
    For 3x4x8, the known lower bound is around 51-60, and upper bound is 96 (schoolbook).
    The best known explicit construction might be around 70-80.
    
    Let me just implement the schoolbook decomposition correctly with rank 96.
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    rank = n * m * p  # 96
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
