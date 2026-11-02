"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix
    by a 4x8 matrix.
    
    We use the block structure: view the 4x8 matrix as two 4x4 blocks.
    The multiplication C = A * B where A is 3x4 and B is 4x8 can be seen as:
    C = A * [B1, B2] = [A*B1, A*B2]
    
    If we can decompose the 3x4x4 multiplication tensor with rank r, then
    the 3x4x8 tensor has rank at most 2*r (since we need to handle both blocks).
    
    Actually, let's think more carefully. The tensor for (n,m,p) matrix multiplication
    has entries T[i,j,k] = delta[i,j,k] where the multiplication is:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T is in R^{n*p} ⊗ R^{m*n} ⊗ R^{m*p} (or similar indexing).
    Actually, the standard indexing is:
    T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    
    The rank of the tensor is the minimum number of rank-1 tensors needed to express T.
    
    For schoolbook: rank = n*m*p = 3*4*8 = 96.
    
    Let's use Strassen's algorithm on the 2x2 blocks.
    
    Actually, let me think about this differently. We can decompose the 4x8 matrix
    into two 4x4 blocks. The multiplication A(3x4) * B(4x8) = C(3x8) can be written as:
    C[:, 0:4] = A * B[:, 0:4]
    C[:, 4:8] = A * B[:, 4:8]
    
    If we have a rank-r decomposition for 3x4x4, then we need 2r for 3x4x8.
    
    But maybe we can do better by exploiting the structure more directly.
    
    Let's try to use the known result that the rank of the 2x2x2 tensor is 7 (Strassen).
    
    For 3x4x8:
    - View A as 3x4, B as 4x8, C as 3x8
    - We can partition into 2x2 blocks where possible
    
    Let me try a different approach: use the fact that 4 = 2*2 and 8 = 2*2*2,
    and apply Strassen recursively.
    
    Actually, let me just implement a search for a good decomposition using
    known algorithms and results.
    
    For now, let's use the schoolbook decomposition as a baseline (rank 96),
    and try to improve it.
    """
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s safety margin
    
    # Start with schoolbook decomposition (rank 96)
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U, V, W, rank = schoolbook()
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try Strassen-based decomposition
    # The idea: partition the matrices into blocks and use Strassen's 2x2x2 rank-7 algorithm
    # For 3x4x8, we can view it as:
    # A is 3x4 = [A1, A2] where A1, A2 are 3x2
    # B is 4x8 = [B1; B2] where B1, B2 are 2x8
    # C is 3x8
    # C = A1*B1 + A2*B2
    # This is a sum of two 3x2x8 multiplications
    
    # Let's try to use the structure more cleverly.
    # We can write the 4x8 matrix as two 4x4 blocks:
    # B = [B_left, B_right] where B_left, B_right are 4x4
    # C = [A*B_left, A*B_right]
    # If we have a rank-r decomposition for 3x4x4, we get rank 2r for 3x4x8.
    
    # What's the best known rank for 3x4x4?
    # The schoolbook rank is 3*4*4 = 48.
    # Using Strassen on 2x2 blocks: 3*4x4 can be decomposed...
    
    # Let me try a direct approach: use the tensor decomposition literature.
    # For small dimensions, the optimal ranks are known or bounded.
    
    # Actually, let me try to construct a decomposition using the following idea:
    # The multiplication tensor T_{n,m,p} has a natural structure. We can try to
    # find a low-rank decomposition by exploiting symmetries and block structures.
    
    # For now, let's stick with schoolbook and return it.
    # The score is rank, so lower is better. Schoolbook gives 96.
    
    # Convert to list format as required
    def to_list(mat):
        return mat.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}

# EVOLVE-BLOCK-END
