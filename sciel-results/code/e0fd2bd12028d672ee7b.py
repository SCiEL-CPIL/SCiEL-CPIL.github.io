"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Matrix multiplication 4x5 by 5x5.
    We use a block-diagonal decomposition.
    
    A is 4x5, B is 5x5. C = A @ B is 4x5.
    
    We can decompose the multiplication tensor.
    The trivial decomposition has rank 4*5*5 = 100.
    
    However, we can do better by noting that each column of C is a linear combination
    of the columns of A, where the coefficients come from the corresponding column of B.
    
    Specifically, C[:, k] = A @ B[:, k] for k=0..4.
    
    For each column k of B, we have 5 multiplications (one for each row of B's column,
    which corresponds to each row of A's columns... wait, let me think more carefully).
    
    Actually, let's think about the standard rank decomposition.
    
    The tensor for matrix multiplication (A_{n x m}, B_{m x p}) -> C_{n x p} has:
    - Input indices: (i, j) for A where i in [0,n), j in [0,m)
    - Input indices: (j, k) for B where j in [0,m), k in [0,p)
    - Output indices: (i, k) for C where i in [0,n), k in [0,p)
    
    The trivial decomposition uses rank n*m*p with:
    U[i*m+j, t] = 1, V[j*p+k, t] = 1, W[k*n+i, t] = 1 for each (i,j,k)
    
    For 4x5 * 5x5, this gives rank 100.
    
    Can we do better? The known lower bounds and upper bounds for matrix multiplication
    ranks:
    - For 4x4 * 4x4, the best known is 49 (Strassen-like).
    - For 2x2 * 2x2, the best known is 7 (Strassen).
    
    For 4x5 * 5x5, we can potentially use a block structure.
    
    Let's try a different approach. We can view the 4x5 matrix as having 4 rows of length 5,
    and the 5x5 matrix as having 5 rows of length 5.
    
    Actually, let me try to use the fact that we can factor through smaller matrices.
    
    One approach: use the identity that we can multiply a 4x5 by a 5x5 by first
    multiplying a 4x4 by a 4x4 (if we can embed), but the dimensions don't match nicely.
    
    Let me try a direct construction. The key insight is that for each output position (i,k),
    we need to compute sum_j A[i,j] * B[j,k].
    
    If we could use Strassen's algorithm ideas, we might reduce the rank.
    
    For now, let me try to see if we can get below 100.
    
    Actually, let me try a simple improvement: if we group the j-index, we can potentially
    share computations. But in a rank decomposition, each term is independent.
    
    Let me try rank = 4 * 5 * 5 = 100 first, and then see if I can find a better decomposition.
    
    Actually, I recall that for m x n * n x p matrices, the rank is at most m*n*p, but can be
    lower. For square matrices, Strassen gives 7 for 2x2.
    
    Let me try to implement a Strassen-like decomposition for 4x4 blocks.
    
    Wait, the dimensions are 4x5 and 5x5. Let me think about this differently.
    
    I'll try to use a decomposition that treats the 5x5 part as 4x4 + 1, or similar.
    
    Actually, let me just go with the trivial decomposition for now, as finding a
    better one requires careful algebraic work. The score is rank, so lower is better.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
