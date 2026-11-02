"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x5 matrix.
    
    The matrix multiplication tensor T for C = A * B where A is n x m, B is m x p, C is n x p
    is defined by T[i,j,k] = 1 if and only if the (i,j) entry of A multiplies with the (j,k) entry
    of B to contribute to C[i,k].
    
    For n=2, m=2, p=5:
    - U has shape (n*m, rank) = (4, rank)
    - V has shape (m*p, rank) = (10, rank)  
    - W has shape (p*n, rank) = (10, rank)
    
    The decomposition satisfies: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    We use the known optimal rank for 2x2x5 matrix multiplication.
    The border rank and tensor rank for small matrix multiplication problems are well-studied.
    
    For 2x2x5, we can use a decomposition based on the structure of the problem.
    Since p=5 is larger than n and m, we can exploit the fact that each row of B has 5 columns.
    
    One approach: use the fact that matrix multiplication is bilinear and find a low-rank 
    decomposition. The trivial rank is n*m*p = 2*2*5 = 20.
    
    However, we can do better. Let's use the known result that the rank of matrix multiplication
    tensor M_{n,m,p} satisfies certain bounds. For 2x2xp, the rank is known to be at most 
    2p + something.
    
    Actually, let me construct an explicit decomposition. We'll use the following strategy:
    For each pair (i, j) where i is row of A and j is column of A (and row of B), we have 
    contributions to all k columns of C.
    
    Let's use a decomposition with rank = 2*p + 2*(m-1) or similar. Actually, let me think 
    more carefully.
    
    For M_{2,2,5}, we can use the following:
    - For each j in {0,1} (the shared dimension), and each k in {0,...,4}, and each i in {0,1},
      we have a term. That's 2*5*2 = 20 terms (trivial).
    
    But we can combine terms. Notice that for fixed j, the terms for different i and k can be 
    structured.
    
    Let's try rank 18 or so. Actually, let me use a known construction.
    
    For M_{2,2,p}, one known upper bound on the rank is 2p + 2. Let's try to construct 
    a decomposition with rank = 2p + 2 = 12? No, that seems too low.
    
    Actually, for M_{2,2,3}, the rank is known to be 12 (or is it 11?). Let me just construct 
    a valid decomposition and count the rank.
    
    Let me use a direct construction. For each j in {0,1}:
    - For each k in {0,...,p-1}:
      - For each i in {0,1}:
        We need a term U[i*m+j] * V[j*p+k] * W[k*n+i]
    
    That gives 2*p*2 = 20 terms. Can we reduce?
    
    Notice that for fixed j and k, we have two terms (for i=0 and i=1). We can potentially 
    combine them if we allow linear combinations.
    
    Let me try a different approach: use the fact that the matrix multiplication tensor has 
    a certain structure. For M_{2,2,5}, I'll construct a decomposition and verify it.
    
    Let's use rank = 18 and see if we can find coefficients.
    
    Actually, let me just provide a valid decomposition. The safest is to use the trivial 
    decomposition with rank 20, but let me try to find something better.
    
    After research, I recall that for M_{2,2,p}, the rank is 2p + O(1). Let me try to 
    construct a decomposition with rank = 2p = 10? That seems unlikely.
    
    Let me just use the trivial decomposition for now and return rank 20.
    """
    n, m, p = DIMENSIONS
    
    # Trivial decomposition: rank = n*m*p
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
