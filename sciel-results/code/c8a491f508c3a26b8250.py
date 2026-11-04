"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank = 30
    rank_school = n * m * p
    
    # Build the multiplication tensor T as a 3-way tensor T[i,j,k] = e_{(i,k)} in the A->B->C space
    # More precisely: T is a tensor of shape (n*m, m*p, p*n) where
    # T[i*m+j, j*p+k, k*n+i] = 1, and 0 otherwise.
    # This is the standard matrix multiplication tensor.
    
    # Let's represent T as a 3D numpy array
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Try to find a lower rank decomposition
    # Start from rank 25 and try to go up
    # For each rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that sum_c U[:,c] outer V[:,c] outer W[:,c] == T
    
    # A simpler approach: use the fact that for matrix multiplication,
    # we can try to find decompositions using random initialization and 
    # alternating minimization, then round to rationals.
    
    # But first, let's try a known result: 
    # The minimum rank for (2,3,5) matrix multiplication.
    # Known lower bounds and constructions:
    # - Schoolbook: 30
    # - We can try to find if rank < 30 is possible.
    
    # Let's implement a simple peeling algorithm:
    # 1. Start with R = T
    # 2. Try to peel off rank-1 terms
    # 3. A rank-1 term is U_c outer V_c outer W_c
    
    # For a 3-way tensor, peeling is hard. Let's use a different approach.
    
    # Approach: Use the "slice rank" or "tensor rank" via flattening.
    # The rank of T is at least the rank of any flattening.
    # Flatten T along the first axis: (n*m) x (m*p * p*n)
    # The rank of this flattening is a lower bound on the tensor rank.
    
    T_flat1 = T.reshape(n*m, m*p*p*n)
    rank_lb1 = np.linalg.matrix_rank(T_flat1)
    
    T_flat2 = T.transpose(1,0,2).reshape(m*p, n*m*p*n)
    rank_lb2 = np.linalg.matrix_rank(T_flat2)
    
    T_flat3 = T.transpose(2,0,1).reshape(p*n, n*m*m*p)
    rank_lb3 = np.linalg.matrix_rank(T_flat3)
    
    # The tensor rank is at least max(rank_lb1, rank_lb2, rank_lb3)
    # For matrix multiplication tensor, the slice rank (which equals min of these flattening ranks)
    # is a lower bound on the rank.
    
    # For (2,3,5): 
    # rank_lb1 = rank of 6 x (15*10=150) matrix
    # rank_lb2 = rank of 15 x (6*10=60) matrix
    # rank_lb3 = rank of 10 x (6*15=90) matrix
    
    # Let's compute these
    # Actually, for the matrix multiplication tensor, the slice rank is known.
    # The slice rank of the (n,m,p) matmul tensor is max(n*m, m*p, p*n) = max(6,15,10) = 15.
    # But the tensor rank can be larger than the slice rank.
    
    # Let's just try to construct a decomposition of rank < 30 using a systematic search.
    
    # Given the complexity, let's just return the schoolbook decomposition for now,
    # but try a few known constructions.
    
    # Actually, let's try a simple approach: use the schoolbook but see if we can
    # combine some terms.
    
    # For now, return schoolbook.
    rank = rank_school
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, column] = 1
                V[j*p+k, column] = 1
                W[k*n+i, column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
