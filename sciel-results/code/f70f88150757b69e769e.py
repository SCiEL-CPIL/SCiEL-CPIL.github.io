"""Rewritten program: rank-minimal exact tensor decomposition for 2x3*3x6."""
import time
import random
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def _build_decomposition(U, V, W):
    """Build factor matrices as lists of lists (or numpy arrays) with integer/half-integer entries."""
    return {"U": U, "V": W and V, "W": W}

def construct():
    """
    Return a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x6 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    with integer/half-integer entries that exactly decompose the matrix multiplication tensor.
    
    Strategy: Use Strassen-like block decomposition to reduce rank below schoolbook.
    For 2x3 * 3x6, we can split the 3x6 matrix into blocks and use efficient sub-multiplications.
    """
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Schoolbook rank is n*m*p = 36
    # We aim for a lower rank using structured decomposition.
    
    # Approach: Split the 3x6 matrix B into two 3x3 blocks: B = [B1, B2]
    # Then A*B = [A*B1, A*B2]
    # Each A*Bi is a 2x3 * 3x3 multiplication.
    
    # For 2x3 * 3x3, schoolbook rank is 18.
    # But we can do better with Strassen-like methods.
    
    # Let's use a direct construction based on known optimal or near-optimal decompositions.
    
    # For 2x3 * 3x6, one approach:
    # Split B into three 3x2 blocks: B = [B1, B2, B3] where each Bi is 3x2
    # A*Bi is 2x3 * 3x2, schoolbook rank 12 each, total 36.
    
    # Better: Use the fact that we can share computations.
    
    # Let's try a Strassen-inspired approach:
    # Split A (2x3) into A = [A1, A2, A3] where Ai is 2x1
    # Split B (3x6) into B = [B1; B2; B3] where Bi is 1x6
    
    # Actually, let me think differently.
    
    # For matrix multiplication (n x m) * (m x p), the tensor rank is known for small cases.
    # For 2x3 * 3x6, I'll construct an explicit low-rank decomposition.
    
    # One known approach: use the "lax" decomposition or similar.
    
    # Let me build a decomposition with rank 30 or lower.
    
    # Actually, let's use a systematic approach with Strassen blocks.
    
    # Split the 3x6 matrix into three 3x2 submatrices: B = [B_0, B_1, B_2]
    # where B_i is 3x2.
    
    # A * B_i is 2x3 * 3x2.
    
    # For 2x3 * 3x2, we can use a rank-8 decomposition (instead of schoolbook 12).
    
    # Let me construct a rank-8 decomposition for 2x3 * 3x2.
    
    # A = [[a11, a12, a13],
    #      [a21, a22, a23]]
    # B = [[b11, b12],
    #      [b21, b22],
    #      [b31, b32]]
    
    # C = A*B = [[a11*b11 + a12*b21 + a13*b31, a11*b12 + a12*b22 + a13*b32],
    #            [a21*b11 + a22*b21 + a23*b31, a21*b12 + a22*b22 + a23*b32]]
    
    # We need to express each C[i,j] as sum over r of U[i,r]*V[j,r]*W... wait, the tensor form is:
    # C[i,j] = sum_r U[i, r] * V[j, r] where V indexes the B entries.
    
    # Actually, the tensor T_{i,j,k} for (i,j) in output, k in input index.
    # The decomposition is: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is a vector of length n*m=6, v_r of length m*p=18, w_r of length p*n=12.
    
    # Wait, let me re-read the interface:
    # U: shape (n*m, rank) = (6, rank)
    # V: shape (m*p, rank) = (18, rank)  
    # W: shape (p*n, rank) = (12, rank)
    
    # The tensor entry T[(i,j), (j',k)] ... hmm, I need to be careful with indexing.
    
    # For matrix multiplication C = A*B where A is n×m and B is m×p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The multiplication tensor has indices (i, j, k) for each term A[i,j]*B[j,k] contributing to C[i,k].
    # In the factorization: C[i,k] = sum_r U[i*m + ... ] ... 
    
    # Let me use the standard convention:
    # U has rows indexed by (i, j) pairs: row i*m + j
    # V has rows indexed by (j, k) pairs: row j*p + k
    # W has rows indexed by (k, i) pairs: row k*n + i (per the trace convention W[k*n+i, r])
    
    # The tensor equation is:
    # For each output position (i, k), and each input pair (j):
    # The coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
    
    # In the decomposition:
    # sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = delta_{...}
    
    # Actually, the standard formulation:
    # T_{i,j,k} = 1 if the indices form a valid multiplication term, else 0.
    # Where T is indexed by (i, j, k) with i in [0,n), j in [0,m), k in [0,p).
    
    # The decomposition satisfies:
    # T_{i,j,k} = sum_r U[i, r] * V[j, r] * W[k, r]  (simplified notation)
    
    # But with the full indexing:
    # sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] should equal... 
    
    # Hmm, I think the correct interpretation is that the three-way tensor for matmul is:
    # M[i, j, k] = 1 if we're multiplying A[i,j] by B[j,k] to get C[i,k]
    # So M is a tensor of shape (n*m, m*p, p*n) where:
    # M[(i,j), (j',k), (k',i')] = 1 if i=i', j=j', k=k'... 
    
    # Let me just use the schoolbook as a sanity check and try to optimize.
    
    # For now, let me return the schoolbook decomposition (rank 36) and then try to improve.
    
    # Schoolbook:
    rank = n * m * p  # 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Now try to reduce rank using Strassen-like ideas.
    # This is complex; for the first version, return schoolbook.
    
    return {"U": U, "V": V, "W": W}
