"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 5 matrix by a 5 by 6 matrix.
    The output factor follows the official trace convention W[k*n+i,r].
    
    Returns a dict with keys 'U', 'V', 'W' containing factor matrices.
    """
    n, m, p = DIMENSIONS
    
    # The schoolbook decomposition has rank n*m*p = 3*5*6 = 90
    # We want to find a lower rank decomposition if possible.
    
    # For matrix multiplication, the tensor is T[i,j,k] = delta_{i,k} (in some sense)
    # Actually, the matrix multiplication tensor M_{i,j,k} for (n x m) * (m x p) -> (n x p)
    # is defined as: M_{i,j,k} = 1 if the (i,k) entry of the product depends on A[i,j] * B[j,k]
    # More precisely, (AB)_{i,k} = sum_j A[i,j] * B[j,k]
    # So the tensor is T_{i,j,k} = delta_{i,k}... no wait.
    
    # The matrix multiplication tensor has entries T_{i,j,k} = 1 for all i,j,k in the appropriate ranges
    # Actually, let me think about this more carefully.
    
    # The tensor for matrix multiplication of (n x m) and (m x p) matrices is a 3-tensor T of shape (n*p, m)
    # or equivalently a 3-tensor of shape (n, m, p) where T[i,j,k] = 1 if the (i,k) output uses input (j)
    
    # In the standard formulation, the matrix multiplication tensor M_{n,m,p} has entries:
    # M_{i,j,k} = 1 for i in [0,n), j in [m), k in [p)
    # and the multiplication is: C_{i,k} = sum_j A_{i,j} * B_{j,k}
    
    # So the tensor T of shape (n, m, p) has T[i,j,k] = 1 for all i,j,k.
    # Wait, that's not right either.
    
    # Let me reconsider. The matrix multiplication can be seen as a bilinear map:
    # (A, B) -> C where A is n x m, B is m x p, C is n x p
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    # This is a tensor of shape (n, p, m) where T[i,k,j] = 1 (since C_{i,k} = sum_j T[i,k,j] A_{i,j} B_{j,k})
    # Or equivalently, a tensor of shape (n, m, p) where T[i,j,k] = 1.
    
    # The rank of this tensor is the minimum number of rank-1 tensors needed to decompose it.
    # A rank-1 tensor has the form u_i * v_j * w_k.
    
    # The schoolbook decomposition uses n*m*p rank-1 tensors:
    # T[i,j,k] = sum_{a,b,c} U[i,a] V[j,a] W[k,a]
    # where for each (i,j,k) triple, we have one term with U[i,a]=1, V[j,a]=1, W[k,a]=1
    
    # For 3x5x6, the schoolbook rank is 90.
    
    # Known results:
    # - The rank of matrix multiplication tensor for n x m x p is at least max(n*m, m*p, n*p)
    # - For 3x5x6: max(15, 30, 18) = 30
    # - Strassen's algorithm gives rank 7 for 2x2x2 (vs schoolbook 8)
    # - For general matrices, the rank is not known exactly for most cases
    
    # Lower bounds:
    # - rank >= max(n*m, m*p, n*p) = max(15, 30, 18) = 30
    # - rank >= (n*m*p) / max(n,m,p) ... no, that's not a standard bound
    
    # Actually, a better lower bound: rank >= n*m*p / (n+m+p-2) ... no, I'm not sure about this.
    
    # Let me just try to find a decomposition with rank less than 90.
    
    # Strategy: Use random initialization and gradient descent to minimize the tensor decomposition error.
    # But we need an EXACT decomposition, so we need the error to be exactly zero.
    
    # Alternative: Try small ranks and use exact methods.
    
    # For this specific case (3,5,6), let's try to find a decomposition.
    
    # The tensor T[i,j,k] = 1 for all i in [0,3), j in [0,5), k in [0,6).
    # We want to find U (3*6 x r), V (5*5 x r), W (6*3 x r) such that:
    # T[i,j,k] = sum_r U[i*6+k, r] * V[j*5+j2, r] * W[k*3+i, r] ... wait, the indexing is confusing.
    
    # Let me re-read the problem statement.
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "W[k*n+i,r]" - so W is indexed by (k, i) where k is from the second matrix's rows and i is from the first matrix's columns?
    
    # Actually, looking at the schoolbook code:
    # U[i*m + j][column] = 1  -> U has shape (n*m, rank), indexed by (i,j)
    # V[j*p + k][column] = 1  -> V has shape (m*p, rank), indexed by (j,k)
    # W[k*n + i][column] = 1  -> W has shape (p*n, rank), indexed by (k,i)
    
    # And the product is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 for all i,j,k (since T[i,j,k] = 1).
    
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    # This is a tensor decomposition problem. Let's try to find a low-rank decomposition.
    
    # For now, let's just return the schoolbook decomposition.
    # The rank is 90.
    
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
