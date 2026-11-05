"""Matrix multiplication 3,4,8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import null_space
import time
import math
import random
import sys

def construct():
    """
    Find a low-rank exact decomposition for the 3x4 x 4x8 matrix multiplication tensor.
    Returns dict with U (12 x rank), V (32 x rank), W (24 x rank).
    W uses trace convention: W[k*n+i, r].
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    nm, mp, pn = n*m, m*p, p*n  # 12, 32, 24
    
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Build the matrix multiplication tensor
    # T[i,j,k] = delta_{i,k} (for standard MM, but let's verify the convention)
    # Actually, for matrix multiplication (AB)_{ik} = sum_j A_{ij} B_{jk}
    # The tensor has T[i,j,k] = 1 if i==k... no that's wrong.
    # T[i,j,k] = 1 for all i,j,k is the trace tensor.
    # For MM: T[i,j,k] = delta_{i,k}? No.
    # (AB)_{ik} = sum_j A_{ij} B_{jk}
    # So the bilinear form is sum_{i,j,k} (AB)_{ik} * T_{i,j,k} * ... 
    # The tensor for MM is T[i,j,k] = 1 for all i,j,k.
    # Because (AB)_{ik} = sum_j A_{ij} B_{jk}, and we want to compute all entries.
    # The tensor is T_{i,j,k} = 1 for all i,j,k.
    
    # Actually let me reconsider. The standard MM tensor has:
    # T[i,j,k] = 1 if the product A_{ij} * B_{jk} contributes to (AB)_{i,k}
    # For standard MM, every (i,j,k) triple contributes to output (i,k),
    # so T[i,j,k] = 1 for all i,j,k.
    
    # The "official trace convention W[k*n+i,r]" means the output index is k*n+i,
    # which is consistent with computing C_{ik} where C = A*B.
    
    # So the tensor is all ones: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # This is the all-ones tensor of shape (n, m, p) = (3, 4, 8).
    
    # The rank of the all-ones tensor is 1!
    # U[i*m+j, r=0] = 1, V[j*p+k, r=0] = 1, W[k*n+i, r=0] = 1
    # Sum over r of U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1*1*1 = 1 = T[i,j,k]. ✓
    
    # Wait, but the seed program uses rank 96 = n*m*p and has U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
    # for each (i,j,k) as a separate column. That's the schoolbook rank.
    # But the all-ones tensor has rank 1.
    
    # Let me verify: if T[i,j,k] = 1 for all i,j,k, then
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # With rank 1: U[:, 0] = 1, V[:, 0] = 1, W[:, 0] = 1
    # Then sum = 1*1*1 = 1. ✓
    
    # So the minimal rank is 1.
    
    # But wait - is the task really about the all-ones tensor? Let me re-read.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 8 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    
    # The phrase "trace convention" is suspicious. In some formulations, the MM tensor
    # is defined differently. Let me think again.
    
    # Standard MM: C = AB, C_{ik} = sum_j A_{ij} B_{jk}
    # The tensor is T[i,j,k] where the product is sum_{i,j,k} A[i,j]*B[j,k]*T[i,j,k] giving C[i,k]
    # For MM, T[i,j,k] = 1 for all i,j,k (since A[i,j]*B[j,k] always contributes to C[i,k]).
    
    # Alternatively, maybe they define it as: compute trace(A*B) or something?
    # No, "multiplying a 3 by 4 matrix by a 4 by 8 matrix" clearly means standard matrix multiplication.
    
    # The "trace convention W[k*n+i,r]" just specifies how the output indices are ordered.
    # Output C has shape (n, p) = (3, 8), so there are n*p = 24 output entries.
    # The index k*n+i means: for output entry C[i,k] (row i, column k), 
    # the flat index is k*n + i. This is column-major ordering of C.
    
    # So the tensor is indeed all-ones, and rank 1 suffices.
    
    # Let me just return rank 1.
    
    rank = 1
    U = [[1.0] for _ in range(nm)]  # (12, 1)
    V = [[1.0] for _ in range(mp)]  # (32, 1)  
    W = [[1.0] for _ in range(pn)]  # (24, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
