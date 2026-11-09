import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 for all i,j,k.
    # The rank of the all-ones tensor of size n x m x p is min(n, m, p).
    # Here min(3, 4, 7) = 3.
    
    rank = min(n, m, p)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Construct a rank 1 decomposition for each of the 'rank' components
    # Since the tensor is all ones, we can just use one rank 1 tensor that is all ones,
    # but we need to distribute it? No, a single rank 1 tensor u ⊗ v ⊗ w 
    # has entries u_i v_j w_k. 
    # If u_i=1, v_j=1, w_k=1, then u_i v_j w_k = 1.
    # So rank 1 is sufficient?
    # Let's check.
    # U[i,j,0] = 1
    # V[j,k,0] = 1
    # W[k,i,0] = 1
    # Sum r=0: 1*1*1 = 1.
    # So rank 1 is sufficient!
    
    # Why did I think min(n,m,p)?
    # The rank of the all-ones tensor is 1.
    # Because 1*1*1 = 1.
    
    # Let's verify.
    # T = sum_r U_r V_r W_r.
    # If R=1, U_{ij} = 1, V_{jk} = 1, W_{ki} = 1.
    # Then (UVW)_{ijk} = 1*1*1 = 1.
    # Yes, rank 1 works.
    
    rank = 1
    
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
