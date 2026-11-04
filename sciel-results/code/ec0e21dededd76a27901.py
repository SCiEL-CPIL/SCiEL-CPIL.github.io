"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a low-rank decomposition
    # The matrix multiplication tensor T has T[i,j,k] = 1 if we're multiplying (i,j) entry of A with (j,k) entry of B
    # Actually, T is defined such that (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor form: T_{i,j,k} where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] contributes to output[i,k] via A[i,j] * B[j,k]
    
    # So the tensor is: T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * delta (but actually it's just the identity in the middle)
    # More precisely: T_{(i,j),(j,k),(i,k)} = 1, and 0 otherwise
    # In the standard form: T[a,b,c] where a indexes A (i*m+j), b indexes B (j*p+k), c indexes C (k*n+i)
    # T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0
    
    # So T is a 3-tensor of size (n*m) x (m*p) x (p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # We want to decompose T as sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    
    # This is equivalent to finding matrices U (n*m x R), V (m*p x R), W (p*n x R)
    # such that for all i,j,k: T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # And for all other (a,b,c) not of the form (i*m+j, j*p+k, k*n+i), the sum should be 0
    
    # Let's try to find a decomposition with rank R
    # We can set up a nonlinear system of equations
    
    # Given time constraints, let's try known constructions
    # For now, return the schoolbook algorithm which we know works
    
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
