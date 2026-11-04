import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x4 matrix by a 4x5 matrix.
    
    Strategy:
    Use the standard schoolbook decomposition which provides an exact
    decomposition with rank n*m*p.
    For n=3, m=4, p=5, the rank is 60.
    """
    n, m, p = 3, 4, 5
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Iterate over all indices to create rank-1 terms
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor T[i, j, k] = 1 if the term is A[i,j] * B[j,k] contributing to C[i,k]
    # In the factorization:
    # U has shape (n*m, rank), indexed by (i*m + j)
    # V has shape (m*p, rank), indexed by (j*p + k)
    # W has shape (p*n, rank), indexed by (k*n + i) -- Note: Task says W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_vec = np.zeros(n * m)
                v_vec = np.zeros(m * p)
                w_vec = np.zeros(p * n)
                
                # Set the 1s in the factor vectors
                u_vec[i * m + j] = 1.0
                v_vec[j * p + k] = 1.0
                w_vec[k * n + i] = 1.0
                
                U_cols.append(u_vec)
                V_cols.append(v_vec)
                W_cols.append(w_vec)
                
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return {'U': U, 'V': V, 'W': W}
