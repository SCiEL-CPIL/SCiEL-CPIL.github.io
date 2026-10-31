import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the 49-rank BCLR decomposition for 4x4x4 matrix multiplication.
    
    The decomposition satisfies:
    T[i,j,k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
    
    This uses a recursive Strassen-like construction:
    1. View 4x4 matrices as 2x2 blocks of 2x2 matrices.
    2. Apply Strassen's 7-multiplication algorithm at the block level.
    3. Each block multiplication is a 2x2x2 multiplication, which is also decomposed
       using Strassen's 7-multiplication algorithm.
    4. Total rank = 7 (outer) * 7 (inner) = 49.
    
    Uses NumPy Kronecker products for efficient construction.
    """
    # Strassen coefficients for 2x2 multiplication
    # beta[s]: coefficient of A in s-th product
    # gamma[s]: coefficient of B in s-th product
    # delta[s]: coefficient of C from s-th product
    
    beta = np.array([
        [[1, 0], [0, 1]],  # M0: (A00+A11)
        [[0, 0], [1, 1]],  # M1: (A10+A11)
        [[1, 0], [0, 0]],  # M2: A00
        [[0, 0], [0, 1]],  # M3: A11
        [[1, 1], [0, 0]],  # M4: (A00+A01)
        [[-1, 0], [1, 0]], # M5: (A10-A00)
        [[0, 1], [0, -1]]  # M6: (A01-A11)
    ], dtype=int)
    
    gamma = np.array([
        [[1, 0], [0, 1]],  # M0: (B00+B11)
        [[1, 0], [0, 0]],  # M1: B00
        [[0, 1], [0, -1]], # M2: (B01-B11)
        [[-1, 0], [1, 0]], # M3: (B10-B00)
        [[0, 0], [0, 1]],  # M4: B11
        [[1, 1], [0, 0]],  # M5: (B00+B01)
        [[0, 0], [1, 1]]   # M6: (B10+B11)
    ], dtype=int)
    
    delta = np.array([
        [[1, 0], [0, 1]],  # M0: C00+=1, C11+=1
        [[0, 0], [1, -1]], # M1: C10+=1, C11-=1
        [[0, 1], [0, 1]],  # M2: C01+=1, C11+=1
        [[1, 0], [1, 0]],  # M3: C00+=1, C10+=1
        [[-1, 1], [0, 0]], # M4: C00-=1, C01+=1
        [[0, 0], [0, 1]],  # M5: C11+=1
        [[1, 0], [0, 0]]   # M6: C00+=1
    ], dtype=int)
    
    rank = 49
    # U, V, W are 16x49 matrices
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # U[i*4+j] corresponds to A[i,j]
            # i = 2*I + i', j = 2*J + j'
            # U[2I+i', 2J+j'] = beta[s][I,J] * beta[t][i',j']
            # This is the Kronecker product beta[s] @ beta[t]
            # The resulting 4x4 matrix has row index i and col index j.
            # Flattening row-major gives index i*4+j.
            U[:, col] = np.kron(beta[s], beta[t]).flatten()
            
            # V[j*4+k] corresponds to B[j,k]
            # j = 2*J + j', k = 2*K + k'
            # V[2J+j', 2K+k'] = gamma[s][J,K] * gamma[t][j',k']
            # This is the Kronecker product gamma[s] @ gamma[t]
            # Flattening row-major gives index j*4+k.
            V[:, col] = np.kron(gamma[s], gamma[t]).flatten()
            
            # W[k*4+i] corresponds to C[i,k] (output index is k*n+i = k*4+i)
            # Note: The tensor is T[i,j,k] = sum U[i,j] V[j,k] W[k,i]
            # The problem statement says: T[i,j,k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
            # So W is indexed by k, i.
            # W[k*4+i] = delta[s][I,K] * delta[t][i',k']
            # where i = 2*I + i', k = 2*K + k'
            # Let M = kron(delta[s], delta[t]). M is 4x4.
            # M[row, col] = delta[s][I,K] * delta[t][i',k'] where row=2I+i', col=2K+k'.
            # So M[i, k] = delta[s][I,K] * delta[t][i',k'].
            # We need W[k*4+i] to be this value.
            # If we flatten M, we get M[0,0], M[0,1]... which is index i*4+k.
            # We need index k*4+i.
            # This is equivalent to flattening M.T (transpose).
            # M.T[k, i] = M[i, k].
            # Flatten M.T row-major: M.T[0,0], M.T[0,1]... index k*4+i.
            W[:, col] = np.kron(delta[s], delta[t]).T.flatten()
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
