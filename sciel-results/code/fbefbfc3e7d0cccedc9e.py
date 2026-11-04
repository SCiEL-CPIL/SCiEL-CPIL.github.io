"""Matrix multiplication 4x5x5 tensor decomposition search."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank = n*m*p = 100
    rank_schoolbook = n * m * p
    
    # Build the multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1 if consistent
    # Actually T is defined such that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] = 1 if the indices are consistent (j from a matches j from b, k from b matches k from c, i from c matches i from a)
    
    # Let's build the target tensor as a 3D array
    # T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find a low-rank decomposition
    # Start with schoolbook and try to reduce
    
    # Known: for small matrices, rank can be significantly lower
    # Let's try random initialization with rank < 100 and see if we can verify
    
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook solution
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try lower ranks with random search
    np.random.seed(42)
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    for rank in range(99, 50, -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for trial in range(20):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Check if this is an exact decomposition
            # Reconstruct T and compare
            T_recon = np.zeros((n*m, m*p, p*n), dtype=float)
            for r in range(rank):
                T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
            if np.allclose(T_recon, T, atol=1e-10):
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                break
        else:
            continue
        break
    
    # Convert to list of lists with integer/half-integer if possible
    def to_output(mat):
        # Round to nearest half-integer if close
        rounded = np.round(mat * 2) / 2
        if np.allclose(rounded, mat, atol=1e-10):
            return rounded.tolist()
        return mat.tolist()
    
    return {
        "U": to_output(best_U),
        "V": to_output(best_V),
        "W": to_output(best_W)
    }
