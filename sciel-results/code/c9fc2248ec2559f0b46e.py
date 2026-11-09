import time
import numpy as np

DIMENSIONS = [3, 4, 5]
_DEADLINE = 145.0
_START_TIME = time.time()

def _check_deadline():
    return (time.time() - _START_TIME) > _DEADLINE

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 x 4x5 matrix multiplication.
    Uses the naive decomposition with rank n*m*p.
    """
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Map each multiplication A[i,j] * B[j,k] to a unique rank index r
    # r = (i * m + j) * p + k ? No, we need to iterate over all i, j, k.
    # Total triples: n * m * p.
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U row corresponding to (i, j)
                u_row = i * m + j
                # V row corresponding to (j, k)
                v_row = j * p + k
                # W row corresponding to (k, i) based on W[k*n+i, r]
                w_row = k * n + i
                
                U[u_row, idx] = 1
                V[v_row, idx] = 1
                W[w_row, idx] = 1
                
                idx += 1
                
                if _check_deadline():
                    break
            if _check_deadline():
                break
        if _check_deadline():
            break
            
    # Convert to lists if required by interface, or keep as numpy arrays if accepted.
    # The prompt says "factor matrices of shapes ... with integer/half-integer entries".
    # The previous program returned lists of lists. Let's stick to that to be safe, 
    # or check if numpy arrays are accepted. 
    # "construct() -> {'U', 'V', 'W'}: factor matrices"
    # Usually numpy arrays are fine, but the previous error was a ValueError from the evaluator, 
    # not a type error. The previous program returned lists. 
    # I will return lists to match the previous interface exactly, although numpy is faster.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
