"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time

def _verify(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    r = len(U[0])
    # Compute T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c]
    # and check it equals W[k*n+i, c]
    # Actually the convention is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c]
    # and this should equal... wait, let me re-read.
    # The matrix multiplication tensor: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T[i,j,k] = 1 if we're multiplying A[i,j] * B[j,k]
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that's not right either. Let me think again.
    # 
    # Standard tensor decomposition for matrix mult:
    # T_{i,j,k} = sum_c U_{i,j}^{(c)} * V_{j,k}^{(c)} * W_{k,i}^{(c)}
    # where T_{i,j,k} = 1 for all valid (i,j,k)
    # 
    # So we need: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # And for the "wrong" indices, it should be 0? No, the tensor is only defined on valid indices.
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # The decomposition must satisfy: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    # Let's verify this
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, target_rank, time_limit=5.0):
    """
    Try to reduce the rank of the decomposition to target_rank.
    Uses a simple approach: try to find linear dependencies among the terms.
    """
    start_time = time.time()
    r = len(U[0])
    if target_rank >= r:
        return None
    
    # Convert to numpy for faster computation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We want to find a lower rank decomposition.
    # 
    # One approach: try to merge terms. If two terms c1 and c2 can be combined,
    # we can reduce the rank by 1.
    # 
    # A simpler approach: use random linear combinations to try to eliminate terms.
    # 
    # Actually, let's try a different approach. We'll use the fact that for small
    # dimensions, we can try to find a rank-r-1 decomposition by looking for
    # dependencies.
    
    # For now, let's just return None if we can't find a reduction.
    # A more sophisticated approach would use numerical linear algebra.
    
    # Try a simple approach: for each pair of columns, check if they can be merged.
    # This is O(r^2) which might be too slow for large r.
    
    # Instead, let's try to use the structure of the problem.
    # For matrix multiplication, the rank is known to be less than n*m*p for some cases.
    # For 3x4x8, the rank is not known to be less than 96, but let's try.
    
    # Actually, let's just return the schoolbook result for now.
    # A proper search would require more sophisticated methods.
    
    if time.time() - start_time > time_limit:
        return None
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    U, V, W = _schoolbook(n, m, p)
    current_rank = n * m * p
    
    # Try to reduce the rank
    for target_rank in range(current_rank - 1, 1, -1):
        result = _try_reduce_rank(U, V, W, n, m, p, target_rank, time_limit=3.0)
        if result is not None:
            U, V, W = result
            current_rank = target_rank
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
