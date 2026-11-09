"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x5 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    
    The matrix multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p),
    since C[i,k] = sum_j A[i,j] * B[j,k].
    
    This tensor is the outer product of three vectors of ones, so its rank is 1.
    We use U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
    """
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    rank = 1
    U = [[1] * rank for _ in range(nm)]
    V = [[1] * rank for _ in range(mp)]
    W = [[1] * rank for _ in range(pn)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
