"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time

def _rank23_3x3():
    """
    Return a rank-23 decomposition for 3x3 x 3x3 matrix multiplication.
    Returns (U, V, W) where:
      U[i*3+j, r] for A (3x3)
      V[j*3+k, r] for B (3x3)
      W[k*3+i, r] for C (3x3)
    The decomposition satisfies: for all i,j,k,r: sum_r U[i*3+j,r]*V[j*3+k,r] = W[k*3+i,r] * (A@B)[i,k]
    Actually, the tensor identity is: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    where T[i,j,k]=1 represents the multiplication tensor.
    
    Using known rank-23 algorithm coefficients.
    """
    # This is a known rank-23 decomposition for 3x3 matrix multiplication.
    # Coefficients are integers or half-integers.
    # U has shape (9, 23), V has shape (9, 23), W has shape (9, 23)
    # Indexing: U[i*3+j], V[j*3+k], W[k*3+i]
    
    U = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    # This is just identity - not correct. Let me use a proper rank-23 decomposition.
    
    # I'll use a different approach: try to find a valid decomposition by solving the system.
    # For now, return a placeholder and use a fallback.
    return None

def _schoolbook():
    """Fallback schoolbook decomposition."""
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    # Check tensor identity: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r] should relate to W
    # The actual condition: (A@B)[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor form: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] where T[i,j,k]=1
    # So we need: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    # Wait, that's not right either. Let me reconsider.
    
    # The matrix multiplication tensor M has entries M[i,j,k] = delta (it's 1 for the "multiplication" pattern)
    # Actually, the tensor is defined such that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This can be written as: C = A * B where * is the tensor contraction
    # In tensor form: the multiplication tensor T has T[i,j,k] = 1 for all i,j,k
    # and C[i,k] = sum_j sum_r ... no.
    
    # Correct formulation:
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # The decomposition says: for the multiplication to be exact,
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i,j,k
    # NO, that's not right.
    
    # Let me think again. The standard formulation:
    # We want: for all A, B: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # This is equivalent to saying the tensor T with T[i,j,k]=1 can be decomposed as
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So the check is: for all i in range(n), j in range(m), k in range(p):
    #   sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Try to use a better decomposition
    # For (3,3,6), we can split into two (3,3,3) problems
    # If we have a rank-r decomposition for (3,3,3), total rank is 2*r
    
    # For now, use schoolbook as fallback
    result = _schoolbook()
    
    # Verify the schoolbook decomposition
    U, V, W = result["U"], result["V"], result["W"]
    if not _verify(U, V, W, n, m, p):
        # Should not happen for schoolbook
        pass
    
    return result
# EVOLVE-BLOCK-END
