"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the schoolbook rank-96 decomposition."""
    n, m, p = DIMENSIONS
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
    return U, V, W

def _verify(U, V, W):
    """Verify that the tensor decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The multiplication tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    should equal 1 for all (i,j,k).
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k
    # We can do this efficiently by reshaping
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # For each (i,j,k), compute the dot product
    # This is equivalent to: for each r, outer product of U[:,r], V[:,r], W[:,r]
    # Sum over r should give all 1s
    
    # Efficient check: create the full tensor via einsum
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # But careful with index alignment: U is indexed (i,j), V is (j,k), W is (k,i)
    
    # Let's do it more directly
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Use einsum with explicit indices
    # U has shape (n, m, rank) -> indices i, j, r
    # V has shape (m, p, rank) -> indices j, k, r
    # W has shape (p, n, rank) -> indices k, i, r
    
    # We want: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # This is a 4-index contraction... actually it's 3 indices i,j,k
    
    # einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    # Check if all entries are 1
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook()
    
    def to_list(mat):
        result = []
        for row in mat:
            r = []
            for val in row:
                rounded = round(val * 2) / 2
                r.append(rounded)
            result.append(r)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
