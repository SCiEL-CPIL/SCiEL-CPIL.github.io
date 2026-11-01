"""Matrix multiplication 2x2 x 2x13: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Standard n*m*p rank decomposition."""
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


def _check_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c] for all i,j,k."""
    # The tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] should equal W[k*n+i,c]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    # Compute tensor: T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c]
    # Use einsum
    T = np.einsum('ijc,jkc->ijk', U3, V3)
    # Compare with W3: W3[k,i,c]... wait, W is indexed as W[k*n+i, c]
    # So W3[k, i, c] where the tensor entry for (i,j,k) should be W3[k, i, ?]
    # Actually the decomposition says: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is T[i,j,k] = A[i,j]*B[j,k], and we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c]  ... NO
    # The correct condition: the multiplication tensor M[i,j,k] = delta (meaning M[i,j,k]=1 always for valid indices)
    # A decomposition is: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, re-reading the task: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # The tensor for matrix multiplication (n x m) * (m x p) -> (n x p) is:
    # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p), and T[i,j,k] = 1 (it's the identity tensor)
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # So we need to check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k
    
    rank = U.shape[1]
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Reshape: U3[i,j,c], V3[j,k,c], W3[k,i,c]
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # = einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    T_computed = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    max_err = np.max(np.abs(T_computed - 1.0))
    return max_err < tol


def _try_block_decomposition(n, m, p, deadline):
    """
    Try to find a lower-rank decomposition by exploiting block structure.
    For 2x2 * 2x13:
    A = [[a11, a12], [a21, a22]]  (2x2)
    B = [[b11,...,b1,13], [b21,...,b2,13]]  (2x13)
    C = A*B  (2x13)
    
    C[0,k] = a11*b1k + a12*b2k
    C[1,k] = a21*b1k + a22*b2k
    
    The tensor has indices (i, j, k) with i in {0,1}, j in {0,1}, k in {0,...,12}
    T[i,j,k] = 1 for all valid i,j,k.
    
    Rank lower bound: The tensor is the "outer product" structure. 
    For each k, the 2x2 slice T[:,:,k] is the all-ones 2x2 matrix.
    The 2x2 matrix multiplication tensor has rank 7 (Strassen).
    But here we have 13 copies of the 2x2 slice, coupled through the j index.
    
    Actually, let's think differently. The tensor T[i,j,k] = 1 for all i,j,k.
    This is simply the all-ones tensor of shape (2,2,13).
    The rank of the all-ones tensor of shape (a,b,c) is max(a,b,c) when... no.
    
    Actually the all-ones tensor of shape (n,m,p) has rank max(n,m,p)? No.
    The all-ones tensor J[i,j,k] = 1 for all i,j,k can be written as:
    J[i,j,k] = sum_{c=1}^{r} u_c[i] * v_c[j] * w_c[k]
    
    If we set u_1 = [1,1,...,1], v_1 = [1,1,...,1], w_1 = [1,1,...,1], then
    sum gives 1 only if r=1 and all vectors are all-ones. But then u_1[i]*v_1[j]*w_1[k] = 1.
    So the all-ones tensor has rank 1!
    
    Wait, that can't be right for matrix multiplication. Let me re-read the task.
    
    "Matrix multiplication 2,2,13" - this is the tensor for multiplying a 2x2 matrix by a 2x13 matrix.
    The multiplication tensor M satisfies: (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    So M[i,j,k] is the coefficient of A[i,j]*B[j,k] in the product, which is 1 for all i,j,k.
    
    So yes, M[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,12}.
    
    And the all-ones tensor of shape (2,2,13) has rank 1:
    M[i,j,k] = 1*1*1 with U[i*m+j,0]=1, V[j*p+k,0]=1, W[k*n+i,0]=1.
    
    Let me verify: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    = U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] = 1*1*1 = 1. Yes!
    
    So rank 1 suffices!
    """
    # Rank 1 decomposition: all entries are 1
    rank = 1
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    
    if _check_decomposition(U, V, W, n, m, p):
        return U, V, W
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # First try rank 1 (the all-ones tensor)
    result = _try_block_decomposition(n, m, p, deadline)
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
