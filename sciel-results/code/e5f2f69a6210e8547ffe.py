"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _build_target_tensor():
    """Build the matrix multiplication tensor T[i, j, k] = 1 for all i,j,k."""
    n, m, p = DIMENSIONS
    # T has shape (n, m, p) with all entries 1
    # The decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Wait, the convention is W[k*n+i, r]
    # So T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Since T is all ones, we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
    T = np.ones((n, m, p), dtype=np.int64)
    return T

def _verify_decomposition(U, V, W, T=None):
    """Verify that the decomposition exactly reconstructs the target tensor."""
    n, m, p = DIMENSIONS
    if T is None:
        T = _build_target_tensor()
    
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    
    # Check shapes
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # Reconstruct T
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
    # This is a bit tricky with einsum
    # U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Let's use einsum: 'ijc,jkc,kic->ijk'
    T_reconstructed = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # Check if T_reconstructed equals T
    if not np.allclose(T_reconstructed, T, atol=1e-10, rtol=1e-10):
        return False
    
    # Also check that entries are integer or half-integer
    # The problem says "integer/half-integer entries"
    # Let's check if 2*U, 2*V, 2*W are all close to integers
    for matrix in [U, V, W]:
        if not np.allclose(2 * matrix, np.round(2 * matrix), atol=1e-10, rtol=1e-10):
            return False
    
    return True

def _try_rank1_decomposition():
    """Try to find a rank-1 decomposition (trivial, won't work for non-trivial tensors)."""
    n, m, p = DIMENSIONS
    # For rank 1, we need U[i,j] * V[j,k] * W[k,i] = 1 for all i,j,k
    # This is only possible if U, V, W are all 1, but then we need to check consistency
    # Actually, for the all-ones tensor, rank 1 would mean:
    # U[i*m+j] * V[j*p+k] * W[k*n+i] = 1 for all i,j,k
    # Let U[i,j] = a_i * b_j, V[j,k] = c_j * d_k, W[k,i] = e_k * f_i
    # Then a_i*b_j*c_j*d_k*e_k*f_i = (a_i*f_i) * (b_j*c_j) * (d_k*e_k) = 1
    # So we need a_i*f_i = alpha, b_j*c_j = beta, d_k*e_k = gamma with alpha*beta*gamma = 1
    # We can set alpha=beta=gamma=1
    # a_i = 1, f_i = 1, b_j = 1, c_j = 1, d_k = 1, e_k = 1
    # So U[i,j] = 1, V[j,k] = 1, W[k,i] = 1
    # Let's verify: 1*1*1 = 1. Yes!
    # Wait, this means the all-ones tensor has rank 1!
    # But that can't be right for matrix multiplication...
    
    # Actually, the matrix multiplication tensor is NOT the all-ones tensor.
    # The tensor T[i,j,k] represents the coefficient of x_{i,k} in (A*B)_{i,k} where
    # A is n x m and B is m x p.
    # (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    # So T[i,j,k] = 1 for all i,j,k.
    
    # Hmm, so the tensor IS the all-ones tensor. And it has rank 1?
    # Let me re-examine. The matrix multiplication tensor for n x m times m x p
    # has entries T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B
    # contribute to the (i,k) entry of AB.
    
    # Yes, T[i,j,k] = 1 for all valid i,j,k.
    # And this tensor has rank 1: U[i,j] = 1, V[j,k] = 1, W[k,i] = 1.
    # Check: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1*1*1 = 1. Correct!
    
    # But wait, this seems too good to be true. Let me double-check with a known result.
    # The rank of the matrix multiplication tensor M_{n,m,p} is known to be at most n*m*p
    # and at least max(n*m, m*p, p*n). For n=m=p=2, the rank is 7 (Strassen).
    # For the all-ones tensor, the rank should be 1.
    
    # I think I'm confusing the matrix multiplication tensor with the all-ones tensor.
    # Let me re-read the problem.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix 
    #  by a 4 by 4 matrix."
    
    # The tensor for matrix multiplication is indeed T[i,j,k] = 1 for all i,j,k,
    # because each entry A_{i,j} * B_{j,k} contributes to C_{i,k}.
    
    # So the tensor is the all-ones tensor, which has rank 1.
    
    # But that contradicts known results. Let me think again...
    
    # Ah, I see the issue. The tensor T is defined such that:
    # C_{i,k} = sum_{j} T[i,j,k] * A_{i,j} * B_{j,k}
    # where T[i,j,k] = 1 for all i,j,k.
    
    # The rank of this tensor is the minimum number of rank-1 tensors that sum to T.
    # A rank-1 tensor has the form u_i * v_j * w_k.
    
    # For the all-ones tensor, we can write it as 1*1*1, so rank 1.
    
    # But wait, I recall that the matrix multiplication tensor M_{2,2,2} has rank 7,
    # not 1. So what's going on?
    
    # I think the confusion is that the "matrix multiplication tensor" in the literature
    # is not the all-ones tensor, but rather the tensor that encodes the bilinear map.
    # The bilinear map is: (A, B) -> A*B, where A is n x m and B is m x p.
    # This bilinear map can be represented as a trilinear form:
    # sum_{i,j,k} C_{i,k} * A_{i,j} * B_{j,k} = sum_{i,j,k} A_{i,j} * B_{j,k} * C_{i,k}
    # where C = A*B.
    
    # The tensor T is defined by T[i,j,k] = 1 for all i,j,k, and the bilinear form is:
    # sum_{i,j,k} T[i,j,k] * A_{i,j} * B_{j,k} * C_{i,k}
    
    # The rank of this tensor is the minimum number of rank-1 terms needed to represent it.
    # A rank-1 term is of the form alpha * u_i * v_j * w_k.
    
    # So we need: sum_c alpha_c * u_{c,i} * v_{c,j} * w_{c,k} = 1 for all i,j,k.
    
    # This is equivalent to finding the minimum number of rank-1 tensors that sum to
    # the all-ones tensor.
    
    # The all-ones tensor J_{n,m,p} has rank 1, because J = 1*1*1.
    
    # So the rank of the matrix multiplication tensor M_{n,m,p} is 1?
    
    # That can't be right. Let me look this up...
    
    # Actually, I think the issue is that the "rank" in the context of matrix multiplication
    # is not the tensor rank of the all-ones tensor, but rather the rank of the bilinear map.
    # The bilinear map (A, B) -> A*B is a map from R^{n*m} x R^{m*p} to R^{n*p}.
    # The rank of this bilinear map is the minimum number of rank-1 bilinear maps
    # that sum to it.
    
    # A rank-1 bilinear map is of the form (A, B) -> (u^T A v) * (w^T B z) * ...
    
    # Hmm, I'm getting confused. Let me just try the rank-1 decomposition and see if it works.
    
    n, m, p = DIMENSIONS
    rank = 1
    U = [[1]]
    V = [[1]]
    W = [[1]]
    
    if _verify_decomposition(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        return None

def _schoolbook_decomposition():
    """Schoolbook decomposition of rank n*m*p."""
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

def construct():
    """Construct a rank-minimal exact tensor decomposition for matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    # First, try rank 1
    result = _try_rank1_decomposition()
    if result is not None:
        return result
    
    # If rank 1 doesn't work, try the schoolbook decomposition
    result = _schoolbook_decomposition()
    if _verify_decomposition(result["U"], result["V"], result["W"]):
        return result
    
    # This should never happen, but just in case
    raise ValueError("Failed to find a valid decomposition")
# EVOLVE-BLOCK-END
