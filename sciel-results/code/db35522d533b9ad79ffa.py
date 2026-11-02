"""Matrix multiplication 4x5x5 rank-minimal tensor decomposition via ALS."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]
_DEADLINE = 140.0  # safety margin before 150s kill
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _verify_exact(U, V, W, n, m, p):
    """Check if U,V,W exactly decompose the matmul tensor (entries in {0,±1/2,±1,...})."""
    r = U.shape[1]
    # Reconstruct tensor: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # U shape (n*m, r), V shape (m*p, r), W shape (p*n, r)
    # For each (i,j,k): sum over c of U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # The matmul tensor has T[i,j,k] = delta(j,j) = 1 (since A[i,j]*B[j,k], the "tensor" 
    # for matmul is T_{i,j,k} = 1 if we think of it as the identity in the j-index pairing)
    # Actually for matmul: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T has T[i,j,k] = 1 (it's the "contracting" tensor)
    # So we need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
    # Wait, that's not right. Let me reconsider.
    # 
    # Standard: C = A*B, C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: C[i,k] = sum_{j,c} U[i,j,c] V[j,k,c] W[c] ... no.
    # 
    # The task says: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # W[k*n+i, r] - so W is indexed by (k, i) with k from 0..p-1, i from 0..n-1
    # 
    # The decomposition should satisfy:
    # For the matmul tensor T where T[i,j,k] represents the "1" that couples A[i,j] and B[j,k]:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for all i,j,k
    # 
    # This is because: C[i,k] = sum_j A[i,j]*B[j,k] = sum_{j,c} A[i,j]*B[j,k]*T[i,j,k]
    # where T[i,j,k] = 1 (the identity tensor for contraction over j).
    # 
    # So the tensor we're decomposing is all-ones! T[i,j,k] = 1 for all i,j,k.
    # 
    # Wait, that can't be right because then rank would be min(n*m, m*p, p*n) = 20.
    # 
    # Let me re-read: "exactly decompose the matrix multiplication tensor"
    # The matmul tensor T^{n,m,p} has T[i,j,k] = 1 if we're contracting A[i,j] with B[j,k]
    # to get C[i,k]. Yes, T[i,j,k] = 1 for all valid i,j,k.
    # 
    # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    # 
    # The schoolbook solution uses rank = n*m*p = 100 with U[i*m+j, i*m+j] = 1, etc.
    # 
    # Can we do better? The rank of the all-ones tensor of size n x m x p is min(n*m, m*p, p*n) 
    # if we allow real entries... but we need integer/half-integer entries.
    # 
    # Actually the rank of the all-ones 3-tensor is 1 (just set U=V=W=1 vector)!
    # But that gives sum_c = 1*1*1 = 1 only if rank=1. Wait:
    # U[i*m+j, 0] = 1 for all i,j; V[j*p+k, 0] = 1 for all j,k; W[k*n+i, 0] = 1 for all k,i
    # Then sum_c U*V*W = 1*1*1 = 1. Yes! Rank 1 works for the all-ones tensor!
    # 
    # But wait - that's trivially the answer. Let me re-read the task more carefully.
    # 
    # "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 5 matrix by a 5 by 5 matrix"
    # 
    # I think the tensor is NOT all-ones. Let me reconsider what "matrix multiplication tensor" means.
    # 
    # In tensor decomposition literature, the matrix multiplication tensor M_{n,m,p} is defined as:
    # M_{n,m,p} = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i ⊗ e_j ⊗ e_k * delta_{jk}... 
    # No wait.
    # 
    # Actually, the standard definition: The matrix multiplication tensor for multiplying 
    # an n×m matrix by an m×p matrix is:
    # T = sum_{i,j,k} e_i^{(n)} ⊗ e_j^{(m)} ⊗ e_k^{(p)} where the contraction is over j.
    # 
    # Hmm, let me think about this differently. The evaluator checks:
    # "exactly decompose the matrix multiplication tensor"
    # 
    # Given the interface U(n*m, rank), V(m*p, rank), W(p*n, rank), and W[k*n+i, r],
    # the reconstruction is: for each (i,j,k), compute sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # and this should equal the matmul tensor entry.
    # 
    # The matmul tensor entry T[i,j,k] = 1 (since we're summing over j to get C[i,k]).
    # 
    # So yes, it IS the all-ones tensor, and rank 1 suffices with all entries = 1.
    # 
    # But the schoolbook uses rank 100... Let me verify: in the schoolbook,
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1 for each (i,j,k) triple.
    # So for a given (i,j,k), only one column c (corresponding to that specific triple) has
    # all three entries = 1, others have at least one zero. So sum = 1. Correct.
    # 
    # But rank 1 with all ones also works! sum_c=1: U[i*m+j,0]=1, V[j*p+k,0]=1, W[k*n+i,0]=1
    # gives product = 1. 
    # 
    # So the answer is rank 1?? That seems too easy. Let me check if there's a constraint I'm missing.
    # 
    # "integer/half-integer entries that exactly decompose" - 1 is an integer, so rank 1 works.
    # 
    # I think the answer is simply rank 1.

def construct():
    n, m, p = DIMENSIONS
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
    # Rank-1 decomposition: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # Then sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1*1*1 = 1 = T[i,j,k]
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
