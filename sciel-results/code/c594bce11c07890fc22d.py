"""Matrix multiplication 3x5x7: search for low-rank exact tensor decomposition."""
import time
import random
import sys
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 135  # safety margin before 150s kill
    n, m, p = DIMENSIONS

    # Build the multiplication tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    #
    # Strategy: start with schoolbook (rank=n*m*p), then try to merge columns.
    # Two columns (r1, r2) can be merged if there exists a linear combination
    # that preserves all tensor entries.

    # Schoolbook: column c = (i,j,k) has U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    #
    # Key insight: if we have two columns c1=(i1,j1,k1) and c2=(i2,j2,k2),
    # we can try to replace them with a single column that has:
    #   U[i*m+j] = a (if (i,j) appears in either), etc.
    #
    # Actually, a better approach: use the fact that for fixed (i,j), the sum over k
    # must equal 1 for each k. So if we group by (i,j), we need the V and W parts
    # to work out.
    #
    # Let me think differently. The tensor T[i,j,k] = 1 for all i,j,k.
    # A decomposition of rank r means: T[i,j,k] = sum_{c=1}^{r} U[i,j,c] * V[j,k,c] * W[k,i,c]
    #
    # Since T is all-ones, this is equivalent to:
    # For all i,j,k: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    #
    # One simple family: let U[i,j,c] = u_c (depends only on c), V[j,k,c] = v_c, W[k,i,c] = w_c
    # Then sum_c u_c*v_c*w_c = 1, which is rank 1! But wait, that's only valid if
    # the decomposition separates. Let's check: T[i,j,k] = sum_c u_c*v_c*w_c.
    # This equals 1 for all i,j,k if and only if sum_c u_c*v_c*w_c = 1.
    # So rank 1 works! With u_1=1, v_1=1, w_1=1.
    #
    # Wait, but that means the tensor is rank 1? Let me re-examine.
    # T[i,j,k] = 1 for ALL i,j,k. Yes! The all-ones tensor is rank 1.
    # U[i,j,1] = 1, V[j,k,1] = 1, W[k,i,1] = 1.
    # sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1*1*1 = 1. Correct!
    #
    # But wait - is this the right tensor? Matrix multiplication (A*B)[i,k] = sum_j A[i,j]*B[j,k].
    # The tensor for this is T[i,j,k] = 1 (coefficient of A[i,j]*B[j,k] in product[i,k]).
    # Yes, all coefficients are 1. So the tensor is indeed the all-ones tensor of shape (3,5,7).
    # And the all-ones tensor has tensor rank 1!
    #
    # Let me verify with the output convention: W[k*n+i, r].
    # U has shape (n*m, rank) = (15, rank), indexed by i*m+j
    # V has shape (m*p, rank) = (35, rank), indexed by j*p+k
    # W has shape (p*n, rank) = (21, rank), indexed by k*n+i
    #
    # For rank 1: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # Check: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1*1*1 = 1 = T[i,j,k]. Correct!

    # The rank-1 decomposition is exact and minimal (rank >= 1 since T is nonzero).
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
