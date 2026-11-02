"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 3x4 x 4x7 matrix multiplication."""
    n, m, p = 3, 4, 7
    deadline = time.time() + 140  # 10s safety margin
    
    # The matrix multiplication tensor T[i,j,k; i',j',k'] where
    # T has a 1 at (i,j,k) in the output if i=i', j=j', k=k'
    # More precisely, T = sum_{i,j,k} e_{i,j}^A ⊗ e_{j,k}^B ⊗ e_{i,k}^C
    
    # We want to find U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # such that for all (i,j,k):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r]  ... no, that's not right.
    
    # The correct condition: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # where U is indexed by (i,j) pairs, V by (j,k) pairs, W by (i,k) pairs.
    # So: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should reconstruct T.
    # But T is a 3-tensor, so the condition is:
    # For the tensor T with T[(i,j),(j,k),(i,k)] = 1 (and 0 otherwise in the appropriate sense)
    # Actually, the standard form: the multiplication tensor satisfies
    # <T, A ⊗ B> = AB (as a bilinear form)
    
    # In component form: T_{(i,j),(j,k),(i,k)} = 1 for the "diagonal" entries
    # The rank-1 decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r ∈ F^{n*m}, v_r ∈ F^{m*p}, w_r ∈ F^{p*n}
    # and the condition is that for all (i,j,k):
    #   sum_r u_r[i*m+j] * v_r[j*p+k] = w_r[k*n+i]  ... this isn't quite right either.
    
    # Let me use the standard verification: the tensor T_{abc} where a=(i,j), b=(j,k), c=(i,k)
    # T_{(i,j),(j,k),(i,k)} = 1, and T is 0 otherwise.
    # A rank-1 tensor is u ⊗ v ⊗ w with (u⊗v⊗w)_{(i,j),(j,k),(i,k)} = u_{i,j} * v_{j,k} * w_{i,k}
    # We need sum_r u_r[i,j] * v_r[j,k] * w_r[i,k] = T_{(i,j),(j,k),(i,k)}
    
    # But T_{(i,j),(j,k),(i,k)} is 1 if the indices are consistent, which they always are by construction.
    # Wait, no. The tensor T is defined on the full spaces F^{n*m} × F^{m*p} × F^{p*n}.
    # T(a,b,c) = sum_{i,j,k} a_{i,j} * b_{j,k} * c_{i,k} if this equals (AB)_{i,k}... 
    
    # I think the correct formulation is:
    # The matrix multiplication tensor M satisfies: for matrices A (n×m) and B (m×p),
    # (AB)_{i,k} = sum_j A_{i,j} B_{j,k} = <M, A ⊗ B>_{i,k}
    # So M is a 3-tensor with M_{i,j,k} = 1 (in the appropriate multi-index sense)
    
    # For the rank decomposition, we need:
    # sum_r U_{i*m+j, r} * V_{j*p+k, r} = W_{k*n+i, r}  ... no.
    
    # The correct condition for the decomposition to be valid:
    # For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] 
    # should equal 1 if (i,j,k) is a "valid" triple, but actually the tensor is:
    
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
    # A rank-1 decomposition term r contributes: U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # We need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    # And: sum_r U[a,r] * V[b,r] * W[c,r] = 0 for all other (a,b,c)
    
    # This is a very constrained problem. Let me just use the schoolbook and try to improve.
    
    # Schoolbook: rank = n*m*p = 84
    # U[i*m+j, i*m+j] = 1, V[j*p+k, i*m+j] = 1, W[k*n+i, i*m+j] = 1 for the (i,j,k) term
    
    # Let me try a block decomposition approach.
    # Split m=4 into two blocks of size 2.
    # For block 1 (j=0,1): compute 3x2 @ 2x7
    # For block 2 (j=2,3): compute 3x2 @ 2x7
    # Each 3x2x7 has schoolbook rank 42, total 84. No improvement.
    
    # Let me try to use a known lower-rank decomposition for small cases.
    # For 2x2x2, Strassen gives rank 7 instead of 8.
    # Can we adapt this?
    
    # Actually, let me just return the schoolbook decomposition for now,
    # but structure the code to allow future improvements.
    
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
# EVOLVE-BLOCK-END
