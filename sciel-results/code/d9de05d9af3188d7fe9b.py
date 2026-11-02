"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time

# Known 23-term tensor decomposition for 3x3x3 matrix multiplication.
# Each term is (U_vec, V_vec, W_vec) of length 9.
# The convention is C[i,j] = sum_r A[i,j'] * B[j',j]
# U term for (i,j), V term for (j,k), W term for (k,i)
# C[k*n+i] = sum_j A[i*m+j] * B[j*p+k]  (for n=m=p=3)
# Actually the standard is: (AB)[i,j] = sum_k A[i,k] B[k,j]
# So C[i,j] depends on A[i,k] and B[k,j]
# U has shape (n*m, rank) = (9, rank): row i*m+j
# V has shape (m*p, rank) = (9, rank): row k*p+j (wait, m=3, p=3, so row k*p+j where k is middle index)
# Actually V row is j*p+k? Let's check the schoolbook:
# U[i*m+j], V[j*p+k], W[k*n+i]
# C[k*n+i] = A[i*m+j] * B[j*p+k]
# So C is indexed by [k*n+i], A by [i*m+j], B by [j*p+k]
# This means: C = B^T * A? Or C[k,i] = sum_j A[i,j] B[j,k]
# Yes, C[k,i] = (B A)_{k,i}? No.
# Let's trace: C[k*n+i] gets contribution from A[i*m+j] * B[j*p+k].
# If we think of A as 3x3 with A[i,j] and B as 3x3 with B[j,k],
# then C[k,i] = sum_j A[i,j] B[j,k].
# So C = B^T * A^T? No. C[k,i] = (B A)_{k,i} if B is 3x3 and A is 3x3.
# Actually (B A)_{k,i} = sum_j B[k,j] A[j,i]. That's not it.
# C[k,i] = sum_j A[i,j] B[j,k] = (A B)_{i,k} if we view A as (i,j) and B as (j,k).
# So C[k,i] = (A B)^T [k,i] = (B^T A^T)[k,i].
# Anyway, the convention is fixed. I need to find 23 vectors (u, v, w) of length 9
# such that for all i,j,k: A[i,j]*B[j,k]*C[k,i] coefficient is 1.
# i.e., sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = delta for the right indices.
# Actually the tensor T[i,j,k] = 1 if we want (AB)[i,k] = sum_j A[i,j]B[j,k].
# So T[i,j,k] = 1 for all i,j,k. The tensor is the identity tensor? No.
# The matrix multiplication tensor T_matmul[i,j,k] = 1 if j is the middle index.
# It's a 3-tensor where T[i,j,k] = 1 for all i,j,k in {0,1,2}.
# Wait, no. The tensor for matrix multiplication is such that
# (A*B)[i,k] = sum_j A[i,j] B[j,k].
# So the tensor has T[i,j,k] = 1 for all i,j,k.
# That's just the all-ones tensor? No, that's not right either.
# The tensor is defined by T(i,j,k) = delta such that
# sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C^T[k,i] = 0? 
# Let me just use the standard: the rank of the 3x3x3 matrix multiplication tensor is 23.
# I need 23 triples of vectors (u_r, v_r, w_r) in F^9 each such that
# sum_r u_r[i*3+j] * v_r[j*3+k] * w_r[k*3+i] = 1 for all i,j,k.

# Since I cannot recall the exact 23 terms, I will use a different approach:
# Use the known rank-23 decomposition from the literature.
# Actually, I will just use the schoolbook rank 54 for 3x3x3, and replicate for 3x3x6.
# For 3x3x6: split into two 3x3x3 blocks. Rank = 2 * 54 = 108.
# But the schoolbook for 3x3x6 is 3*3*6 = 54. Wait, 3*3*6 = 54.
# So schoolbook is already 54 for 3x3x6.

# Let me think about whether I can do better than 54 for 3x3x6.
# 3x3x6 can be seen as multiplying a 3x3 matrix by a 3x6 matrix.
# This is like doing two 3x3 multiplications simultaneously (columns 0-2 and 3-5).
# So rank is at most 2 * rank(3x3x3) = 2 * 23 = 46.
# So I need the rank-23 decomposition of 3x3x3.

# The rank-23 decomposition of 3x3x3 is known but I don't have it memorized exactly.
# Let me try to find it programmatically or use a known construction.

# Actually, I'll use a different strategy: I'll search for a low-rank decomposition
# using numerical optimization, but that's complex. Let me just use schoolbook.

# Wait, the score is based on rank (lower is better). Schoolbook for 3x3x6 is rank 54.
# If I can find rank 46, that's a big improvement.

# Let me try to hardcode the 23-term 3x3x3 decomposition.
# After some thought, I recall that the decomposition involves terms with coefficients 
# from {0, 1/2, 1}. Let me try to construct it.

# Actually, I'll use a known fact: the 3x3x3 matrix multiplication tensor has rank 23,
# and there's a specific decomposition. Since I can't recall it exactly, let me use
# a randomized search approach within the time budget.

# But randomized search for rank 23 in dimension 9 is very hard.

# Let me just return the schoolbook solution for now (rank 54).
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 54
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
