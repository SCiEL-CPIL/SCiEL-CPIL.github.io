"""
Rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.

Strategy:
  1. Return the known rank-23 decomposition for 3x3*3x3 (Coppersmith-style) as the baseline.
  2. Attempt a search for a lower rank (20, 21, 22) using random linear combinations and
     rank-reduction of the multiplication tensor.
  3. Deadline guard to avoid timeout.

The output factor W uses the official trace convention: W[k*n + i, r].
"""

import time
import random
import itertools

# Use numpy if available for linear algebra
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

DIMENSIONS = [3, 3, 4]

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # 150s kill, leave 5s safety

def _time_left():
    return max(0.0, _DEADLINE - time.time())


# ---------------------------------------------------------------------------
# Known rank-23 decomposition for 3x3 * 3x3 (Coppersmith 1983)
# Extended to 3x3 * 3x4 by padding the 4th column.
#
# The 23 terms are given by 23x23 matrices of coefficients.
# For 3x3*3x3, the decomposition is:
#   C = A * B where A, B are 3x3
#   C_{i,k} = sum_r U_{i,r} * V_{k,r}  (in the "two-sided" form)
#
# But for our task, the tensor is T[i,j,k] with the convention:
#   U is (n*m, rank) = (9, rank)
#   V is (m*p, rank) = (3*4, rank) = (12, rank)
#   W is (p*n, rank) = (4*3, rank) = (12, rank)
#   T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
#
# For the 3x3*3x3 core, we can use the known rank-23 decomposition.
# For the 4th column, we need additional terms or we can use a different approach.
#
# Actually, the standard way: the tensor for (n,m,p) matrix multiplication has
# dimension n*m*p = 3*3*4 = 36 scalar entries. The schoolbook rank is 36.
#
# Known best upper bounds:
#   3x3*3: rank 23 (Coppersmith)
#   3x3*4: we can use the 23-term decomposition for the 3x3 part plus
#           additional terms for the 4th column.
#
# A clean approach: use the rank-23 for 3x3*3, and for the 4th output column,
# we need 3 more terms (one per row of the result), giving rank 26.
# But we might do better by sharing terms.
#
# Let's use a known construction:
#   For 3x3*3x4, the tensor can be seen as 4 copies of the 3x3 tensor
#   (one for each column of the 3x4 matrix), but the first factor A is shared.
#
# The multiplication tensor for (n,m,p) is:
#   T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
#   where e_{i,j} is the basis vector in A (position (i,j)),
#         e_{j,k} is the basis vector in B (position (j,k)),
#         e_{k,i} is the basis vector in C (position (k,i)).
#
# For 3x3*3x4, j ranges 0..2, k ranges 0..3, i ranges 0..2.
#
# We can decompose this as:
#   T = sum_{k=0}^{2} T_k  +  T_3
#   where T_k is the 3x3*3x1 part for column k, and T_3 is the 3x3*1x1 part.
#
# Each T_k for k=0,1,2 is a rank-23 tensor (3x3*3x1), but they share the A factors.
# So we can't just add 23+23+23+3.
#
# A better approach: use the known rank-23 decomposition for 3x3*3x3,
# and add 3 more terms for the 4th column, giving rank 26.
#
# But we can do better. Let's try to find a rank-25 or lower decomposition.
#
# For now, let's implement the rank-26 construction:
#   - 23 terms for the 3x3*3x3 part (columns 0,1,2)
#   - 3 additional terms for the 3x3*1x1 part (column 3)
#
# The 3 additional terms are just the schoolbook terms for column 3.
#
# Actually, we can be smarter. The rank-23 decomposition for 3x3*3x3
# gives us a way to compute C = A*B for 3x3 B.
# For 3x4 B, we can split B into [B_3 | b] where B_3 is 3x3 and b is 3x1.
# Then C = A*B_3 + A*b.
# A*B_3 uses 23 terms.
# A*b uses 3 terms (schoolbook for 3x3 * 3x1).
# Total: 26 terms.
#
# But can we share some terms between A*B_3 and A*b?
# The A*b part has the form: C_i = sum_j A_{i,j} * b_j.
# The A*B_3 part has the form: C_{i,k} = sum_j A_{i,j} * B_{j,k} for k=0,1,2.
#
# If we can find a rank-22 or 23 decomposition for the 3x3*3x4 tensor
# that shares terms between the 3x3 part and the 1-column part, we'd do better.
#
# Known result: rank(3,3,4) ≤ 25 (from various sources).
# Let's try to find rank 25.
#
# For now, I'll implement rank 26 as a safe baseline, and try to search
# for lower ranks.
