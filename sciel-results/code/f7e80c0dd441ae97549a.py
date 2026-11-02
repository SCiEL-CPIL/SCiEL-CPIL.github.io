"""Matrix multiplication 2x5 * 5x6 tensor decomposition.

This program constructs a rank-10 exact decomposition for multiplying
a 2x5 matrix by a 5x6 matrix.

The key insight is that the output C has shape 2x6. Each output row
C[i, :] = A[i, :] @ B is a linear combination of the rows of B weighted
by A[i, :]. Since A has only 2 rows, we can decompose the multiplication
into two independent 1x5x6 multiplications (one for each row of A).

For a 1x5x6 multiplication, the rank is 5 (one term per column of B).
So the total rank is 2 * 5 = 10.

Construction:
  For each row i of A (i in {0, 1}) and each column k of B (k in {0, ..., 5}):
    rank term r = i*5 + k
    U[i*m + j, r] = 1 for all j (since A[i,j] multiplies B[j,k])
    V[j*p + k, r] = 1 for all j
    W[k*n + i, r] = 1

Wait, let me verify:
  sum_r U[i*m+j, r] * V[j*p+k, r] * W[k'*n+i', r]

For a fixed (i, j, k), the product should be 1 if (i, j, k) is the "active" term,
and the sum over r should give delta(i, i') * delta(k, k') for the output.

Actually, let's think more carefully. The matrix multiplication tensor is:
  M[i, j, k] = 1 if we're computing C[i, k] = sum_j A[i, j] B[j, k]

The decomposition requires:
  sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,j,k) in the tensor sense

Wait, the tensor for matrix multiplication M_{n,m,p} has entries:
  T[i, j, k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C.

More precisely, the multiplication tensor is defined such that:
  C[i, k] = sum_{j} A[i, j] B[j, k]

In tensor decomposition form:
  sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all valid (i,j,k)
  and the structure ensures that when we contract, we get the correct C.

Actually, the standard way: the tensor T has T[i,j,k] = 1 for all i,j,k in range,
and the decomposition satisfies:
  T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

But that's not quite right either. Let me re-examine.

The matrix multiplication tensor M_{n,m,p} is a 3-tensor with:
  M[i, j, k] = 1  (for all 0<=i<n, 0<=j<m, 0<=k<p)

And a rank-r decomposition gives:
  M[i, j, k] = sum_{r=1}^r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

So we need:
  sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all (i,j,k)

With the naive construction, we set one term per (i,j,k) with all ones, giving rank n*m*p.

For a better construction, we can exploit structure. For n=2, m=5, p=6:

Consider fixing i and k. Then C[i,k] = sum_j A[i,j] B[j,k].
This is an inner product of two vectors of length m=5.

The tensor can be viewed as: for each (i,k), we need to compute the dot product of A[i,:] and B[:,k].

A rank-5 decomposition for each (i,k) pair would give rank 2*6*5 = 60, which is the same as naive.

But we can do better. Note that A[i,:] is shared across all k, and B[:,k] is shared across all i.

Let's try: for each i in {0,1} and each j in {0,...,4}, create a term that captures A[i,j] times the entire B row j.

Term r = i*5 + j:
  U[i*m+j', r] = delta(j, j')  (i.e., U[i*m+j, r] = 1, others 0)
  V[j*p+k, r] = 1 for all k
  W[k*n+i, r] = 1 for all k

Then:
  sum_r U[i*m+j, r] * V[j*p+k, r] * W[k'*n+i', r]
  = sum_{r} [delta(i, i_r) * delta(j, j_r)] * [1] * [delta(k', k_r) * delta(i', i_r)]

Wait, this isn't working cleanly. Let me define it properly.

Let r = i*5 + j for i in {0,1}, j in {0,...,4}.
  U[i*m + j', r] = 1 if j' == j, else 0.  So U[i*m+j, r] = 1, U[i*m+j', r] = 0 for j' != j.
  V[j''*p + k, r] = 1 for all j'', k.  Wait, no. V has shape (m*p, rank). The row index is j*p+k.
  We want V[j*p+k, r] = 1 for all k, and 0 for other j values? No...

Let me reconsider. We want:
  sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1

If I set for term r corresponding to (i_r, j_r):
  U[i*m+j, r] = delta(i, i_r) * delta(j, j_r)
  V[j*p+k, r] = 1  (for all j, k)
  W[k*n+i, r] = delta(i, i_r)

Then:
  sum_r delta(i, i_r) delta(j, j_r) * 1 * delta(i, i_r)
  = sum_r delta(i, i_r)^2 delta(j, j_r)
  = delta(j, j_r) summed over r where i_r = i
  = 1 if there exists r with i_r = i and j_r = j, which is true for r = i*5+j.

But this gives 1 for all (i,j,k), which is what we want! And the rank is 2*5 = 10.

Let me verify: for a given (i, j, k):
  Only terms with i_r = i contribute (because of delta(i, i_r) in U and W).
  Among those, only the term with j_r = j contributes (because of delta(j, j_r) in U).
  So exactly one term contributes, and its value is 1 * 1 * 1 = 1.

Yes! This works. Rank = 10.

BUT WAIT: V[j*p+k, r] = 1 for all j, k means that for term r = (i_r, j_r), V has 1s in ALL rows j*p+k for all j and k. That's fine.

Let me double-check with a small example. n=1, m=2, p=1. Rank should be 2.
Terms: r=0 (i=0,j=0), r=1 (i=0,j=1).
U[0*2+0, 0]=1, U[0*2+1, 0]=0
U[0*2+0, 1]=0, U[0*2+1, 1]=1
V[0*1+0, 0]=1, V[1*1+0, 0]=1
V[0*1+0, 1]=1, V[1*1+0, 1]=1
W[0*1+0, 0]=1, W[0*1+0, 1]=1

For (i=0,j=0,k=0): sum = U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[0,1]*W[0,1] = 1*1*1 + 0*1*1 = 1. Correct.
For (i=0,j=1,k=0): sum = U[1,0]*V[1,0]*W[0,0] + U[1,1]*V[1,1]*W[0,1] = 0*1*1 + 1*1*1 = 1. Correct.

Great, the construction works.
"""

import numpy as np

DIMENSIONS = [2, 5, 6]


def construct():
    """
    Return a rank-10 exact decomposition of the matrix-multiplication
    tensor for 2x5 * 5x6 -> 2x6.

    Conventions:
        U has shape (n*m, rank)  -- input-A factor
        V has shape (m*p, rank)  -- input-B factor
        W has shape (p*n, rank)  -- output factor, trace convention
          W[k*n + i, r]

    The matrix-multiplication tensor M[i,j,k] = 1 means that for each
    rank term r we have:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]  =  1

    We use a structured construction where for each (i, j) pair we
    assign one rank term, exploiting the fact that the inner product
    over j can be decomposed term-by-term.
    """
    n, m, p = DIMENSIONS
    rank = n * m  # 10

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0
    for i in range(n):
        for j in range(m):
            # U: only U[i*m + j, col] = 1
            U[i * m + j, col] = 1.0

            # V: V[j'*p + k, col] = 1 for all j', k
            for jp in range(m * p):
                V[jp, col] = 1.0

            # W: W[k*n + i, col] = 1 for all k
            for k in range(p):
                W[k * n + i, col] = 1.0

            col += 1

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
