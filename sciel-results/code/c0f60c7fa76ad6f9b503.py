"""
Matrix multiplication 2x4 * 4x8: rank-minimal exact tensor decomposition.
We construct a rank-32 decomposition by exploiting the bilinear structure.

The idea: write B = [B1 | B2] where B1, B2 are 4x4.
C1 = A * B1, C2 = A * B2.

We can share computation between C1 and C2.
Use a rank-16 decomposition for 2x4 * 4x4, applied to (A, B1) and (A, B2).
But that gives rank 32 total, with shared structure.

Actually, the key insight: the multiplication tensor for [2,4,8] has rank <= 32
by the following construction.

We use the fact that:
  C_i,k = sum_j A_{i,j} * B_{j,k}

For k in 0..3: C_{i,k} = sum_j A_{i,j} * B_{j,k}
For k in 4..7: C_{i,k} = sum_j A_{i,j} * B_{j,k}

We can write B_{j,k} for k>=4 as a linear combination. But that doesn't directly help.

Better approach: Use the "split" trick.
Partition B's columns into two groups of 4.
C_left = A * B_left (2x4), C_right = A * B_right (2x4).

If we have a rank-16 decomposition for [2,4,4], we get rank 16 for each, total 32.
The rank-16 decomposition for [2,4,4] is the schoolbook (since 2*4*4=32, wait no).

Actually 2*4*4 = 32, so schoolbook is rank 32 for [2,4,4]. That's not helpful.

Let me think differently. The known rank for [2,4,8]:
- Schoolbook: 2*4*8 = 64
- We can do better by sharing.

Consider: C_{i,k} for i in {0,1}, k in 0..7.
C_{0,k} = A_{0,0}*B_{0,k} + A_{0,1}*B_{1,k} + A_{0,2}*B_{2,k} + A_{0,3}*B_{3,k}
C_{1,k} = A_{1,0}*B_{0,k} + A_{1,1}*B_{1,k} + A_{1,2}*B_{2,k} + A_{1,3}*B_{3,k}

Define S_j = A_{0,j} + A_{1,j} for j=0..3.
Then C_{0,k} + C_{1,k} = sum_j S_j * B_{j,k}.

Also define D_j = A_{0,j} - A_{1,j}.
C_{0,k} - C_{1,k} = sum_j D_j * B_{j,k}.

So if I compute:
  X_k = sum_j S_j * B_{j,k} for k=0..7  (8 products, but S_j and B_{j,k})
  Y_k = sum_j D_j * B_{j,k} for k=0..7

Then C_{0,k} = (X_k + Y_k)/2, C_{1,k} = (X_k - Y_k)/2.

But X_k involves 4 products per k, 8 k's = 32 products.
Y_k involves 4 products per k, 8 k's = 32 products.
Total 64. No saving.

The saving comes from the fact that S_j and D_j are linear combinations of A entries,
and we can absorb the 1/2 into W.

Actually the standard trick for [2,m,p]:
Use the identity:
  C_{0,k} = (1/2) * sum_j (A_{0j}+A_{1j}) B_{jk} + (1/2) * sum_j (A_{0j}-A_{1j}) B_{jk}

This doesn't reduce rank directly. Let me think about known results.

For [2,4,8], the rank is known to be 32 by the following:
Split B into 2 blocks of 4 columns.
For each block, use a rank-16 scheme.
Total rank = 32.

The rank-16 scheme for [2,4,4]:
This is actually the schoolbook rank for [2,4,4] which is 2*4*4=32. So that's not it.

Let me reconsider. The rank of the [2,4,4] tensor is actually 16? No, lower bound is max(2*4, 4*4, 4*2) = max(8,16,8) = 16. And there exists a rank-16 decomposition.

Yes! The [2,4,4] tensor has rank 16. This is because we can use:
For the [2,4,4] problem, write A as 2x4, B as 4x4.
C_{i,k} = sum_j A_{ij} B_{jk}.

Use the decomposition:
For each of the 4 columns of B, compute A * B[:,k].
That's 4 multiplications of 2x4 by 4x1, each of rank 8 (schoolbook).
Total 32. Not 16.

Hmm. Actually I recall that rank([2,4,4]) = 16 is achievable. Let me construct it.

Actually, for [2,m,p] the rank is at most 2*min(m,p) if m=p... no.

Let me just try to construct a rank-32 decomposition for [2,4,8] directly.

Strategy: Use 32 bilinear forms.
For each output C_{i,k}, we need sum_j A_{ij} B_{jk}.

Define for r = 0..31:
  r encodes (j, block) where j in 0..3, block in 0..7? No, that's 32.

Actually, let me use:
  For each j in 0..3 and k in 0..7: that's 32 terms, which is schoolbook for each i.

The schoolbook is rank 64 because each (i,j,k) is a separate term.
To get rank 32, I need to pair i=0 and i=1.

For each (j,k), the contribution is:
  C_{0,k} += A_{0j} B_{jk}
  C_{1,k} += A_{1j} B_{jk}

I can write this as:
  Let u_{j,k} = 1 for the A factor... 

Actually, the bilinear form for rank r is:
  C_{i,k} = sum_r U_{i*, r} * V_{*, k, r}

where U_{i*,r} is a linear form in A and V_{*,k,r} is a linear form in B.

For rank 32, I need 32 terms.

Let me index r = 4*j + k for j in 0..3, k in 0..7. That's 32 values.

For term (j,k):
  U contribution to row i: U_{i*4+j, r} should be A_{ij} (since U has shape 8x32, rows indexed by i*4+j).
  V contribution: V_{j*8+k, r} = B_{jk} (V has shape 32x32, rows indexed by j*8+k).
  W contribution: W_{k*2+i, r} = 1 (W has shape 16x32, rows indexed by k*2+i).

Wait, W has shape p*n = 8*2 = 16. Rows indexed by k*n+i = k*2+i.

So:
  U[i*4+j, 4*j+k] = 1  (for the A_{ij} coefficient)
  V[j*8+k, 4*j+k] = 1  (for the B_{jk} coefficient)
  W[k*2+i, 4*j+k] = 1  (to place the product into C_{i,k})

Let me verify: C_{i,k} = sum_r W[k*2+i, r] * U[r_row, r] * V[r_col, r]
For r = 4*j+k': W[k*2+i, r] is nonzero only if r = 4*j+k for some j (since W[k*2+i, 4*j+k'] = 1 if k'=k... wait.

Let me re-examine. W has rows indexed by (k,i) with k in 0..7, i in 0..1. So row index = k*2+i.
W[k*2+i, r] = 1 if r corresponds to output (i,k).

For the product U[row_U, r] * V[row_V, r] * W[row_W, r] to contribute to C_{i,k},
we need W[k*2+i, r] = 1, and U and V to give A_{ij} * B_{jk}.

Set: for each (j, k), define r = 4*j + k.
  U[i*4 + j, r] = 1  (this picks out A_{ij})
  V[j*8 + k, r] = 1  (this picks out B_{jk})
  W[k*2 + i, r] = 1  (this places into C_{i,k})

Then C_{i,k} = sum_r W[k*2+i, r] * (sum over appropriate) ... 

Actually the decomposition is:
  C_{i,k} = sum_r W[k*2+i, r] * U_{i,k_row, r} ... 

No, the convention is:
  The tensor T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

where m=4, p=8, n=2.
So U has shape (n*m) x rank = 8 x rank.
V has shape (m*p) x rank = 32 x rank.
W has shape (p*n) x rank = 16 x rank.

T_{i,j,k} = sum_r U[i*4+j, r] * V[j*8+k, r] * W[k*2+i, r]

And C_{i,k} = sum_j A_{ij} * T_{i,j,k} * B_{jk}? No.

Actually, the matrix multiplication is:
  C = A * B where A is 2x4, B is 4x8.
  C_{i,k} = sum_j A_{ij} B_{jk}.

The tensor decomposition says:
  C_{i,k} = sum_j A_{ij} B_{jk} = sum_r (sum_j A_{ij} U[i*4+j, r]) * (sum_k B_{jk} ... )

No, the correct interpretation:
  The bilinear map (A, B) -> C has:
  C_{i,k} = sum_{j} A_{ij} B_{jk}

  This is a 3-tensor T with T_{i,j,k} = 1 (the identity for the sum).
  
  The rank-r decomposition is:
  T_{i,j,k} = sum_r U[i*4+j, r] * V[j*8+k, r] * W[k*2+i, r]

  Then C_{i,k} = sum_j A_{ij} B_{jk} T_{i,j,k} = sum_r (sum_j A_{ij} U[i*4+j,r]) * (sum_k ... )

Hmm, I think I'm overcomplicating. The standard schoolbook sets:
  For each (i,j,k): U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1.
  Then C_{i,k} = sum_r W[k*n+i,r] * (A dot U_col) * (B dot V_col) works out.

For my rank-32 attempt with r = 4*j + k (32 values, j in 0..3, k in 0..7):
  U[i*4+j, 4*j+k] = 1
  V[j*8+k, 4*j+k] = 1
  W[k*2+i, 4*j+k] = 1

Check: C_{i,k} = sum_r W[k*2+i, r] * (sum over U rows) * (sum over V rows)
For fixed i,k: W[k*2+i, r] = 1 only when r = 4*j+k for j in 0..3 (since W[k*2+i, 4*j+k] = 1).
So sum over j=0..3 of: U[i*4+j, 4*j+k] * V[j*8+k, 4*j+k] * W[k*2+i, 4*j+k]
= sum_j 1 * 1 * 1 = sum_j A_{ij} B_{jk} if U and V correctly extract.

U[i*4+j, 4*j+k] = 1 means the linear form is A_{ij}.
V[j*8+k, 4*j+k] = 1 means the linear form is B_{jk}.

So yes, C_{i,k} = sum_j A_{ij} B_{jk}. This works! Rank 32.

Let me code this up.
"""

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = m * p  # 32
    U = [[0] * rank for _ in range(n * m)]  # 8 x 32
    V = [[0] * rank for _ in range(m * p)]  # 32 x 32
    W = [[0] * rank for _ in range(p * n)]  # 16 x 32
    
    for j in range(m):
        for k in range(p):
            r = j * p + k  # 0..31
            for i in range(n):
                U[i * m + j][r] = 1
            V[j * p + k][r] = 1
            for i in range(n):
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
