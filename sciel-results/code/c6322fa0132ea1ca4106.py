"""
Ladder/shifted tensor decomposition for 2x5x6 matrix multiplication.

The idea is to exploit the small dimension n=2 to construct a decomposition
with rank less than the schoolbook rank of 60.

For n=2, m=5, p=6, we can try to use the structure of the tensor to find
a lower rank decomposition. The tensor for matrix multiplication A(2x5) * B(5x6) -> C(2x6)
has the form:

T[i, j, k] = 1 for all valid indices, where C[i,k] = sum_j A[i,j] * B[j,k]

This means T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k, but with the constraint that
the index j is shared between A and B.

Actually, the tensor is:
T[i, j, k] = 1 for i in {0,1}, j in {0,...,4}, k in {0,...,5}

So T = sum_{j=0}^{4} sum_{k=0}^{5} (e_0 ⊗ e_j ⊗ e_k + e_1 ⊗ e_j ⊗ e_k)
    = sum_{j,k} (e_0 + e_1) ⊗ e_j ⊗ e_k

Wait, that's not right. Let me be more careful.

The matrix multiplication tensor M_{n,m,p} has entries:
M[i, j, k] = 1 if we multiply row i of A with column k of B through entry (i,j) and (j,k)

So for each i, j, k, the term is e_i^A ⊗ e_j^B ⊗ e_k^C.

The full tensor is:
T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k

This is the schoolbook decomposition with rank n*m*p.

For n=2, m=5, p=6, the schoolbook rank is 60.

Now, the "ladder" or "shifted" approach: since n=2 is small, we can try to
combine terms. Specifically, we can write:

T = sum_{j=0}^{4} sum_{k=0}^{5} (e_0 ⊗ e_j ⊗ e_k + e_1 ⊗ e_j ⊗ e_k)

For each pair (j,k), we have two terms. If we could combine these two into
one term with some clever choice of vectors, we would save rank.

But e_0 ⊗ e_j ⊗ e_k + e_1 ⊗ e_j ⊗ e_k = (e_0 + e_1) ⊗ e_j ⊗ e_k

So T = sum_{j,k} (e_0 + e_1) ⊗ e_j ⊗ e_k

This is a valid decomposition with rank m*p = 30!

Wait, is this correct? Let's verify.

If we set:
U[i, l] where l indexes (j,k) pairs, U[i, l] = 1 if i is in {0,1} (always true)
Actually, let me reconsider.

We want U (n*m x rank), V (m*p x rank), W (p*n x rank).

Using the decomposition T = sum_{j,k} u_{j,k} ⊗ v_{j,k} ⊗ w_{j,k}

where:
u_{j,k}[i] = 1 for i=0,1 (i.e., u_{j,k} = e_0 + e_1 in the A-space... but wait)

Actually, let me think about this differently. The standard formulation is:

C[i,k] = sum_{l=0}^{r-1} W[k*n+i, l] * (sum_{j} U[i*m+j, l] * V[j*p+k, l])

So the tensor is decomposed as:
T = sum_l U_l ⊗ V_l ⊗ W_l

where U_l is a vector in R^{n*m}, V_l in R^{m*p}, W_l in R^{p*n}.

For the schoolbook, each term corresponds to one (i,j,k) triple.

Now, the key insight: since n=2, for each (j,k), the terms for i=0 and i=1
are:
- i=0: U has 1 at position (0,j), V has 1 at (j,k), W has 1 at (k,0)
- i=1: U has 1 at position (1,j), V has 1 at (j,k), W has 1 at (k,1)

Can we combine these? If we set:
U_l[0,j] = 1, U_l[1,j] = 1 (so U_l = e_{0j} + e_{1j} in the A-space)
V_l[j,k] = 1
W_l[k,0] = 1, W_l[k,1] = 1 (so W_l = e_{k0} + e_{k1} in the C-space)

Then U_l ⊗ V_l ⊗ W_l would give:
For i in {0,1}, j' in {0,...,4}, k' in {0,...,5}:
(U_l[i*m+j']) * (V_l[j'*p+k']) * (W_l[k'*n+i])

This is non-zero only when j'=j, k'=k, and i in {0,1}, and the value is 1*1*1 = 1.

But we need the contribution to T[i, j, k] to be 1 for each (i,j,k).

The tensor T[i,j,k] = 1 for all i,j,k.

The decomposition sum_l U_l ⊗ V_l ⊗ W_l gives:
(sum_l U_l[i*m+j] * V_l[j*p+k] * W_l[k*n+i]) should equal 1 for all i,j,k.

With my proposed single term for each (j,k):
For fixed j,k, the contribution to T[i,j,k] from this term is:
U_l[i*m+j] * V_l[j*p+k] * W_l[k*n+i] = 1 * 1 * 1 = 1 (if i in {0,1})

But we also have contributions from other (j',k') terms that might affect T[i,j,k].

Wait, I need to be more careful. The tensor T is indexed by (i, j, k) where
i is the row of A/C, j is the column of A/row of B, k is the column of B/C.

In the decomposition, the sum over l gives:
T_decomposed[i, j, k] = sum_l U_l[i*m+j] * V_l[j*p+k] * W_l[k*n+i]

For this to equal 1 for all i, j, k, we need each (i,j,k) to get contribution 1.

If I use one term per (j,k) pair with U_l having 1s at positions (0,j) and (1,j),
V_l having 1 at (j,k), and W_l having 1s at (k,0) and (k,1), then:

T_decomposed[i, j, k] = sum_{l} U_l[i*m+j] * V_l[j*p+k] * W_l[k*n+i]

For a fixed (i,j,k), the only l that can contribute is the one corresponding to (j,k).
For that l: U_l[i*m+j] = 1, V_l[j*p+k] = 1, W_l[k*n+i] = 1.
So the contribution is 1.

But wait, could other l's contribute? For another l' corresponding to (j',k'):
U_l'[i*m+j] is non-zero only if j'=j (since U_l' has 1s only at positions with column j').
V_l'[j*p+k] is non-zero only if j'=j and k'=k.
So only l corresponding to (j,k) contributes.

Therefore, T_decomposed[i,j,k] = 1 for all i,j,k. This is correct!

So the rank is m*p = 5*6 = 30.

Let me verify with the shapes:
- U has shape (n*m, rank) = (10, 30)
- V has shape (m*p, rank) = (30, 30)
- W has shape (p*n, rank) = (12, 30)

For each l in 0..29, let j = l // 6, k = l % 6.
U[i*m+j, l] = 1 for i in {0,1} (i.e., U[0*m+j, l] = 1 and U[1*m+j, l] = 1)
V[j*p+k, l] = 1
W[k*n+i, l] = 1 for i in {0,1} (i.e., W[k*n+0, l] = 1 and W[k*n+1, l] = 1)

This gives rank 30, which is much better than 60!
"""

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    rank = m * p  # 30
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for l in range(rank):
        j = l // p
        k = l % p
        
        # U: for this (j,k) term, set U[i*m + j, l] = 1 for i in {0, 1}
        for i in range(n):
            U[i * m + j][l] = 1
        
        # V: V[j*p + k, l] = 1
        V[j * p + k][l] = 1
        
        # W: for this (j,k) term, set W[k*n + i, l] = 1 for i in {0, 1}
        for i in range(n):
            W[k * n + i][l] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
