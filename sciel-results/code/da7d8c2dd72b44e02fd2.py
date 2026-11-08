<<<<<<< SEARCH
"""
Rank-minimal exact tensor decomposition for 6x6 matrix multiplication.

Structural change: replace the Strassen-2x2 block construction (rank 189) with a
direct rank-108 construction derived from the block-naive insight.
The 6x6 multiplication tensor T[i,j,k] (C[i,k] = sum_j A[i,j] B[j,k]) is a
rank-216 tensor under the naive decomposition.  However, by viewing the 6x6
product as a 3x3 grid of 2x2 block products, each 2x2*2x2 block product can be
handled with 4 rank-1 terms (one per shared block-internal index), giving
3*3*3*4 = 108 total rank.

Key insight from inspiration: the "naive" 2x2*2x2 decomposition has rank 4,
which is *better* than Strassen's rank 7 for 2x2 blocks.  When tiled over a
3x3 grid of blocks with 3 summation indices, the total rank is 27*4 = 108.

We additionally attempt a further structural reduction: since the 6x6 tensor
is a sub-tensor of a 2x2 block structure with 3x3 blocks, we can exploit
the fact that many of the 108 rank-1 terms are linearly dependent when viewed
as vectors in the appropriate tensor product space.  We attempt to find a
lower-rank decomposition by solving for a rank-R (R < 108) set of factors
that spans the same tensor, using a randomized linear-algebra approach.

However, the safe and verified construction is rank 108, which is a significant
improvement over the seed's rank 216 (naive) and even over Strassen's 189.
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]
_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Evaluator kills at 150s; keep 10s margin


def _remaining_time():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _build_naive_decomposition(n, m, p):
    """Naive O(n*m*p) decomposition. Rank = n*m*p = 216 for 6x6x6."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    return {"U": U, "V": V, "W": W}


def _build_block_naive_decomposition(n, m, p, block_size=2):
    """
    Block-naive decomposition for n=m=p=6 with 2x2 blocks.
    
    The 6x6 product C = A*B where A, B are 6x6.
    Split into 3x3 grid of 2x2 blocks: A[i,k], B[k,j], C[i,j] for i,j,k in 0..2.
    C[i,j] = sum_k A[i,k] * B[k,j].
    
    Each 2x2*2x2 product A[i,k]*B[k,j] has a rank-4 decomposition:
    For shared block-internal index t in {0,1}:
      Term t: U has 1s at (2i+a, 2k+t) for a in {0,1}
              V has 1s at (2k+t, 2j+c) for c in {0,1}
              W has 1s at (2i+a, 2j+c) for a,c in {0,1}
    
    Total rank = 3*3*3*2 = 54? No wait.
    
    Let me re-derive carefully.
    
    C[2i+a, 2j+c] = sum_{t=0}^{5} A[2i+a, t] B[t, 2j+c]
    
    Split t = 2k + t' where k in 0..2, t' in 0..1:
    C[2i+a, 2j+c] = sum_{k=0}^{2} sum_{t'=0}^{1} A[2i+a, 2k+t'] B[2k+t', 2j+c]
    
    For fixed (i, j, k, t'), the term is:
      A[2i+a, 2k+t'] * B[2k+t', 2j+c] for all a, c in {0,1}
    
    This is a rank-1 contribution to the tensor? No, it contributes to 4 entries
    of C (for a,c in {0,1}), but it's a single multiplicative term.
    
    In tensor decomposition form:
      T[a_idx, b_idx, c_idx] = sum_r U[a_idx, r] V[b_idx, r] W[c_idx, r]
    where a_idx = i*m+j (row of A, col of A), b_idx = j*p+k (row of B, col of B),
    c_idx = k*n+i (row of C, col of C) -- wait, the indexing is:
    
    Actually the tensor is: for the matrix multiplication C = AB,
    C[i, k] = sum_j A[i, j] B[j, k]
    
    The tensor T has entries T[i, j, k] = A[i,j] * B[j,k] -> C[i,k]
    But in the decomposition form:
      C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, that's not right either. Let me re-read the contract:
    "The output factor follows the official trace convention W[k*n+i,r]."
    
    So: C[i,k] = sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
    
    No wait. Let me think about this more carefully.
    
    The standard tensor decomposition of matrix multiplication:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    This can be written as:
    C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For this to work, for each r, the product sum_j U[i*m+j, r] * V[j*p+k, r]
    must give something that, when multiplied by W[k*n+i, r] and summed over r,
    gives C[i,k].
    
    In the naive case, r = (i,j,k) and:
    U[i*m+j, (i',j',k')] = 1 if (i,j) == (i',j') else 0
    V[j*p+k, (i',j',k')] = 1 if (j,k) == (j',k') else 0
    W[k*n+i, (i',j',k')] = 1 if (k,i) == (k',i') else 0
    
    Then sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    = sum_{r: i'=i, j'=j, k'=k} sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    = U[i*m+j, (i,j,k)] * V[j*p+k, (i,j,k)] * W[k*n+i, (i,j,k)]
    = 1 * 1 * 1 = 1 (for the matching j)
    
    Hmm, that gives 1 for each valid (i,j,k), but we need sum_j A[i,j]B[j,k].
    
    I think the correct interpretation is:
    C[i,k] = sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
    
    And in the naive case, for r corresponding to a specific (j) value:
    Actually, let me just use the verification function to check correctness.
    
    The verification does:
    C_computed[i,k] = sum_r Ur[i,:] @ Vr[:,k] * Wr[k*n+i]
    where Ur = U[:,r].reshape(n,m), Vr = V[:,r].reshape(m,p), Wr = W[:,r].reshape(p,n)
    
    So: C_computed[i,k] = sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
    
    For this to equal sum_j A[i,j] B[j,k], we need:
    sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = sum_j A[i,j] B[j,k]
    
    In the naive decomposition with rank = n*m*p:
    For each (i0, j0, k0), define factor r = (i0, j0, k0):
    U[i*m+j, r] = 1 if i==i0 and j==j0, else 0
    V[j*p+k, r] = 1 if j==j0 and k==k0, else 0
    W[k*n+i, r] = 1 if k==k0 and i==i0, else 0
    
    Then sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
    = sum_{r: matches} (sum_j delta(i,i0)delta(j,j0) * delta(j,j0)delta(k,k0)) * delta(k,k0)delta(i,i0)
    = For r where i0=i, j0=j, k0=k: (1 * 1) * 1 = 1
    But this only happens when j=j0, so sum_j gives 1 for the matching j.
    
    Wait, I'm confusing myself. Let me just trust the seed program's naive decomposition
    and verify it works, then build the block version.
    
    Actually, looking at the seed's naive:
    U[i * m + j][idx] = 1  where idx corresponds to (i,j,k)
    V[j * p + k][idx] = 1
    W[k * n + i][idx] = 1
    
    And the verification:
    Ur = U[:, r].reshape(n, m)  -- so Ur[i, j] = U[i*m+j, r]
    Vr = V[:, r].reshape(m, p)  -- so Vr[j, k] = V[j*p+k, r]
    Wr = W[:, r].reshape(p, n)  -- so Wr[k, i] = W[k*n+i, r]
    
    prod = Ur @ Vr  -- prod[i, k] = sum_j Ur[i,j] * Vr[j,k]
    C_computed += prod * Wr  -- C_computed[i,k] += prod[i,k] * Wr[k,i]
    
    So C_computed[i,k] = sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
    
    For the naive decomposition, for a given r = (i0, j0, k0):
    U[i*m+j, r] is 1 only when i=i0 and j=j0
    V[j*p+k, r] is 1 only when j=j0 and k=k0
    W[k*n+i, r] is 1 only when k=k0 and i=i0
    
    So for fixed (i, k), sum_r (sum_j ...) * W[k*n+i, r]:
    Only r where i0=i and k0=k contribute to W.
    For such r, sum_j U[i*m+j, r] * V[j*p+k, r] = sum_j delta(j,j0) * delta(j,j0) = 1 (for j=j0)
    
    So C_computed[i,k] = sum_{j0} 1 * 1 = m = 6?
    
    That can't be right. The naive decomposition must be wrong in my understanding.
    
    Let me re-examine. Oh wait, in the naive decomposition, for each (i,j,k) triple,
    there's exactly one factor r. So for fixed (i,k), the factors that contribute
    are those with i0=i and k0=k, and j0 varies over 0..m-1.
    
    For each such factor r=(i, j0, k):
    sum_j U[i*m+j, r] * V[j*p+k, r] = sum_j delta(j, j0) * delta(j, j0) = 1
    W[k*n+i, r] = 1
    
    So C_computed[i,k] = sum_{j0=0}^{m-1} 1 * 1 = m = 6.
    
    But C_expected[i,k] = sum_j A[i,j] * B[j,k], which is not 6.
    
    So the naive decomposition as written is INCORRECT?
    
    Wait, I must be misunderstanding the tensor structure. Let me re-read.
    
    Oh! I think the issue is that U, V, W are not deltas but should encode the
    actual values. No wait, in a tensor decomposition, U, V, W are fixed factors,
    and the "input" matrices A and B are not part of the decomposition. The
    decomposition is of the multiplication *operation* itself.
    
    Hmm, but that doesn't make sense either. The multiplication tensor T is a
    fixed 3-tensor, and we're decomposing it. The matrices A and B are just
    a way to test the decomposition.
    
    Let me reconsider. The matrix multiplication tensor T is defined by:
    T[i, j, k] = 1 if the entry (i,k) of C=AB receives contribution from A[i,j]*B[j,k]
    
    More precisely, T is a 3-tensor where:
    T[a, b, c] for a in 0..n*m-1, b in 0..m*p-1, c in 0..p*n-1
    
    But the indices are constrained: a = i*m+j, b = j*p+k, c = k*n+i
    for i in 0..n-1, j in 0..m-1, k in 0..p-1.
    
    So T is not a full n*m x m*p x p*n tensor; it's a sparse tensor with
    n*m*p non-zero entries, each equal to 1.
    
    The decomposition T = sum_r U_r ⊗ V_r ⊗ W_r means:
    T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    For the naive decomposition, T[a, b, c] = 1 if (a,b,c) corresponds to a valid
    (i,j,k) triple, else 0.
    
    With the naive factors:
    U[a, r] = 1 if a = i*m+j and r = (i,j,k) for some k, else 0
    V[b, r] = 1 if b = j*p+k and r = (i,j,k) for some i, else 0
    W[c, r] = 1 if c = k*n+i and r = (i,j,k) for some j, else 0
    
    Then T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
    = 1 if there exists r=(i,j,k) such that a=i*m+j, b=j*p+k, c=k*n+i
    = 1 if (a,b,c) is a valid triple, else 0.
    
    Yes! That's correct. The tensor T itself is being decomposed, not the
    multiplication of specific matrices A and B.
    
    Now, the verification function tests this by:
    C_computed[i,k] = sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
    
    And this should equal sum_j T[i*m+j, j*p+k, k*n+i] ... no, that's not right.
    
    Actually, the verification is checking that the decomposition correctly
    reproduces matrix multiplication. Let me think about what it's really checking.
    
    If we define:
    M_r[i, k] = (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
    
    Then C_computed = sum_r M_r.
    
    For this to equal A*B for all A, B, we need:
    sum_r M_r[i, k] = sum_j A[i,j] B[j,k] for all A, B.
    
    This means that the tensor sum_r U_r ⊗ V_r ⊗ W_r (appropriately indexed)
    must equal the multiplication tensor.
    
    OK so I was overcomplicating this. The seed's naive decomposition is correct,
    and the verification confirms it. Let me just build the block version and verify.
    
    For the block-naive decomposition with 2x2 blocks:
    
    C[2i+a, 2j+c] = sum_{k=0}^{2} sum_{t=0}^{1} A[2i+a, 2k+t] B[2k+t, 2j+c]
    
    For each (i, j, k, t), we have a term:
    A[2i+a, 2k+t] * B[2k+t, 2j+c] for a, c in {0,1}
    
    This contributes to C[2i+a, 2j+c] for a, c in {0,1}.
    
    In tensor decomposition form, for factor r = (i, j, k, t):
    U[2i+a, 2k+t] = 1 for a in {0,1}  (i.e., U[(2i+a)*6 + (2k+t), r] = 1)
    V[2k+t, 2j+c] = 1 for c in {0,1}  (i.e., V[(2k+t)*6 + (2j+c), r] = 1)
    W[2j+c, 2i+a] = 1 for a, c in {0,1}  (i.e., W[(2j+c)*6 + (2i+a), r] = 1)
    
    Wait, W is indexed as W[k*n+i, r] where k is the row of C and i is the col of C.
    So W[c_row * n + c_col, r].
    
    C[2i+a, 2j+c] has c_row = 2i+a, c_col = 2j+c.
    So W[(2i+a)*6 + (2j+c), r] = 1.
    
    Let me verify: for factor r = (i, j, k, t):
    sum_j' U[(2i+a)*6 + j', r] * V[j'*6 + (2j+c), r]
    = sum_j' delta(j' == 2k+t) * delta(j' == 2k+t)
    = 1 (for j' = 2k+t)
    
    Then multiplied by W[(2i+a)*6 + (2j+c), r] = 1.
    
    So M_r[2i+a, 2j+c] = 1.
    
    And C_computed[2i+a, 2j+c] = sum_{r: i_r=i, j_r=j, k_r=k, t_r=t} 1
    = sum_{k=0}^{2} sum_{t=0}^{1} 1 = 6.
    
    Again, this gives 6, not the correct value.
    
    I think the issue is that the tensor decomposition is NOT about making
    C_computed = A*B. It's about decomposing the multiplication tensor T itself.
    
    The verification function is checking something specific. Let me re-read it:
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        prod = Ur @ Vr
        C_computed += prod * Wr
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        return False
    
    So it IS checking that C_computed = A*B for random A, B.
    
    For this to work, we need:
    sum_r (Ur @ Vr) * Wr = A @ B for all A, B.
    
    This is a very strong condition. It means that the decomposition must
    correctly implement the bilinear map (A, B) -> A*B.
    
    In the naive case, for r = (i0, j0, k0):
    Ur[i, j] = 1 if i=i0, j=j0 else 0
    Vr[j, k] = 1 if j=j0, k=k0 else 0
    Wr[k, i] = 1 if k=k0, i=i0 else 0
    
    (Ur @ Vr)[i, k] = sum_j Ur[i,j] Vr[j,k] = 1 if i=i0, k=k0 else 0
    (Ur @ Vr) * Wr: element (i,k) = (Ur@Vr)[i,k] * Wr[k,i]
    = 1 if i=i0, k=k0 else 0 (since Wr[k0, i0] = 1)
    
    C_computed[i, k] = sum_r (Ur_r @ Vr_r)[i,k] * Wr_r[k,i]
    = sum_{r: i0=i, k0=k} 1
    = sum_{j0} 1 = m = 6.
    
    This is still 6, not A[i,:] @ B[:,k].
    
    I'm clearly misunderstanding the tensor structure. Let me look at this from a
    different angle.
    
    The matrix multiplication tensor T is a 3-tensor in (A-space) ⊗ (B-space) ⊗ (C-space).
    T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the indices are (i,j) for A, (j,k) for B,
    (i,k) for C.
    
    A rank-R decomposition is T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r where
    u_r in A-space (dim n*m), v_r in B-space (dim m*p), w_r in C-space (dim p*n).
    
    The condition that this decomposition "implements" matrix multiplication is:
    For any A (n×m) and B (m×p), C = A*B is given by:
    C[i,k] = sum_r (u_r^T vec(A)) * (v_r^T vec(B)) * w_r[i,k]
    
    Wait, that's not right either. Let me think about this more carefully.
    
    Actually, I think the correct interpretation is:
    C = sum_r (u_r ⊗ v_r) * w_r where the product is defined appropriately.
    
    Or perhaps: C[i,k] = sum_r w_r[i,k] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    No, that doesn't make sense dimensionally.
    
    Let me just trust the seed program and its verification. The seed's naive
    decomposition passes verification (score 1.0), so it must be correct.
    
    Let me re-examine the naive decomposition:
    U[i * m + j][idx] = 1  where idx = i*m*p + j*p + k (or some ordering of (i,j,k))
    V[j * p + k][idx] = 1
    W[k * n + i][idx] = 1
    
    In the verification:
    Ur = U[:, r].reshape(n, m)  -- Ur[i, j] = U[i*m+j, r]
    Vr = V[:, r].reshape(m, p)  -- Vr[j, k] = V[j*p+k, r]
    Wr = W[:, r].reshape(p, n)  -- Wr[k, i] = W[k*n+i, r]
    
    prod = Ur @ Vr  -- prod[i, k] = sum_j Ur[i,j] * Vr[j,k]
    C_computed[i,k] += prod[i,k] * Wr[k,i]
    
    For the naive factor r=(i0,j0,k0):
    Ur[i,j] = 1 if (i,j) = (i0,j0) else 0
    Vr[j,k] = 1 if (j,k) = (j0,k0) else 0
    Wr[k,i] = 1 if (k,i) = (k0,i0) else 0
    
    prod[i,k] = sum_j Ur[i,j] Vr[j,k] = Ur[i0,j0] Vr[j0,k0] = 1 if (i,k) = (i0,k0) else 0
    Wr[k0,i0] = 1
    
    So C_computed[i,k] += 1 if (i,k) = (i0,k0) else 0.
    
    Summing over all r = (i0,j0,k0):
    C_computed[i,k] = sum_{j0} 1 = m = 6.
    
    This is still 6! But the seed program scores 1.0, meaning it passes verification.
    
    I must be making an error. Let me actually run through a small example.
    
    Take n=m=p=1. Then A=[a], B=[b], C=[ab].
    Naive: rank = 1.
    U[0,0] = 1, V[0,0] = 1, W[0,0] = 1.
    Ur = [[1]], Vr = [[1]], Wr = [[1]].
    prod = [[1]].
    C_computed[0,0] = 1 * 1 = 1.
    C_expected[0,0] = a*b.
    1 != a*b unless a*b = 1.
    
    So the naive decomposition is WRONG for n=m=p=1!
    
    But the seed program scores 1.0... Let me re-read the verification function.
    
    Oh wait, I see. The verification uses random integer matrices A and B with
    entries in {-3, -2, -1, 0, 1, 2, 3}. If A and B are random, then C_expected
    is random, and C_computed = 6 (for 6x6) or 1 (for 1x1) won't match.
    
    Unless... the verification is actually passing because of some other reason.
    
    Let me re-read the seed program's verification more carefully.
    
    Actually, I think I've been misunderstanding the tensor structure. Let me re-read
    the task description:
    
    "Find a rank-minimal exact tensor decomposition for multiplying a 6 by 6 matrix
    by a 6 by 6 matrix."
    
    "Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes
    (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries that
    exactly decompose the matrix multiplication tensor"
    
    So U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    
    The matrix multiplication tensor T is in R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}.
    T[a, b, c] where a indexes (i,j) pairs, b indexes (j,k) pairs, c indexes (k,i) pairs.
    
    But not all (a,b,c) are independent; they must be consistent (the j in a must
    match the j in b, the k in b must match the k in c, the i in c must match the i in a).
    
    The tensor T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some (i,j,k), else 0.
    
    A rank-R decomposition: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
    
    The "exactly decompose the matrix multiplication tensor" means this equality
    must hold for all (a,b,c).
    
    The verification function checks this indirectly by testing that the
    decomposition correctly computes C = A*B for random A, B.
    
    But as I showed, the naive decomposition gives C_computed = 6 (for 6x6),
    not A*B.
    
    I'm confused. Let me just look at the actual seed program and see if it's
    doing something different.
    
    Oh! I think I see the issue. The tensor T is not just 0/1. The matrix
    multiplication tensor has entries that depend on the structure of the
    multiplication.
    
    Actually, no. The matrix multiplication tensor is a fixed tensor that
    encodes the bilinear map (A, B) -> A*B. Its entries are 0 or 1, and
    the decomposition must reproduce this tensor exactly.
    
    Let me try a different approach. Instead of trying to understand the
    verification, let me just build a decomposition and test it.
    
    Actually, you know what, let me just use the seed program's approach and
    verify it works, then modify it.
    
    The key insight from the inspiration is that the block-naive approach
    should give rank 108 for 6x6. Let me just build it and verify.
    
    Here's my plan:
    1. Build the block-naive decomposition with 2x2 blocks (rank 108).
    2. Verify it using the same verification function as the seed.
    3. If it passes, return it.
    4. If not, fall back to the seed's naive decomposition (rank 216).
    
    Let me code this up.
"""
=======
"""
Rank-minimal exact tensor decomposition for 6x6 matrix multiplication.

Structural change: Use Strassen 2x2 block construction (rank 189) as a verified
baseline, then apply randomized linear algebra to find a lower-rank equivalent
decomposition by identifying linear dependencies among the 189 factor vectors.

The 6x6 product is viewed as a 3x3 grid of 2x2 blocks. Each 2x2*2x2 block
product uses Strassen's 7-term algorithm. Total: 27 * 7 = 189 factors.

Then we attempt to reduce rank by:
1. Building the full 189-factor decomposition
2. Constructing the tensor as a vector in R^{36*36*36}
3. Using randomized SVD / QR to find the intrinsic rank of the set of factor
   triples
4. If rank < 189, reconstruct a minimal decomposition
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]
_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Evaluator kills at 150s; keep 10s margin


def _remaining_time():
    return _TIME_LIMIT - (time.time() - _START_TIME)
>>>>>>> REPLACE
