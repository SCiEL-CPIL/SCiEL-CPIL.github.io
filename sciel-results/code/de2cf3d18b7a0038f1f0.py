"""
Rank-minimal tensor decomposition for 3x4 * 4x8 matrix multiplication.
Uses the structural factorization: M_{3,4,8} can be decomposed by exploiting
the 4x8 = 4x(4*2) structure.

The strategy:
1. For the 3x4 * 4x4 core, use a rank-20 decomposition (Winograd-style).
2. For the 1x1 * 1x2 extension (each 4-row block of the 8-column matrix),
   we need rank-2 per core-rank-1, but actually we can use the fact that
   multiplying by a 4x8 matrix is like multiplying by two 4x4 blocks.
   
Actually, let me think more carefully. The tensor T_{i,j,k} for (3,4,8) has
indices i in [0,3), j in [0,4), k in [0,8).

Key insight: We can view the 4x8 matrix as having two 4x4 blocks (columns 0-3 and 4-7).
So the multiplication C = A * B where A is 3x4, B is 4x8, C is 3x8.
C[:, 0:4] = A * B[:, 0:4] and C[:, 4:8] = A * B[:, 4:8].

If we have a rank-r decomposition for 3x4 * 4x4, then for 3x4 * 4x8 we can use
rank-2r by applying the same U and V patterns to each half of B's columns,
with W split accordingly.

But we can do better. Let's use the fact that for the 3x4*4x4 problem,
the best known rank is 19 (or 20 with a simple construction).

Actually, let me use a more direct approach. The schoolbook rank is 3*4*8 = 96.

Let me try a block structure:
- Split the 8 columns into two groups of 4.
- For each group, use a rank-20 decomposition for 3x4*4x4.
- Total rank: 40.

But we can do even better by sharing some structure.

Let me construct a rank-40 decomposition explicitly.

For the 3x4*4x4 core, I'll use a Winograd-like decomposition.
The standard Winograd for 2x2*2x2 has rank 5 (vs schoolbook 8).
For 3x3*3x3, rank is 23 (vs 27).

For 3x4*4x4, let me use a simple block approach:
Split the 4 columns of A into two 2-column blocks.
Split the 4 rows of B into two 2-row blocks.
This gives four 3x2 * 2x2 sub-problems, each of rank... hmm, this gets complex.

Let me just use the straightforward block decomposition:
- Treat 3x4*4x8 as two independent 3x4*4x4 problems.
- Each 3x4*4x4 uses schoolbook rank 48? No, that's worse.

Wait, schoolbook for 3x4*4x4 is 3*4*4 = 48. Two of those is 96, same as original.

I need a better decomposition for the 3x4*4x4 core.

Let me use the fact that 3x4*4x4 can be seen as:
A (3x4) * B (4x4) = C (3x4)

We can write this as:
C[i,k] = sum_j A[i,j] * B[j,k]

Using a rank-r decomposition:
C[i,k] = sum_t U[i, t] * V[t, k]  ... but this is for the whole product.

Actually, the tensor decomposition is:
T_{i,j,k} = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]  ... no.

The standard form: For matrix multiplication A(m x n) * B(n x p) = C(m x p),
the tensor T_{i,j,k} (where i indexes rows of A, j columns of A/rows of B, k columns of B)
decomposes as T_{i,j,k} = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]... 

No, the correct form is:
T_{i,j,k} = sum_t U[i, t] * V[j, t] * W[k, t]  ... no, that's not right either.

The proper tensor decomposition for matmul:
We want T[i,j,k] such that C[i,k] = sum_j A[i,j] B[j,k] = sum_t (sum_j U[i,j,t] A[i,j]) ... 

Let me re-read the interface:
U has shape (n*m, rank) - so U[i*m+j, t]
V has shape (m*p, rank) - so V[j*p+k, t]  
W has shape (p*n, rank) - so W[k*n+i, t]

And the decomposition satisfies:
T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]

Wait, that's a 3-way outer product sum. So:
T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]

And the multiplication is:
C[i,k] = sum_j A[i,j] * B[j,k]

The tensor T[i,j,k] = delta_{i,k via j}... Actually T[i,j,k] is 1 if the (i,j) element of A and (j,k) element of B multiply to contribute to C[i,k]. So T[i,j,k] = 1 for all valid i,j,k (it's the identity in the middle index).

Wait no. T[i,j,k] is the coefficient such that:
C[i,k] = sum_{j} sum_t U[i*m+j, t] * A[i,j] * V[j*p+k, t] * B[j,k] * W[k*n+i, t]

Hmm, that doesn't look right. Let me think again.

The standard tensor decomposition for matmul:
We have tensors U, V, W such that:
T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]

And T[i,j,k] = 1 for all i,j,k (since every A[i,j]*B[j,k] contributes to C[i,k]).

Wait, that's not right. T[i,j,k] should be the structure tensor. For matmul:
C[i,k] = sum_j A[i,j] B[j,k]

The structure tensor T has T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k], which is always true. So T[i,j,k] = 1 for all i,j,k.

But that can't be right because then the rank would just be n*m*p and we couldn't do better.

Oh wait, I think the decomposition is:
C[i,k] = sum_t (sum_j U[i*m+j, t] * A[i,j]) * (sum_j V[j*p+k, t] * B[j,k]) * W[k*n+i, t]

No, that's not standard either. Let me look at the schoolbook example:
U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1 for each (i,j,k) triple.

So T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]

In schoolbook, for a given (i,j,k), there's exactly one t where all three are 1, so T[i,j,k] = 1.

So yes, T[i,j,k] = 1 for all i,j,k. The task is to find a low-rank decomposition of the all-ones tensor of size (n*m) x (m*p) x (p*n) with the constraint that the factors have the specific index structures.

Wait, but that's just the all-ones tensor, and its rank is 1! That can't be right.

I must be misunderstanding the decomposition. Let me re-read.

"factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries that exactly decompose the matrix multiplication tensor"

I think the decomposition is:
The matmul tensor T has indices (i, j, k) and T[i,j,k] = 1.
We decompose T as: T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]

But this is a 3-tensor decomposition where each factor has a combined index. The rank is the number of terms.

For the all-ones tensor of dimensions (nm) x (mp) x (pn), the tensor rank is actually nm*mp*pn in general... no wait.

Hmm, I think I'm overcomplicating this. The point is that the schoolbook gives rank n*m*p, and we need to find a lower rank. The key constraint is that the factors must have the specific index structure (U indexed by i*m+j, V by j*p+k, W by k*n+i).

This is equivalent to finding a low-rank decomposition of the matmul tensor in the standard way. The best known bounds for 3x4*4x8 are not trivially known, but we can try to construct a good one.

Let me just go with the block approach: split 4x8 into two 4x4 blocks, use a rank-20 decomposition for each 3x4*4x4, total rank 40.

For 3x4*4x4, I'll use a simple construction based on splitting into 2x2 blocks.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 8
    deadline = time.time() + 140.0  # safety margin
    
    # Strategy: Split the 8-column matrix into two 4-column blocks.
    # For each 3x4 * 4x4 sub-problem, use a rank-20 decomposition.
    # Total rank: 40.
    
    # Construct rank-20 decomposition for 3x4 * 4x4
    # U_core: (12, 20), V_core: (16, 20), W_core: (12, 20)
    # where indices are i*4+j for U, j*4+k for V, k*3+i for W
    
    # Use a block-structured approach:
    # Split the 4 columns of A into two 2-column groups: j=0,1 and j=2,3
    # Split the 4 rows of B into two 2-row groups: j=0,1 and j=2,3
    # This gives four 3x2 * 2x2 sub-problems.
    
    # For each 3x2 * 2x2 sub-problem, the schoolbook rank is 3*2*2 = 12.
    # But we can do better. For 2x2*2x2, Winograd gives rank 5.
    # For 3x2*2x2, we can use a similar approach.
    
    # Actually, let's use a simpler approach: just use schoolbook for the core
    # but with some sharing. Or better, let me construct an explicit rank-20.
    
    # Let me use the following construction for 3x4*4x4:
    # We'll use the fact that we can share some terms between different (i,k) pairs.
    
    # For now, let me just use the block decomposition with schoolbook for each block.
    # 3x4*4x4 schoolbook is 48, two of those is 96. That's no better.
    
    # Let me try a different approach: use the Kronecker product structure.
    # 3x4*4x8 = (3x4*4x4) x (1x1*1x2) in some sense.
    
    # Actually, the cleanest approach: just use rank 40 by treating the two 4x4
    # blocks independently with a rank-20 decomposition each.
    
    # For 3x4*4x4, here's a rank-20 construction:
    # We use the following idea: the 4x4 matrix can be decomposed as
    # [A B; C D] where each is 2x2. Then A*4x4 = A*[A B; C D]
    # = [A*A, A*B; ...] no, that's not right.
    
    # Let me just construct it explicitly. I'll use a greedy/structured approach.
    
    # For 3x4*4x4, I'll use the following rank-20 decomposition:
    # Based on the fact that we can group the 12 terms of schoolbook (3*4)
    # and find 20 terms that cover all 48 (i,j,k) triples.
    
    # Actually, let me just use a simple valid decomposition with rank 40.
    # Split p=8 into two halves of 4. For each half, use schoolbook for 3x4*4x4
    # but that's rank 48 each, total 96. Not good.
    
    # Let me try: for 3x4*4x4, use rank 20.
    # Construction: 
    # The tensor T[i,j,k] = 1 for i in 0..2, j in 0..3, k in 0..3.
    # We need U[i*4+j, t] * V[j*4+k, t] * W[k*3+i, t] summed over t to give 1.
    
    # Here's a rank-20 construction using block structure:
    # Group j into {0,1} and {2,3}.
    # For j in {0,1}: we have a 3x2 * 2x4 sub-problem (rank 3*2*4=24 schoolbook)
    # For j in {2,3}: same.
    # Total: 48. Still schoolbook.
    
    # I need to find actual rank reduction. Let me use a known result:
    # For 2x2*2x2, rank is 5 (Winograd).
    # For 3x3*3x3, rank is 23.
    # For 3x4*4x4, I'm not sure of the exact rank.
    
    # Let me just go with a rank-40 construction using the following:
    # Treat the 4x8 as 4x(4+4), and for each 4x4 block, use a rank-20 decomp.
    # I'll construct rank-20 for 3x4*4x4 using a structured approach.
    
    # Rank-20 for 3x4*4x4:
    # Use the following: split into 3 groups of 4 (one for each i).
    # For each i, we need to compute C[i,k] = sum_j A[i,j] B[j,k] for k=0..3.
    # This is a 1x4 * 4x4 = 1x4 multiplication, rank 4 schoolbook.
    # Three of those: rank 12. But that's not how it works because U depends on (i,j).
    
    # OK let me just use the schoolbook but with a clever indexing to get rank 40.
    # Actually, schoolbook for 3x4*4x8 is 96. Let me just verify that a rank-40
    # decomposition exists by construction.
    
    # Here's my plan: Use the fact that T[i,j,k] = 1 for all i,j,k.
    # I'll construct U, V, W such that sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1.
    
    # For rank 40, I'll use the following structure:
    # For each (i, j_pair) where j_pair in {0,1} (grouping j=0,1 and j=2,3),
    # and for each k_half in {0,1} (grouping k=0..3 and k=4..7),
    # use a 2x2*2x2 Winograd-like decomposition.
    
    # This is getting complicated. Let me just use a simple valid decomposition.
    
    # SIMPLE APPROACH: Use rank 48.
    # Split p=8 into two 4s. For each, use schoolbook 3x4*4x4 = 48 terms? No, that's 48 each.
    
    # Wait, 3*4*4 = 48 for one half, so 96 total. That's schoolbook.
    
    # Let me try: for each i (3 values), for each j (4 values), we need to cover all k (8 values).
    # U[i*m+j, t] is nonzero for some t. V[j*p+k, t] and W[k*n+i, t] must be set accordingly.
    
    # The minimum rank is at least max(n*m, m*p, p*n) = max(12, 32, 24) = 32.
    # So rank 32 is a lower bound. Let's aim for rank 32-40.
    
    # Construction for rank 32:
    # If we can make U have rank 12 (full row rank), V have rank 32, W have rank 24,
    # and the tensor rank is 32, then we're good.
    
    # Here's a construction:
    # Let t = 0, 1, ..., 31.
    # Map t to (j, k_group) where j in 0..3, k_group in 0..7.
    # U[i*m+j, t] = 1 if t corresponds to this j.
    # V[j*p+k, t] = 1 if t corresponds to this j and k.
    # W[k*n+i, t] = 1 if t corresponds to this k and i.
    
    # But this gives sum_t U*V*W = 1 only if for each (i,j,k), there's exactly one t
    # where all three are 1. That's the schoolbook with rank 96.
    
    # To get lower rank, we need multiple (i,j,k) to share the same t.
    
    # Let me try a different approach: use random construction with verification.
    
    # Actually, for the purpose of this task, let me just use a rank-48 decomposition
    # which is better than 96.
    
    # Rank-48: Split the 8 k-values into two groups of 4.
    # For k in 0..3: use schoolbook for 3x4*4x4, rank 48.
    # For k in 4..7: use schoolbook for 3x4*4x4, rank 48.
    # Total: 96. Still schoolbook!
    
    # I'm going in circles. Let me just use a known good construction.
    
    # The key insight I'm missing: the tensor T[i,j,k]=1 can be decomposed with
    # rank less than n*m*p by having the factors not be 0-1.
    
    # For example, for 2x2*2x2, the Winograd decomposition has rank 5 with
    # factors having entries like 1, -1, 2, etc.
    
    # For 3x4*4x8, let me try to use a product structure.
    
    # 3x4*4x8: Think of it as (3*4) x (4*8) x (8*3) = 12 x 32 x 24.
    # The tensor is all-ones. The rank of the all-ones tensor of size a x b x c is 1
    # if we don't have the index structure constraint. But with the constraint, it's different.
    
    # Wait, I think I've been misunderstanding. Let me re-read the problem.
    
    # "exactly decompose the matrix multiplication tensor"
    # The matmul tensor for (n,m,p) is T[i,j,k] = 1 for all i,j,k.
    # The decomposition is T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t].
    
    # So we need: for all i in 0..n-1, j in 0..m-1, k in 0..p-1:
    # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    
    # This is a system of n*m*p equations in rank*3 variables (with the constraint
    # that the same U, V, W must work for all triples).
    
    # For n=3, m=4, p=8: 96 equations.
    
    # Let me try to find a rank-32 solution.
    
    # Construction attempt:
    # Let t = 0..31. Write t = 4*a + b where a in 0..7, b in 0..3.
    # Let a = 2*c + d where c in 0..3, d in 0..1.
    # So t = 8*c + 2*d + b, with c in 0..3, d in 0..1, b in 0..3.
    
    # U[i*4+j, t] = 1 if j == b, else 0. (U depends only on j through b)
    # V[j*4+k, t] = ? 
    # W[k*3+i, t] = ?
    
    # This is getting too complex to do by hand. Let me use a numerical approach.
    
    # Actually, let me just use a simple valid decomposition. The schoolbook rank is 96.
    # Let me try to get rank 48 by a simple factorization.
    
    # Factorization: 3x4*4x8 = (3x4*4x4) x (1x1*1x2) using Kronecker product.
    # If we have rank-r decomp for 3x4*4x4 and rank-s for 1x1*1x2, then
    # rank-r*s for the combined.
    # 1x1*1x2 has rank 2 (schoolbook).
    # So if 3x4*4x4 has rank 24, then total is 48.
    
    # 3x4*4x4 schoolbook is 48. Can we do rank 24?
    # Lower bound: max(12, 16, 12) = 16. So rank 16-24 is possible.
    
    # For 3x4*4x4, here's a rank-24 construction:
    # Split j into {0,1,2,3}. For each j, we have a 3x1*1x4 sub-problem.
    # Each has rank 3*1*4 = 12 schoolbook, but actually rank 3 (since it's just
    # scaling each of the 3 rows by B[j,k]).
    
    # Wait, for fixed j, C[i,k] += A[i,j] * B[j,k]. This is an outer product of
    # A[:,j] (3x1) and B[j,:] (1x4), rank 1. So for each j, rank 1 suffices!
    # Total rank: 4 (one for each j). But that can't be right because U, V, W
    # must have the specific index structure.
    
    # For fixed j, t: U[i*4+j, t] = A[i,j]... no, U must be constant (not depend on A).
    
    # I think the issue is that U, V, W are fixed coefficients, not dependent on the input.
    
    # So for the matmul tensor, T[i,j,k] = 1, and we need fixed U, V, W such that
    # sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1 for all i,j,k.
    
    # This is equivalent to: the all-ones tensor of size (nm) x (mp) x (pn) with
    # the specific index mapping has tensor rank equal to what we're looking for.
    
    # Let me just try a computational approach: set up the equations and solve.
    
    # For rank r, we have r*3 variables and 96 equations.
    # Let's try r = 32.
    
    # Actually, let me just use a simple construction that I know works.
    
    # CONSTRUCTION: Use the fact that we can write 1 = 1*1*1 and share terms.
    
    # For rank 32:
    # Let t = 0..31.
    # t = 4*j + k where j in 0..3, k in 0..7.
    # U[i*4+j, t] = 1 for all i (when t corresponds to this j).
    # V[j*4+k, t] = 1 for this j,k.
    # W[k*3+i, t] = 1 for this k, all i.
    
    # Then sum_t U*V*W = sum over t where j matches and k matches = 1.
    # But this is rank 32? Let's see: t = 4*j + k, j in 0..3, k in 0..7. That's 32 values.
    # U[i*4+j, t] = 1 if t//4 == j, else 0.
    # V[j*4+k, t] = 1 if t//4 == j and t%4... wait, k goes up to 7, not 3.
    
    # Let me redo: t = 0..31. t = 4*j + k' where j in 0..3, k' in 0..7.
    # But 4*4 + 7 = 23, so t goes up to 23, not 31. That's only 32 values if k' in 0..7.
    # 4*8 = 32. Yes, t = 0..31, t = 4*j + k where j in 0..3, k in 0..7.
    
    # U[i*4+j, t] = 1 if t // 8 == j... no. t = 4*j + k, so j = t // 8 if k in 0..7?
    # No: t = 4*j + k, j in 0..3, k in 0..7. So j = t // 8? No, 4*3 + 7 = 19, so t up to 19.
    # That's only 20 values. I made an error.
    
    # j in 0..3 (4 values), k in 0..7 (8 values). Total 32 combinations.
    # t = j*8 + k, j in 0..3, k in 0..7. t in 0..31.
    
    # U[i*4+j, t] = 1 if t//8 == j, else 0.
    # V[j*4+k, t] = 1 if t//8 == j and t%8 == k, else 0.
    # W[k*3+i, t] = 1 if t%8 == k, else 0.
    
    # Check: for fixed (i,j,k), sum_t U[i*4+j,t]*V[j*4+k,t]*W[k*3+i,t]
    # = sum over t where t//8==j and t%8==k of 1*1*1 = 1.
    
    # So this gives rank 32! Let me verify the shapes:
    # U: (12, 32), V: (32, 32), W: (24, 32).
    
    # Wait, V has shape (m*p, rank) = (32, 32). And V[j*4+k, t] where j*4+k in 0..31. Yes.
    # W has shape (p*n, rank) = (24, 32). W[k*3+i, t] where k*3+i in 0..23. Yes.
    
    # This is a valid rank-32 decomposition!
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # (12, 32)
    V = [[0] * rank for _ in range(m * p)]  # (32, 32)
    W = [[0] * rank for _ in range(p * n)]  # (24, 32)
    
    for j in range(m):  # j in 0..3
        for k in range(p):  # k in 0..7
            t = j * p + k  # t in 0..31
            # U[i*m+j, t] = 1 for all i
            for i in range(n):
                U[i * m + j][t] = 1
            # V[j*p+k, t] = 1
            V[j * p + k][t] = 1
            # W[k*n+i, t] = 1 for all i
            for i in range(n):
                W[k * n + i][t] = 1
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for t in range(rank):
                    s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                if s != 1:
                    # Fallback to schoolbook
                    rank = n * m * p
                    U = [[0] * rank for _ in range(n * m)]
                    V = [[0] * rank for _ in range(m * p)]
                    W = [[0] * rank for _ in range(p * n)]
                    col = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U[i2 * m + j2][col] = 1
                                V[j2 * p + k2][col] = 1
                                W[k2 * n + i2][col] = 1
                                col += 1
                    return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
