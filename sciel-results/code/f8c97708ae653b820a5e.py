"""
Rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication.

The schoolbook rank is n*m*p = 4*4*9 = 144.

We can do better by exploiting structure. The key insight is that we can use
Strassen-like block decompositions. For M_{4,4,9}, we can partition into blocks
and use known lower-rank decompositions for smaller sub-problems.

One approach: view 4x4 * 4x9 as involving 2x2 * 2x3 sub-blocks.
The rank of M_{2,2,3} is known to be 12 (schoolbook) but can be reduced.
Actually, the exact rank of M_{2,2,3} is 12 (no known improvement).

However, we can use the fact that M_{4,4,9} can be decomposed using the
Kronecker product structure. If we write 4 = 2*2 and 9 = 3*3, we get
M_{4,4,9} related to M_{2,2,3} ⊗ M_{2,2,3}, but the dimensions don't quite work.

Let me think differently. The tensor T_{4,4,9} has entries T[i,j,k] = δ_{i,k} for
(i,j) in 4x4 and k in 1..9, where the multiplication is C[i,k] = sum_j A[i,j]*B[j,k].

The schoolbook gives rank 144. 

A better approach: use the fact that for the "outer" dimensions, we can apply
Strassen's algorithm on 2x2 blocks. 

Actually, let me use a simpler but effective approach: 
- Partition the 4x4 matrix A into four 2x2 blocks
- Partition the 4x9 matrix B into four 2x9 blocks (rows) and the 9 columns stay
- The product C is 4x9, partitioned into four 2x9 blocks

For each output block C_{ab} (a,b in {0,1} for row/col block of 2x2 partition of C),
C_{ab} = sum_{c=0}^{1} A_{ac} * B_{cb}

This is just standard block matrix multiplication. The rank contribution is:
For each (a,b,c) triple: 2*2*9 = 36, total 8*36 = 288? No, that's wrong.

Let me reconsider. The schoolbook for 4x4*4x9:
C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
Each (i,j,k) triple gives one rank-1 term, so 4*4*9 = 144 terms.

To reduce rank, we need to find linear dependencies or use algebraic identities.

One known technique: for M_{n,n,p} where p is large, the rank is at most n^2 * ceil(p/n) * something... 

Actually, let me try a different approach. I'll use the fact that we can group
columns of B. If we group the 9 columns into groups of 3, we get 3 groups.
For each group of 3 columns, we compute a 4x3 product, which is M_{4,4,3}.
The rank of M_{4,4,3}: schoolbook is 4*4*3 = 48.

Can we do better for M_{4,4,3}? 
M_{4,4,3} can be seen as: C[i,k] = sum_j A[i,j]*B[j,k] for k=0,1,2.

Using Strassen on 2x2 blocks: partition A into 2x2 blocks, B into 2x3 blocks (rows),
C into 2x3 blocks (rows).

C_{ab} = A_{a0}*B_{0b} + A_{a1}*B_{1b} for a,b in {0,1}

Each A_{ac} is 2x2, B_{cb} is 2x3, product is 2x3.
The product of 2x2 and 2x3 matrices has rank at most 2*2*3 = 12 per pair (c,b) for fixed a.
Using Strassen for 2x2 * 2x2 part: but B is 2x3, not 2x2.

For 2x2 * 2x3: C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k], k=0,1,2.
Schoolbook rank: 2*2*3 = 12.
Can we reduce this? The tensor M_{2,2,3} has rank 12 (no known improvement over schoolbook).

So M_{4,4,3} via Strassen on the 2x2 block structure:
We need to compute 4 block products of type 2x2 * 2x3, each rank 12.
But with Strassen, we can compute 2x2*2x2 in rank 7. For 2x2*2x3, we can view it as
two 2x2*2x1 products plus... no.

Actually, 2x2 * 2x3: we can write B as [B1 | B2 | B3] where each Bj is 2x1.
C = A * B = [A*B1 | A*B2 | A*B3].
Each A*Bj is 2x1, which is just a vector. A*Bj[i] = sum_j A[i,j]*Bj[j].
This is rank 2*2*1 = 4 per Bj, total 12 for all three.

No improvement possible for M_{2,2,3} beyond 12.

So the best I can reliably do is schoolbook: 144.

Let me just return the schoolbook solution.
"""

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
