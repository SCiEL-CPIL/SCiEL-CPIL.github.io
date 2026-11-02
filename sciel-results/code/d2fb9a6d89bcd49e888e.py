"""Matrix multiplication 2x3x3 rank-minimal tensor decomposition.

We use the known algebraic structure for a low-rank decomposition.
For a 2x3x3 matrix product, we can exploit the fact that multiplying a 2x3 matrix by a 3x3 matrix
can be viewed as two 1x3 times 3x3 products, which have rank 9 each, but we can share some structure.

However, the most straightforward improvement over the schoolbook rank 18 is to use the
Strassen-like or other algebraic identities. For 2x3x3, one can argue that rank 15 is achievable
by combining the two 1x3x3 multiplications with a shared 3x3x3 core in a clever way.

But to be safe and correct, let's implement a known low-rank decomposition.
A classic result: the rank of multiplying a 2xn by nxm matrix is at most 2nm - n - m + (n-1)(m-1)? 
Not exactly. Let's use a constructive approach.

Actually, for 2x3x3, we can use the following:
The product C = A * B where A is 2x3 and B is 3x3.
C[i,k] = sum_j A[i,j] * B[j,k] for i in 0,1 and k in 0,1,2.

We can group the terms. One way to reduce rank is to use the identity:
For two rows, we can use 2*3*3 = 18 terms, but we can combine some.

A known low-rank algorithm for 2xnxm matrices uses rank 2nm - n - m + 1 if n,m > 1? 
Let me just implement a correct rank-15 decomposition if I can find one, otherwise use a safe lower rank.

Actually, I recall that for 2x3x3, the rank is 15. Here's a construction:
We can think of the 2x3x3 multiplication as:
C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]

We can use the following 15 rank-1 terms:
Let's define:
For the first row of C:
  Use the standard 3 terms for k=0,1,2? No, that's 3 terms per row, 6 total, plus we need to share.

Actually, a better approach: use the "baby-step giant-step" or just hardcode a known decomposition.

Since I don't have a verified rank-15 decomposition in memory, I'll use a different strategy:
Use the fact that the multiplication tensor for (2,3,3) has rank at most 15 by the following argument:
The space of 2x3 matrices has dimension 6, 3x3 has dimension 9, and the product is 2x3 (dim 6).
The trivial rank is 18. We can save 3 terms by using the identity that for the two rows, we can combine the B matrix terms.

Let me just implement a rank-15 decomposition based on a known result or a safe construction.

After some thought, I'll use a rank-15 decomposition that I can verify. Here's a construction:

Let the 15 rank-1 terms be indexed by (i, j, k) but with some sharing.

Actually, I found a reference in my mind: the rank of the (2,3,3) multiplication is 15.
Here's a concrete decomposition:

We'll use the following 15 terms:
1-3: For C[0,0], C[0,1], C[0,2] using A[0,0]*B[0,*], A[0,1]*B[1,*], A[0,2]*B[2,*] -> 3 terms
4-6: For C[1,0], C[1,1], C[1,2] using A[1,0]*B[0,*], A[1,1]*B[1,*], A[1,2]*B[2,*] -> 3 terms
That's 6 terms, not 15. I'm confused.

Let me restart with a clear approach. I'll use a safe rank that I know is correct and lower than 18.
The minimum rank for 2x3x3 is actually 15, but I need to be sure.

I'll implement a rank-15 decomposition using the following method:
We can write the 2x3x3 multiplication as a 1x3x3 multiplication for each row, but we can share the B matrix.
Specifically, we can use 3 terms for the B matrix (one for each row of B) and then for each of the 2 rows of A, we need 3 terms, but we can combine.

Actually, I'll use a different strategy: I'll use the rank-15 decomposition from the literature.
Since I can't recall it exactly, I'll use a rank-16 decomposition as a safe bet, but I want to aim for 15.

Let me just output a correct rank-15 decomposition if I can construct one, otherwise I'll use a rank-18 with a note.

After careful thought, I'll use the following rank-15 decomposition:

The 15 terms are:
- 3 terms for the "diagonal" of B (B[0,0], B[1,1], B[2,2]) shared between both rows of A
- 6 terms for the off-diagonal of B (B[0,1], B[0,2], B[1,0], B[1,2], B[2,0], B[2,1]) but split between rows
- 6 more terms for the A matrix elements

This is getting too vague. I'll use a known correct rank-15 decomposition.

Here's a verified rank-15 decomposition for 2x3x3:

Let U, V, W be 15-dimensional.

I'll use the following 15 rank-1 components:

1. A[0,0]*B[0,0] -> C[0,0]
2. A[0,0]*B[0,1] -> C[0,1]
3. A[0,0]*B[0,2] -> C[0,2]
4. A[0,1]*B[1,0] -> C[0,0]
5. A[0,1]*B[1,1] -> C[0,1]
6. A[0,1]*B[1,2] -> C[0,2]
7. A[0,2]*B[2,0] -> C[0,0]
8. A[0,2]*B[2,1] -> C[0,1]
9. A[0,2]*B[2,2] -> C[0,2]
10. A[1,0]*B[0,0] -> C[1,0]
11. A[1,0]*B[0,1] -> C[1,1]
12. A[1,0]*B[0,2] -> C[1,2]
13. A[1,1]*B[1,0] -> C[1,0]
14. A[1,1]*B[1,1] -> C[1,1]
15. A[1,1]*B[1,2] -> C[1,2]

Wait, this only covers up to B[1,2] for the second row, and I'm missing B[2,*] for the second row.
This is rank 15 but it's incomplete. I need 18 terms for the standard, and I can't just drop 3.

I think I'm overcomplicating. Let me just use the schoolbook rank 18 but with a clean implementation, and then I'll try to reduce it.

Actually, I found a better approach: use the rank-15 decomposition that I can verify by construction.

I'll output a rank-15 decomposition based on the following:
The 2x3x3 multiplication can be done with 15 scalar multiplications using the following algorithm:
Let P1 = A[0,0]*(B[0,0]+B[1,0]+B[2,0])
Let P2 = A[0,1]*(B[0,1]+B[1,1]+B[2,1])
Let P3 = A[0,2]*(B[0,2]+B[1,2]+B[2,2])
Let P4 = A[1,0]*(B[0,0]+B[1,0]+B[2,0])
Let P5 = A[1,1]*(B[0,1]+B[1,1]+B[2,1])
Let P6 = A[1,2]*(B[0,2]+B[1,2]+B[2,2])
Then we need to combine these, but this doesn't directly give the result.

I give up on finding a rank-15 in my head. I'll use a rank-18 decomposition but with a clean, correct implementation.

Actually, I just remembered: the rank of 2x3x3 is 15, and here's a concrete decomposition:

Let's use the following 15 rank-1 terms:
- 5 terms for the first row of C
- 5 terms for the second row of C
- 5 shared terms

But I can't recall the exact coefficients.

I'll use a safe rank-18 decomposition.

"""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
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
# EVOLVE-BLOCK-END
