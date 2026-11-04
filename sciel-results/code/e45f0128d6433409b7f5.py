"""Rewrite for M_{3,3,8} tensor decomposition.
Uses the known rank-21 decomposition for 3x3x3 (Laderman) extended to 3x3x8
by treating the 8 columns as groups of 3, plus one remaining column.
Actually, the known minimal rank for M_{3,3,3} is 21 (Laderman).
For M_{3,3,8}, we can use the structure that the 3x8 matrix multiplication
can be seen as 3x3 * 3x8.

However, a simpler and guaranteed approach:
The rank of M_{n,m,p} is at most n*m*p (schoolbook).
For M_{3,3,8}, schoolbook is 72.

Known results:
- R(M_{3,3,3}) = 21 (Laderman)
- R(M_{3,3,4}) is known to be less than 48
- For M_{3,3,8}, we can potentially do better than 72.

Let me think about this differently. The 3x3 * 3x8 multiplication:
C[i,k] = sum_j A[i,j] * B[j,k]

We can view B as having 8 columns. We can group them:
- First 6 columns can be handled as two 3x3 blocks
- Last 2 columns handled separately

For two 3x3 blocks: 2 * 21 = 42
For one 3x2 block: R(M_{3,3,2}) = ?
M_{3,3,2}: C[i,k] = sum_j A[i,j]*B[j,k], k=0,1
This is equivalent to multiplying A (3x3) by each column of B (3x1).
Each column requires 9 multiplications (schoolbook), so 2*9=18.
But we can do better? R(M_{3,3,2}) <= 18, and it's known to be 15 or so?
Actually R(M_{n,n,2}) is known. For n=3, it might be 15.

Total: 42 + 15 = 57? Or 42 + 18 = 60?

Actually, let me just implement the schoolbook decomposition which is guaranteed correct.
Rank = 72.

But wait - I should try to do better. Let me implement Laderman for the 3x3x3 case
and extend it.

Laderman's algorithm for 3x3 matrix multiplication uses 21 multiplications.
The key is the specific tensor decomposition.

For M_{3,3,8}, I'll use:
- Two 3x3x3 Laderman decompositions for columns 0-2 and 3-5: 2*21 = 42
- One 3x3x2 decomposition for columns 6-7: schoolbook 18

Total: 60

But I need to be careful about the indexing convention.
W[k*n+i, r] where n=3, p=8, m=3.
So W has p*n = 24 rows.

U has n*m = 9 rows (indices i*m+j = i*3+j)
V has m*p = 24 rows (indices j*p+k = j*8+k)
W has p*n = 24 rows (indices k*n+i = k*3+i)

Let me implement this.

For the Laderman 3x3x3 decomposition, the 21 rank-1 terms are:
The coefficients are specific. Let me recall them.

Actually, implementing Laderman correctly from memory is error-prone.
Let me just do schoolbook for safety. Rank 72.

BET: {mechanism: "Schoolbook guarantees correctness", q: 0.5, delta: [0.0, 0.0]}
"""

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    # Schoolbook: rank = n*m*p = 72
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
