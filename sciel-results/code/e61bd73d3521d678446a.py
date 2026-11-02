"""
Matrix multiplication 2x4 by 4x7.
Uses block decomposition: A (2x4) = [A1 A2] where A1, A2 are 2x2.
B (4x7) = [B1; B2] where B1, B2 are 2x7.
C = A1*B1 + A2*B2, each 2x7.

For each 2x7 multiplication, use Strassen-style with 7 multiplications.
Each multiplication result is a 2x2 matrix (since 2x2 * 2x2 = 2x2).
Wait, no: 2x7 = 2x(7) = 2x(2*3 + 1). This doesn't split cleanly.

Let me reconsider. The known result is rank 50.
50 = 2 * 25? Or some other split.

Actually, let me think about this differently.
The tensor is for (2,4,7) multiplication.
Known bounds: rank >= 32 (Landsberg), rank <= 56 (schoolbook).

The "50" hint suggests a specific construction. Let me try:
- Split the 4 middle dimension into 2+2: two 2x2x7 sub-problems
- For each 2x2x7: we can use the fact that 2x2x2 has rank 7 (Strassen)
- 2x2x7 = 2x2x(3*2+1). Hmm.

Alternative: think of it as multiplying by a 4x7 matrix.
Split columns of B into groups.

Let me try a different approach: use the known rank-7 decomposition for 2x2x2,
and extend to 2x2x7 by treating the 7 as 3+2+2 or similar.

Actually, for 2x2xp, the rank is at most 3p-1 for p>=2 (using Strassen-like).
For p=7: rank <= 3*7-1 = 20 per 2x2 block? No, that's not right either.

Let me just implement the schoolbook (rank 56) which is guaranteed correct,
and try to find a better construction.

After more thought, I believe the rank-50 construction uses:
- 2x4x7 = combining a 2x2x7 part twice
- Each 2x2x7 has rank 25 (since 50/2 = 25)

For 2x2xp, rank = 2p+1 for p>=2? For p=7: 2*7+1 = 15. No, 15*2=30, not 50.

I'll go with schoolbook rank 56 as it's guaranteed valid.
"""

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    rank = n * m * p
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
