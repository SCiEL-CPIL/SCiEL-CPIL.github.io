"""
Rank-minimal exact tensor decomposition for 4x4 by 4x5 matrix multiplication.

Approach: Use a block decomposition with Strassen's algorithm for 2x2 blocks.
Split the 4x4x5 problem into 2x2 block multiplications.

The 4x4 matrix A is split into 2x2 blocks:
A = [[A11, A12],
     [A21, A22]]

The 4x5 matrix B is split into 2x2 blocks (with the last column treated specially):
B = [[B11, B12, B13],
     [B21, B22, B23]]
where B11, B21 are 2x2, B12, B22 are 2x2, B13, B23 are 2x1.

Actually, let me use a cleaner approach: treat the 4x5 as having 5 columns, and use
a known rank reduction for the 4x4x4 part combined with the 4x4x1 part.

For the 4x4x4 part, I'll use a construction based on the fact that we can reduce the
rank below 64 using algebraic identities. One known approach uses the "Bini-Capovani"
or similar constructions, but for simplicity and correctness, let me use a direct
construction with rank 56 for the 4x4x4 part (if I can find one) plus 16 for the 4x4x1.

Actually, let me use a simpler guaranteed improvement: use the schoolbook but with
some sharing. For 4x4x5, the schoolbook gives 80. 

Let me try a different approach: use the fact that matrix multiplication can be
represented more efficiently using the "transpose" trick or other algebraic identities.

Given the complexity, let me return a correct rank-80 decomposition first, and then
try to improve it.
"""

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Use schoolbook decomposition: rank = n*m*p = 80
    # This is correct but not minimal. Let me try to do better.
    
    # For now, return the schoolbook which is guaranteed correct
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
