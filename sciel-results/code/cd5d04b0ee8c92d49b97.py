"""Matrix multiplication 2x4 by 4x8 tensor decomposition.

We construct a rank-48 exact decomposition by applying a rank-6 algorithm
for 2x2x2 (a known optimal decomposition) to each of the 8 independent
(2 intermediate rows x 4 output column blocks) sub-problems.

The rank-6 decomposition for 2x2x2:
  m1 = (A00+A11)(B00+B11)
  m2 = (A10+A11)B00
  m3 = A00(B01-B11)
  m4 = A11(B10-B00)
  m5 = (A00+A01)B11
  m6 = (A10-A00)(B00+B01)
  m7 = (A01-A11)(B10+B11)
  
Wait, that's 7. Let me use the standard Strassen 7-product algorithm.
Total rank = 8 * 7 = 56.

Actually, let me reconsider. The task asks for rank-minimal. For 2x4x8,
the trivial decomposition gives 2*4*8 = 64. Using Strassen on 2x2 blocks
gives 8 * 7 = 56.

Can we do better? The tensor 2x4x8 can be viewed as two copies of 2x2x8
or four copies of 2x2x4 or eight copies of 2x2x2.

For 2x2x4, the rank is known to be 10 (by Landsberg's bound and explicit constructions).
So 4 * 10 = 40 would be better than 56.

Let me construct a rank-10 decomposition for 2x2x4.

Actually, let me think about this differently. The matrix multiplication
tensor for (n,m,p) has rank at most n*m*p. For (2,4,8), that's 64.

Using block structure: partition the middle dimension into two blocks of
size 2. Then C = A * B where A is 2x4 and B is 4x8.

C[i,k] = sum_j A[i,j] * B[j,k]

Partition j into j=0,1 and j=2,3:
C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]

This is two independent 2x2x8 multiplications.

For 2x2x8, we can use Strassen's algorithm. The rank of 2x2x8 is at most
7*4 = 28 (using 4 independent 2x2x2 blocks in the output dimension).
But actually, we can do better by exploiting the structure.

Let me use a different approach. I'll use the fact that 2x4x8 = 2x2x8 ⊕ 2x2x8.

For 2x2x8: A is 2x2, B is 2x8, C is 2x8.
C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]

This can be decomposed using 4 rank-1 terms for each output column k, giving
rank 4*2 = 8 per 2x2x2 block, total 4*8 = 32 for 2x2x8.

But we can use Strassen: for each pair of output columns, we use 7 products.
So for 2x2x8, we have 4 pairs, giving 4*7 = 28.

Total for 2x4x8: 2 * 28 = 56.

Alternatively, for 2x2x8, we can view it as multiplying a 2x2 matrix by
a 2x8 matrix. The rank is at most 2*2*8/2 = 16? No, that's not right.

Actually, the bilinear rank of 2x2x8 is at most 7*4 = 28 using Strassen
on each 2x2x2 block of the output.

So total rank = 2 * 28 = 56.

Let me just use the rank-56 decomposition with Strassen's 7-product algorithm.
"""

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS

    # Strassen's 7 products for 2x2 * 2x2.
    # Each entry: (U_coeffs, V_coeffs, W_coeffs)
    #   U_coeffs: list of (i, j, c)  -- coefficient c on A[i,j]
    #   V_coeffs: list of (j, k, c)  -- coefficient c on B[j,k]
    #   W_coeffs: list of (k, i, c)  -- coefficient c on C[k,i]  (i=row, k=col)
    #
    # Product index r in 0..6
    strassen_products = [
        # r=0: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # r=1: (A10+A11)B00 -> C10 += 1, C11 -= 1
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1), (1,1,-1)]),
        # r=2: A00(B01-B11) -> C01 += 1, C11 += 1
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
        # r=3: A11(B10-B00) -> C00 += 1, C10 += 1
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # r=4: (A00+A01)B11 -> C00 -= 1, C01 += 1
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (1,0,1)]),
        # r=5: (A10-A00)(B00+B01) -> C11 += 1
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # r=6: (A01-A11)(B10+B11) -> C00 += 1
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]

    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for ar in range(2):          # intermediate row block  (2 blocks of size 2)
        for qc in range(4):       # output column block    (4 blocks of size 2)
            for u_coeff, v_coeff, w_coeff in strassen_products:
                for li, lj, c in u_coeff:
                    gi = li
                    gj = 2 * ar + lj
                    U[gi * m + gj][col] += c
                for lj, lk, c in v_coeff:
                    gj = 2 * ar + lj
                    gk = 2 * qc + lk
                    V[gj * p + gk][col] += c
                for lk, li, c in w_coeff:
                    gk = 2 * qc + lk
                    gi = li
                    W[gk * n + gi][col] += c
                col += 1

    if time.time() > deadline:
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
