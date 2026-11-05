"""Matrix multiplication 6x6x6 via 2x2 block decomposition of 3x3 blocks
with BCLR rank-21 decomposition for each 3x3 block multiplication.
Total rank: 4 block positions * 21 = 84.

The BCLR (Bini-Capovani-Lotti-Romani) decomposition gives rank 21 for 3x3
matrix multiplication, which is the best known exact decomposition.

The 21 multiplication terms in BCLR are defined as follows:
We partition the 3x3 matrices into blocks and use 21 scalar multiplications.

Actually, let me use the known BCLR rank-21 decomposition. The 21 terms are:

For 3x3 matrix multiplication C = A * B:
The 21 products M_k = (sum of A entries) * (sum of B entries)

The BCLR decomposition uses these 21 terms:
M1  = A[0,0] * B[0,0]
M2  = A[0,1] * B[1,0]
M3  = A[0,2] * B[2,0]
M4  = A[1,0] * B[0,1]
M5  = A[1,1] * B[1,1]
M6  = A[1,2] * B[2,1]
M7  = A[2,0] * B[0,2]
M8  = A[2,1] * B[1,2]
M9  = A[2,2] * B[2,2]
M10 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
M11 = (A[0,1]+A[1,2]) * (B[1,0]+B[2,1])
M12 = (A[1,0]+A[2,2]) * (B[0,1]+B[2,2])
M13 = (A[0,2]+A[2,0]) * (B[2,0]+B[0,2])
M14 = (A[0,1]+A[1,0]) * (B[0,1]+B[1,0])
M15 = (A[1,2]+A[2,1]) * (B[1,2]+B[2,1])
M16 = (A[0,0]+A[2,2]) * (B[0,2]+B[2,0])
M17 = (A[0,1]+A[2,2]) * (B[1,0]+B[2,2])  -- hmm, this doesn't look right
...

Actually, I don't have the exact BCLR coefficients memorized. Let me use a different approach.

Since I can't reliably recall the exact BCLR rank-21 coefficients, let me use the
Strassen approach at the block level which I already have working at rank 189,
and see if I can improve it.

Current approach: 2x2 blocks of 3x3, Strassen at block level (7 terms), schoolbook
within each 3x3 (27 terms per product) = 7*27 = 189.

Alternative: Use schoolbook at block level (8 terms) with schoolbook within = 8*27 = 216.
Strassen is better at block level.

Can I do better? What about using a rank-19 or rank-20 decomposition for 3x3?
Pan's algorithm gives rank 19 for 3x3. Let me try to implement that.

Actually, the known results:
- 3x3: rank 19 (Pan 1978), rank 21 (BCLR 1978)
- The rank 19 is better!

But I need the exact coefficients for Pan's rank-19 algorithm.

Pan's 19-term decomposition for 3x3 matrix multiplication:
The 19 products are:
P1  = A00*B00
P2  = A01*B00  
P3  = A02*B00
P4  = A10*B00
P5  = A11*B00
P6  = A12*B00
P7  = A20*B00
P8  = A21*B00
P9  = A22*B00
P10 = (A00+A11+B11)*B01  -- no, that's not right either

I don't have the exact Pan coefficients. Let me just keep the Strassen approach at
rank 189 which I know works correctly.

Wait - actually, I can try a different block partition. What if I use 3x2 blocks?
6x6 = 2x3 grid of 3x2 blocks? No, that doesn't work for matrix multiplication.

What about 6x6 = 3x2 grid of 2x3 blocks? 
A is 6x6, partitioned as 3 rows of 2x3 blocks, 2 cols of 3x2 blocks... no this gets messy.

Let me just stick with the working Strassen-at-block-level approach at rank 189.
"""

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    bs = 3  # block size

    # Strassen decomposition at the 2x2 block level (each entry is a 3x3 matrix)
    # Block indices: 0=TL, 1=TR, 2=BL, 3=BR
    # Block (bi): row offset = (bi // 2) * bs, col offset = (bi % 2) * bs

    # Strassen 7 virtual products:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)

    # Each term: (list of (block_idx, coeff) for A, list of (block_idx, coeff) for B)
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # P1
        ([(2, 1), (3, 1)], [(0, 1)]),           # P2
        ([(0, 1)], [(1, 1), (3, -1)]),          # P3
        ([(3, 1)], [(2, 1), (0, -1)]),          # P4
        ([(0, 1), (1, 1)], [(3, 1)]),           # P5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),  # P6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),  # P7
    ]

    # Output reconstruction:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    # output_contrib[term_idx] = list of (output_block_idx, coeff)
    output_contrib = [
        [(0, 1), (3, 1)],    # P1 -> C00(+1), C11(+1)
        [(2, 1), (3, -1)],   # P2 -> C10(+1), C11(-1)
        [(1, 1), (3, 1)],    # P3 -> C01(+1), C11(+1)
        [(0, 1), (2, 1)],    # P4 -> C00(+1), C10(+1)
        [(0, -1), (1, 1)],   # P5 -> C00(-1), C01(+1)
        [(3, 1)],            # P6 -> C11(+1)
        [(0, 1)],            # P7 -> C00(+1)
    ]

    # Total rank: 7 Strassen terms * 27 schoolbook terms per 3x3 product = 189
    rank = 7 * 27

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for term_idx in range(7):
        a_comb, b_comb = strassen_terms[term_idx]
        for i in range(bs):   # local row in 3x3 block
            for k in range(bs):  # local col in A / local row in B
                for j in range(bs):  # local col in B / local col in C
                    # U contributions: A entries
                    for (blk_a, coeff_a) in a_comb:
                        row_a = (blk_a // 2) * bs
                        col_a = (blk_a % 2) * bs
                        u_idx = (row_a + i) * m + (col_a + k)
                        U[u_idx][col] += coeff_a

                    # V contributions: B entries
                    for (blk_b, coeff_b) in b_comb:
                        row_b = (blk_b // 2) * bs
                        col_b = (blk_b % 2) * bs
                        v_idx = (row_b + k) * p + (col_b + j)
                        V[v_idx][col] += coeff_b

                    # W contributions: C entries (W index = global_col * n + global_row)
                    for (blk_c, coeff_c) in output_contrib[term_idx]:
                        row_c = (blk_c // 2) * bs
                        col_c = (blk_c % 2) * bs
                        w_idx = (col_c + j) * n + (row_c + i)
                        W[w_idx][col] += coeff_c

                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
