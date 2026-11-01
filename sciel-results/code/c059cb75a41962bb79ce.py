"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Verify the tensor decomposition: T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]"""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a][d] * V[b][d] * W[c][d] for d in range(rank))
                if val != 1:
                    return False
    # Also verify that non-matching entries are 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    for jp in range(m):
                        for kp in range(p):
                            if (i, j, k) != (ip, jp, kp):
                                a = i * m + j
                                b = jp * p + kp
                                c = k * n + ip
                                val = sum(U[a][d] * V[b][d] * W[c][d] for d in range(rank))
                                if val != 0:
                                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin

    n, m, p = DIMENSIONS  # 2, 3, 8

    # Block decomposition strategy:
    # A (2x3) has 2 columns, each a 3-vector = (2x1) + (1x1)
    #   a_i = (a_i0, a_i1, a_i2) = (a_i0, a_i1, a_i2)
    #   Represent as blocks: A_i = [a_i0; a_i1] (2x1) and a_i2 (scalar)
    # B (3x8) has 3 rows, each a 8-vector = (4x1) + (4x1)
    #   b_j = (b_j0,...,b_j7) = (b_j0..b_j3, b_j4..b_j7)
    #   Represent as blocks: B_j = [b_j0..b_j3] (1x4) and [b_j4..b_j7] (1x4)
    #
    # Product: c_i,k = sum_j a_i,j * b_j,k
    #   = sum_j (a_i0*b_j0 + a_i1*b_j1 + a_i2*b_j2)  for k in 0..3
    #   + sum_j (a_i0*b_j4 + a_i1*b_j5 + a_i2*b_j6)  for k in 4..7
    #
    # This doesn't quite work as a clean block matrix mult. Let me try differently.
    #
    # Better approach: view as 2x2 block structure.
    # Split A's 3 columns into (2,1): A = [A_L (2x2) | A_R (2x1)]
    # Split B's 3 rows into (2,1): B = [B_T (2x8); B_B (1x8)]
    # Then C = A_L * B_T + A_R * B_B
    #
    # A_L * B_T is 2x2 * 2x8 = 2x8 (rank 2*2*8=32 schoolbook)
    # A_R * B_B is 2x1 * 1x8 = 2x8 (rank 2*1*8=16 schoolbook)
    # Total: 48. No savings.
    #
    # Alternative: Use Strassen-like decomposition.
    # For 2x2 * 2x2, Strassen uses 7 products instead of 8.
    #
    # Let me try: split into blocks where I can apply Strassen.
    # A (2x3) -> treat as 2x2 * 2x2 with padding? No, 3 is odd.
    #
    # New approach: Use the fact that 8 = 2*4.
    # Split B's 8 columns into 4 groups of 2.
    # For each group of 2 columns, we have a 3x2 sub-matrix.
    # C[:, 2t:2t+2] = A * B[:, 2t:2t+2]  (2x3 * 3x2 = 2x2)
    #
    # If I can do 2x3 * 3x2 in rank 11, then 4 groups * 11 = 44.
    # If I can do it in rank 10, then 40.
    #
    # Let me try to find a rank-11 algorithm for 2x3 * 3x2.
    #
    # Actually, let me try a completely different structural approach:
    # Use the "doubling" trick.
    #
    # For 2x3 * 3x8:
    # c_i,k = sum_j a_i,j * b_j,k
    #
    # Define for each j: s_j = a_0,j, t_j = a_0,j + a_1,j
    # Then a_1,j = t_j - s_j
    #
    # For each (j,k): need a_0,j*b_j,k and a_1,j*b_j,k
    # = s_j*b_j,k and (t_j-s_j)*b_j,k
    #
    # If I compute s_j*b_j,k and t_j*b_j,k, I get both.
    # That's 2*3*8 = 48 terms. Same as schoolbook.
    #
    # To save, I need to share computations across k.
    # For fixed j, I need s_j * b_j,0..7 and t_j * b_j,0..7.
    # That's 16 products for 3 values of j = 48.
    #
    # Alternative: for fixed j, compute only t_j * b_j,k for all k,
    # and s_j * b_j,k for all k. Still 16 per j.
    #
    # What if I use linear combinations of the b's?
    # For fixed j, define 8 products with different linear combos of a's.
    #
    # Let me try: for each j, use 8 products:
    #   p_{j,k} = (a_0,j + alpha_k * a_1,j) * b_{j,k}
    # This gives a_0,j*b_{j,k} + alpha_k * a_1,j*b_{j,k}
    # I need both a_0,j*b_{j,k} and a_1,j*b_{j,k} separately,
    # so I need at least 2 products per (j,k) pair. 3*8*2 = 48.
    #
    # Unless I can share across j. But the a's are different for each j.
    #
    # Let me try yet another approach: use the structure of the problem.
    # The tensor for 2x3*3x8 has dimension 6*6*16 = 576 entries.
    # Schoolbook uses 48 rank-1 terms.
    #
    # Known result: rank of 2x3*3x2 is 11 (or 12?).
    # If rank(2x3*3x2) = 11, then tiling gives rank(2x3*3x8) <= 44.
    #
    # Let me construct a rank-11 algorithm for 2x3 * 3x2.
    #
    # A = [[a00, a01, a02],
    #      [a10, a11, a12]]
    # B = [[b00, b01],
    #      [b10, b11],
    #      [b20, b21]]
    # C = [[c00, c01],
    #      [c10, c11]]
    #
    # c00 = a00*b00 + a01*b10 + a02*b20
    # c01 = a00*b01 + a01*b11 + a02*b21
    # c10 = a10*b00 + a11*b10 + a12*b20
    # c11 = a10*b01 + a11*b11 + a12*b21
    #
    # 12 products needed schoolbook. Can we do 11?
    #
    # Try:
    # t0 = a00*b00
    # t1 = a00*b01
    # t2 = a01*b10
    # t3 = a01*b11
    # t4 = a02*b20
    # t5 = a02*b21
    # t6 = (a00+a10)*b00
    # t7 = (a00+a10)*b01
    # t8 = (a01+a11)*b10
    # t9 = (a01+a11)*b11
    # t10 = (a02+a12)*b20
    #
    # c00 = t0 + t2 + t4
    # c01 = t1 + t3 + t5
    # c10 = (t6-t0) + (t8-t2) + (t10-t4)
    # c11 = (t7-t1) + (t9-t3) + ???  need a12*b21
    #
    # t10 = (a02+a12)*b20 = a02*b20 + a12*b20
    # t10 - t4 = a12*b20. Good for c10.
    # But for c11, I need a12*b21. I don't have it.
    #
    # If I use t10 = (a02+a12)*b21 instead:
    # c10 = (t6-t0) + (t8-t2) + ???  need a12*b20
    # c11 = (t7-t1) + (t9-t3) + (t10-t5)
    #
    # Still missing one.
    #
    # What if I use a different combination?
    # t10 = (a02+a12)*(b20+b21)
    # Then t10 = a02*b20 + a02*b21 + a12*b20 + a12*b21
    # t10 - t4 - t5 = a12*b20 + a12*b21
    # This contributes to both c10 and c11, which is wrong.
    #
    # I think rank 11 might not be achievable for 2x3*3x2.
    # Let me just use rank 12 (schoolbook) for the 2x3*3x2 block.
    # 4 blocks * 12 = 48. Same as schoolbook.
    #
    # Let me try a different split. What about 2x3*3x4?
    # If I can find a better algorithm for that, 2*rank(2x3*3x4) might help.
    #
    # Actually, let me just go with schoolbook rank 48, which is correct.
    # The problem asks for rank-minimal, and 48 is the schoolbook upper bound.
    # Without a proven better algorithm, 48 is the safe answer.

    rank = n * m * p  # 48
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

    # Verify
    assert _verify(U, V, W, n, m, p, rank)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
