import time
DIMENSIONS = [2, 5, 6]

def construct():
    t0 = time.time()
    dl = t0 + 130.0
    n, m, p = DIMENSIONS

    # We use a block-partition Kronecker decomposition.
    # Split m=5 into m1=1, m2=4.
    # For each block (i-range, j-range, k-range), we use a rank-7 Strassen-like
    # tensor for 2x2x2, or rank-8 direct for other small blocks, but here we
    # only need blocks of size 2 (in i) and 2 (in j) and 2 (in k) to apply Strassen.
    #
    # Strategy:
    #  - j=0: 1 column in A, 1 row in B -> 2x1x6: direct rank 12.
    #  - j=1,2,3,4: 4 columns in A, 4 rows in B -> 2x4x6.
    #    Split j into two blocks of 2: (1,2) and (3,4).
    #    Split k into three blocks of 2: (0,1), (2,3), (4,5).
    #    Each 2x2x2 block uses Strassen rank 7.
    #    Total: 2 * 3 * 7 = 42.
    #  - Total rank: 12 + 42 = 54.
    #
    # However, we can try to reduce the direct part.
    #  For j=0: 2x1x6. Can we do better than 12?
    #  The tensor T_{i,j,k} = delta_{j,0} * A_{i,j} * B_{j,k}.
    #  This is just A_{i,0} * B_{0,k}. So C_{i,k} = A_{i,0} * B_{0,k}.
    #  This is a rank-1 tensor in (i,k) for each fixed... no, it's a product of
    #  a vector in i and a vector in k. So the tensor is A_col * B_row^T.
    #  The rank of this tensor (as a 3D tensor) is... actually, the decomposition
    #  C_{i,k} = A_{i,0} * B_{0,k} can be written as sum over r of U_{i,r} * V_{r,k} * W_{r,?}
    #  but we need to match the 3-way tensor. The standard direct decomposition
    #  for 2x1x6 is rank 12 (one term per (i,k) pair). Can we do less?
    #  The tensor is separable: T_{i,j,k} = a_i * b_k * delta_{j,0}.
    #  So T = u ⊗ v ⊗ w where u_i = delta_{i,?}... no, u is a vector in i, v in j, w in k.
    #  T_{i,j,k} = (A_{i,0}) * (delta_{j,0}) * (B_{0,k}).
    #  This is a rank-1 tensor! So rank 1 suffices.
    #  Wait, but the factor matrices must satisfy:
    #  C_{i,k} = sum_r U_{i*m + j, r} * V_{j*p + k, r} * W_{k*n + i, r}
    #  For j=0: C_{i,k} = sum_r U_{i*5 + 0, r} * V_{0*6 + k, r} * W_{k*2 + i, r}
    #  We want this to equal A_{i,0} * B_{0,k}.
    #  Let rank=1. Set U_{i*5+0, 0} = A_{i,0}, V_{k, 0} = B_{0,k}, W_{k*2+i, 0} = 1.
    #  But U, V, W must be fixed matrices, not depending on A or B!
    #  So we cannot do rank 1 for the direct part. The decomposition must be
    #  independent of the input matrices. The direct decomposition has one term
    #  for each (i, j, k) triple, with U_{i*m+j, r} = 1 if (i,j) matches, etc.
    #  For 2x1x6, we need 2*1*6 = 12 terms. So rank 12 is minimal for direct.
    #
    #  Actually, we can use the Strassen decomposition on a larger block that
    #  includes j=0. For example, split m=5 into 2+3. Then j=0,1 form a 2x2x2
    #  block with k=0,1. But we need to cover all k. This gets complicated.
    #
    #  Let's try a different split: m=5 into 2+2+1.
    #  Block 1: j=0,1 (2 cols), k=0,1,2,3,4,5.
    #    Split k into three 2-blocks. Each 2x2x2 uses Strassen rank 7. Total: 21.
    #  Block 2: j=2,3 (2 cols), k=0,1,2,3,4,5.
    #    Same: 21.
    #  Block 3: j=4 (1 col), k=0,1,2,3,4,5.
    #    Direct: 12.
    #  Total: 21 + 21 + 12 = 54. Same.
    #
    #  What if we use a 2x2x3 or 2x3x2 Strassen-like decomposition?
    #  The rank of 2x2x3 matrix multiplication is at most 2*2*3 = 12, but can be
    #  lower. Known: rank(2x2x3) = 10 (I think). Let's assume we can find a
    #  decomposition of rank 10 for 2x2x3.
    #
    #  Alternative: Use the fact that 2x5x6 can be decomposed as 2x2x6 + 2x3x6.
    #  Rank(2x2x6) + Rank(2x3x6).
    #  Rank(2x2x6) = 2 * Rank(2x2x2) + ... no, 6=3*2, so 3 blocks of 2x2x2: 3*7=21.
    #  Rank(2x3x6): 3=1+2. 2x1x6 (12) + 2x2x6 (21) = 33. Total: 21+33=54.
    #
    #  It seems 54 is hard to beat with simple block decompositions.
    #  Let's try to verify that 54 is correct and just output it cleanly.

    total_rank = 54
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    col = 0

    # Part 1: j=0, direct 2x1x6
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Part 2: j=1..4, Strassen on 2x2x2 blocks
    strassen = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    for jb in range(2):
        if time.time() > dl:
            return {"U": U, "V": V, "W": W}
        j0 = 1 + jb * 2
        for kb in range(3):
            k0 = kb * 2
            for uc, vc, wc in strassen:
                for li, lj, v in uc:
                    U[(li) * m + (j0 + lj)][col] += v
                for lj, lk, v in vc:
                    V[(j0 + lj) * p + (k0 + lk)][col] += v
                for li, lk, v in wc:
                    W[(k0 + lk) * n + li][col] += v
                col += 1

    return {"U": U, "V": V, "W": W}
