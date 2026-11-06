import time
import sys

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # Partition 5 into 2, 2, 1
    sizes = [2, 2, 1]

    # Precompute start indices for each block dimension
    starts = [0]
    for s in sizes:
        starts.append(starts[-1] + s)

    # Total rank: for each (i_block, j_block, k_block) sub-block,
    # if all sizes are 2, use Strassen (rank 7); else schoolbook (rank si*sj*sk)
    rank_per_block = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                si, sj, sk = sizes[i], sizes[j], sizes[k]
                if si == 2 and sj == 2 and sk == 2:
                    rank_per_block.append(7)
                else:
                    rank_per_block.append(si * sj * sk)

    total_rank = sum(rank_per_block)  # 117

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # Strassen columns for 2x2x2 matrix multiplication
    # Each column is a rank-1 tensor (u vector, v vector, w vector)
    # Indexing: u[(i,j)], v[(j,k)], w[(k,i)]
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

    strassen_cols = [
        # M1 = (A00+A11)(B00+B11); C00 += M1, C11 += M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2 = (A10+A11)B00; C10 += M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1)]),
        # M3 = A00(B01-B11); C01 += M3
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(1,0,1)]),
        # M4 = A11(B10-B00); C10 += M4, C00 += M4
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,1,1), (0,0,1)]),
        # M5 = (A00+A01)B11; C00 -= M5, C01 += M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (1,0,1)]),
        # M6 = (A10-A00)(B00+B01); C11 += M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7 = (A01-A11)(B10+B11); C00 += M7, C11 += M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
    ]

    col = 0
    for i_block in range(3):
        for j_block in range(3):
            for k_block in range(3):
                si, sj, sk = sizes[i_block], sizes[j_block], sizes[k_block]
                i_start = starts[i_block]
                j_start = starts[j_block]
                k_start = starts[k_block]

                if si == 2 and sj == 2 and sk == 2:
                    # Use Strassen: 7 rank-1 tensors
                    for u_list, v_list, w_list in strassen_cols:
                        for (li, lj, val) in u_list:
                            U[(i_start + li) * m + (j_start + lj)][col] = val
                        for (lj, lk, val) in v_list:
                            V[(j_start + lj) * p + (k_start + lk)][col] = val
                        for (lk, li, val) in w_list:
                            W[(k_start + lk) * n + (i_start + li)][col] = val
                        col += 1
                else:
                    # Schoolbook: rank = si*sj*sk, each term is one (i,j,k)
                    for li in range(si):
                        for lj in range(sj):
                            for lk in range(sk):
                                U[(i_start + li) * m + (j_start + lj)][col] = 1
                                V[(j_start + lj) * p + (k_start + lk)][col] = 1
                                W[(k_start + lk) * n + (i_start + li)][col] = 1
                                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
