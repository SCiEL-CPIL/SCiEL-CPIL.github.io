# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction

def construct():
    """Rank-33 decomposition for 2x2 * 2x12 matrix multiplication.
    
    Partition 12 columns into 3 groups of 4.
    For each 2x2x4 block:
    - Use 7 Strassen products for the first 2 columns (2x2x2 part).
    - Use 4 additional products for the last 2 columns (direct multiplication).
    Total rank per block = 11.
    Total rank = 3 * 11 = 33.
    
    Indexing conventions:
      U[i*m+j, r]: row = i*m+j (i in 0..n-1, j in 0..m-1) for A[i][j]
      V[j*p+k, r]: row = j*p+k (j in 0..m-1, k in 0..p-1) for B[j][k]
      W[k*n+i, r]: row = k*n+i (k in 0..p-1, i in 0..n-1) for C[i][k] (trace convention)
    """
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS
    num_blocks = p // 4
    rank = num_blocks * 11
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen products for 2x2x2
    products_2x2 = [
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)]),
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)]),
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)]),
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)]),
    ]
    
    output_contribs_2x2 = [
        [((0,0), 1), ((1,1), 1)],
        [((1,0), 1), ((1,1), -1)],
        [((0,1), 1), ((1,1), 1)],
        [((0,0), 1), ((1,0), 1)],
        [((0,0), -1), ((0,1), 1)],
        [((1,1), 1)],
        [((0,0), 1)],
    ]
    
    # Additional products for the 2 extra columns (indices 2 and 3 in the 2x2x4 block)
    # These are direct multiplications for C[i, 2] and C[i, 3]
    # C[i, 2] = A[i, 0] B[0, 2] + A[i, 1] B[1, 2]
    # C[i, 3] = A[i, 0] B[0, 3] + A[i, 1] B[1, 3]
    # We can create 4 products:
    # M8 = A[0,0]B[0,2] + A[0,1]B[1,2] -> C[0,2]
    # M9 = A[1,0]B[0,2] + A[1,1]B[1,2] -> C[1,2]
    # M10 = A[0,0]B[0,3] + A[0,1]B[1,3] -> C[0,3]
    # M11 = A[1,0]B[0,3] + A[1,1]B[1,3] -> C[1,3]
    
    products_2x2_4 = products_2x2 + [
        ([((0,0), 1), ((0,1), 1)], [((0,2), 1), ((1,2), 1)]), # M8: A row 0, B col 2
        ([((1,0), 1), ((1,1), 1)], [((0,2), 1), ((1,2), 1)]), # M9: A row 1, B col 2
        ([((0,0), 1), ((0,1), 1)], [((0,3), 1), ((1,3), 1)]), # M10: A row 0, B col 3
        ([((1,0), 1), ((1,1), 1)], [((0,3), 1), ((1,3), 1)]), # M11: A row 1, B col 3
    ]
    
    output_contribs_2x2_4 = output_contribs_2x2 + [
        [((2, 0), 1)], # C[0,2] from M8
        [((2, 1), 1)], # C[1,2] from M9
        [((3, 0), 1)], # C[0,3] from M10
        [((3, 1), 1)], # C[1,3] from M11
    ]
    
    col = 0
    for t in range(num_blocks):
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
        base_col = 4 * t
        for s in range(11):
            # U contributions
            for (ai, aj), coeff in products_2x2_4[s][0]:
                U[ai * m + aj][col] = coeff
            # V contributions
            for (bj, bk_rel), coeff in products_2x2_4[s][1]:
                b_col = base_col + bk_rel
                V[bj * p + b_col][col] = coeff
            # W contributions
            for (ck, ci), coeff in output_contribs_2x2_4[s]:
                c_col = base_col + ck
                W[c_col * n + ci][col] = coeff
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
