"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time

def construct():
    """Rank-42 decomposition for 2x2 * 2x12 matrix multiplication.
    
    Partition 12 columns into 6 pairs of 2. Apply 7-product Strassen to each 2x2 block.
    Total rank = 6 * 7 = 42.
    
    Indexing conventions:
      U[i*m+j, r]: row = i*m+j (i in 0..n-1, j in 0..m-1) for A[i][j]
      V[j*p+k, r]: row = j*p+k (j in 0..m-1, k in 0..p-1) for B[j][k]
      W[k*n+i, r]: row = k*n+i (k in 0..p-1, i in 0..n-1) for C[i][k] (trace convention)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    num_pairs = p // 2  # 6
    
    # Strassen 7 products for 2x2 * 2x2:
    rank = num_pairs * 7  # 42
    U = [[0] * rank for _ in range(n * m)]   # 4 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 24 x 42
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # A element to U row: A[i][j] -> row i*m+j
    # B element to V row: B[j][k] -> row j*p+k
    # C element to W row: C[i][k] -> W[k*n+i]
    
    # 7 Strassen products: (A_coeffs, B_coeffs)
    products = [
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)]),
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)]),
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)]),
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)]),
    ]
    
    output_contribs = [
        [((0,0), 1), ((1,1), 1)],
        [((1,0), 1), ((1,1), -1)],
        [((0,1), 1), ((1,1), 1)],
        [((0,0), 1), ((1,0), 1)],
        [((0,0), -1), ((0,1), 1)],
        [((1,1), 1)],
        [((0,0), 1)],
    ]
    
    col = 0
    for t in range(num_pairs):
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
        base_col = 2 * t
        for s in range(7):
            for (ai, aj), coeff in products[s][0]:
                U[ai * m + aj][col] = coeff
            for (bj, bk_rel), coeff in products[s][1]:
                V[bj * p + base_col + bk_rel][col] = coeff
            for (ci, ck_rel), coeff in output_contribs[s]:
                W[(base_col + ck_rel) * n + ci][col] = coeff
            col += 1
    
    # Quick validation
    remaining = deadline - time.time()
    if remaining > 1.0:
        import random
        rng = random.Random(42)
        for _ in range(min(500, int(remaining * 50))):
            i = rng.randrange(n)
            j = rng.randrange(m)
            k = rng.randrange(p)
            dot = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
            if abs(dot - 1.0) > 1e-9:
                break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
