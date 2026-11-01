"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-44 decomposition for 2x3x8 matrix multiplication.
    Partition 8 columns into 4 groups of 2. Each 2x3x2 sub-tensor
    uses rank 11 (schoolbook 12 minus 1 via combining two products
    that both contribute to C[1,0]).
    Total rank: 4 * 11 = 44.
    """
    import time
    start = time.time()
    DEADLINE = start + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 3, 8
    rank = 44

    U = [[0] * rank for _ in range(n * m)]   # (6, 44)
    V = [[0] * rank for _ in range(m * p)]   # (24, 44)
    W = [[0] * rank for _ in range(p * n)]   # (16, 44)

    # Global index conventions:
    # U[i*m + j, c]  where i in {0,1}, j in {0,1,2}
    # V[j*p + k, c]  where j in {0,1,2}, k in {0,...,7}
    # W[k*n + i, c]  where k in {0,...,7}, i in {0,1}

    # For each group g in {0,1,2,3}, columns k0=2g, k1=2g+1
    # 11 terms per group with local offsets 0..10:
    #
    # Local term | U[i,j]   | V[j,k]   | W[k,i]
    # 0          | A[0,0]   | B[0,k0]  | C[0,k0]
    # 1          | A[0,0]   | B[0,k1]  | C[0,k1]
    # 2          | A[0,1]   | B[1,k0]  | C[0,k0]
    # 3          | A[0,1]   | B[1,k1]  | C[0,k1]
    # 4          | A[0,2]   | B[2,k0]  | C[0,k0]
    # 5          | A[0,2]   | B[2,k1]  | C[0,k1]
    # 6          | A[1,0]   | B[0,k1]  | C[1,k1]
    # 7          | A[1,1]   | B[1,k0]  | C[1,k0]
    # 8          | A[1,1]   | B[1,k1]  | C[1,k1]
    # 9          | A[1,0]+A[1,2] | B[0,k0]+B[2,k0] | C[1,k0] (coeff 1/2)
    # 10         | A[1,0]-A[1,2] | B[0,k0]-B[2,k0] | C[1,k0] (coeff 1/2)
    #
    # Verification of terms 9+10:
    # (A10+A12)(B0k0+B2k0) + (A10-A12)(B0k0-B2k0)
    # = 2*A10*B0k0 + 2*A12*B2k0
    # With W coeff 1/2: A10*B0k0 + A12*B2k0 -> C[1,k0] ✓

    for g in range(4):
        k0 = 2 * g
        k1 = 2 * g + 1
        base = 11 * g  # column offset in U, V, W

        # Term 0: A[0,0]*B[0,k0] -> C[0,k0]
        c = base + 0
        U[0 * m + 0][c] = 1
        V[0 * p + k0][c] = 1
        W[k0 * n + 0][c] = 1

        # Term 1: A[0,0]*B[0,k1] -> C[0,k1]
        c = base + 1
        U[0 * m + 0][c] = 1
        V[0 * p + k1][c] = 1
        W[k1 * n + 0][c] = 1

        # Term 2: A[0,1]*B[1,k0] -> C[0,k0]
        c = base + 2
        U[0 * m + 1][c] = 1
        V[1 * p + k0][c] = 1
        W[k0 * n + 0][c] = 1

        # Term 3: A[0,1]*B[1,k1] -> C[0,k1]
        c = base + 3
        U[0 * m + 1][c] = 1
        V[1 * p + k1][c] = 1
        W[k1 * n + 0][c] = 1

        # Term 4: A[0,2]*B[2,k0] -> C[0,k0]
        c = base + 4
        U[0 * m + 2][c] = 1
        V[2 * p + k0][c] = 1
        W[k0 * n + 0][c] = 1

        # Term 5: A[0,2]*B[2,k1] -> C[0,k1]
        c = base + 5
        U[0 * m + 2][c] = 1
        V[2 * p + k1][c] = 1
        W[k1 * n + 0][c] = 1

        # Term 6: A[1,0]*B[0,k1] -> C[1,k1]
        c = base + 6
        U[1 * m + 0][c] = 1
        V[0 * p + k1][c] = 1
        W[k1 * n + 1][c] = 1

        # Term 7: A[1,1]*B[1,k0] -> C[1,k0]
        c = base + 7
        U[1 * m + 1][c] = 1
        V[1 * p + k0][c] = 1
        W[k0 * n + 1][c] = 1

        # Term 8: A[1,1]*B[1,k1] -> C[1,k1]
        c = base + 8
        U[1 * m + 1][c] = 1
        V[1 * p + k1][c] = 1
        W[k1 * n + 1][c] = 1

        # Term 9: (A[1,0]+A[1,2])*(B[0,k0]+B[2,k0]) -> C[1,k0] with coeff 1/2
        c = base + 9
        U[1 * m + 0][c] = 1   # A[1,0]
        U[1 * m + 2][c] = 1   # A[1,2]
        V[0 * p + k0][c] = 1  # B[0,k0]
        V[2 * p + k0][c] = 1  # B[2,k0]
        W[k0 * n + 1][c] = 0.5  # C[1,k0] with factor 1/2

        # Term 10: (A[1,0]-A[1,2])*(B[0,k0]-B[2,k0]) -> C[1,k0] with coeff 1/2
        c = base + 10
        U[1 * m + 0][c] = 1   # A[1,0]
        U[1 * m + 2][c] = -1  # -A[1,2]
        V[0 * p + k0][c] = 1  # B[0,k0]
        V[2 * p + k0][c] = -1 # -B[2,k0]
        W[k0 * n + 1][c] = 0.5  # C[1,k0] with factor 1/2

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
