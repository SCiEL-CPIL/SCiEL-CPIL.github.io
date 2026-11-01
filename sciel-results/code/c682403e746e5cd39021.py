"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    t_start = time.time()
    DEADLINE = 120.0  # safety margin before 150s kill

    # Block decomposition: 7 pairs of columns using Strassen (rank 7 each)
    # + 1 single column using schoolbook (rank 4)
    # Total rank = 7 * 7 + 4 = 53

    rank = 53
    U = [[0.0] * rank for _ in range(n * m)]      # 4 x 53
    V = [[0.0] * rank for _ in range(m * p)]      # 30 x 53
    W = [[0.0] * rank for _ in range(p * n)]      # 30 x 53

    # A indices in U: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices in V: j*p + k, j in {0,1}, k in {0..14}
    # C indices in W: k*n + i, k in {0..14}, i in {0,1}

    def add_strassen_block(base_r, k0, k1):
        """Add Strassen decomposition for column pair (k0, k1)."""
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6

        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0][r1] += 1.0
        U[3][r1] += 1.0
        V[0*p + k0][r1] += 1.0
        V[1*p + k1][r1] += 1.0

        # M2 = (A10 + A11) * B0k0
        U[2][r2] += 1.0
        U[3][r2] += 1.0
        V[0*p + k0][r2] += 1.0

        # M3 = A00 * (B0k1 - B1k1)
        U[0][r3] += 1.0
        V[0*p + k1][r3] += 1.0
        V[1*p + k1][r3] += -1.0

        # M4 = A11 * (B1k0 - B0k0)
        U[3][r4] += 1.0
        V[1*p + k0][r4] += 1.0
        V[0*p + k0][r4] += -1.0

        # M5 = (A00 + A01) * B1k1
        U[0][r5] += 1.0
        U[1][r5] += 1.0
        V[1*p + k1][r5] += 1.0

        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[2][r6] += 1.0
        U[0][r6] += -1.0
        V[0*p + k0][r6] += 1.0
        V[0*p + k1][r6] += 1.0

        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[1][r7] += 1.0
        U[3][r7] += -1.0
        V[1*p + k0][r7] += 1.0
        V[1*p + k1][r7] += 1.0

        # Reconstruction:
        # C0k0 = M1 + M4 - M5 + M7
        W[2*k0 + 0][r1] += 1.0
        W[2*k0 + 0][r4] += 1.0
        W[2*k0 + 0][r5] += -1.0
        W[2*k0 + 0][r7] += 1.0

        # C1k0 = M2 + M4
        W[2*k0 + 1][r2] += 1.0
        W[2*k0 + 1][r4] += 1.0

        # C0k1 = M3 + M5
        W[2*k1 + 0][r3] += 1.0
        W[2*k1 + 0][r5] += 1.0

        # C1k1 = M1 - M2 + M3 + M6
        W[2*k1 + 1][r1] += 1.0
        W[2*k1 + 1][r2] += -1.0
        W[2*k1 + 1][r3] += 1.0
        W[2*k1 + 1][r6] += 1.0

    # Add 7 Strassen blocks for column pairs (0,1), (2,3), ..., (12,13)
    for i in range(7):
        if time.time() - t_start > DEADLINE:
            break
        k0 = 2 * i
        k1 = 2 * i + 1
        base_r = 7 * i
        add_strassen_block(base_r, k0, k1)

    # Add schoolbook block for column 14
    base_r = 49
    k_last = 14

    r1 = base_r
    r2 = base_r + 1
    r3 = base_r + 2
    r4 = base_r + 3

    # Q1 = A00 * B0k_last -> C0k_last
    U[0][r1] += 1.0
    V[0*p + k_last][r1] += 1.0
    W[2*k_last + 0][r1] += 1.0

    # Q2 = A01 * B1k_last -> C0k_last
    U[1][r2] += 1.0
    V[1*p + k_last][r2] += 1.0
    W[2*k_last + 0][r2] += 1.0

    # Q3 = A10 * B0k_last -> C1k_last
    U[2][r3] += 1.0
    V[0*p + k_last][r3] += 1.0
    W[2*k_last + 1][r3] += 1.0

    # Q4 = A11 * B1k_last -> C1k_last
    U[3][r4] += 1.0
    V[1*p + k_last][r4] += 1.0
    W[2*k_last + 1][r4] += 1.0

    result = {"U": U, "V": V, "W": W}

    # Quick verification with random inputs
    if time.time() - t_start < DEADLINE - 5.0:
        rng = np.random.RandomState(42)
        A = rng.rand(n, m)
        B = rng.rand(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        # C[i,k] = sum_r U[i,m+k_a] * V[j*p+k_b] * W[k*n+i] ... 
        # Actually: C = (U^T * V) contracted with W
        # For each r: outer(U[:,r], V[:,r]) gives A*B contribution, then *W[:,r] gives C
        # More precisely: C[i,k] = sum_r sum_a sum_b U[a,r]*V[b,r]*W[k*n+i,r] where a=(i,j), b=(j,k)
        # U indexed by (i,j) -> index i*m+j
        # V indexed by (j,k) -> index j*p+k  
        # W indexed by (k,i) -> index k*n+i
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C_computed[i, k] += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback: return schoolbook rank-60
            rank_sb = n * m * p
            U_sb = [[0.0] * rank_sb for _ in range(n * m)]
            V_sb = [[0.0] * rank_sb for _ in range(m * p)]
            W_sb = [[0.0] * rank_sb for _ in range(p * n)]
            r_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i*m+j][r_idx] = 1.0
                        V_sb[j*p+k][r_idx] = 1.0
                        W_sb[k*n+i][r_idx] = 1.0
                        r_idx += 1
            result = {"U": U_sb, "V": V_sb, "W": W_sb}

    return result
# EVOLVE-BLOCK-END
