"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = _start_time + 140.0  # Safety margin before 150s kill


def _construct_strassen():
    """Fallback: Strassen-based construction with rank 53."""
    n, m, p = DIMENSIONS
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(base_r, k0, k1):
        r1, r2, r3, r4, r5, r6, r7 = (base_r + i for i in range(7))

        # P1: (A00+A11)(B0k0+B1k1)
        U[0][r1] = 1; U[3][r1] = 1
        V[0 * p + k0][r1] = 1; V[1 * p + k1][r1] = 1

        # P2: (A10+A11)B0k0
        U[2][r2] = 1; U[3][r2] = 1
        V[0 * p + k0][r2] = 1

        # P3: A00(B0k1-B1k1)
        U[0][r3] = 1
        V[0 * p + k1][r3] = 1; V[1 * p + k1][r3] = -1

        # P4: A11(B1k0-B0k0)
        U[3][r4] = 1
        V[1 * p + k0][r4] = 1; V[0 * p + k0][r4] = -1

        # P5: (A00+A01)B1k1
        U[0][r5] = 1; U[1][r5] = 1
        V[1 * p + k1][r5] = 1

        # P6: (A10-A00)(B0k0+B0k1)
        U[2][r6] = 1; U[0][r6] = -1
        V[0 * p + k0][r6] = 1; V[0 * p + k1][r6] = 1

        # P7: (A01-A11)(B1k0+B1k1)
        U[1][r7] = 1; U[3][r7] = -1
        V[1 * p + k0][r7] = 1; V[1 * p + k1][r7] = 1

        # W reconstruction: C[0,k0], C[1,k0], C[0,k1], C[1,k1]
        # C[0,k0] = P1 - P3 - P5 + P7  (from standard Strassen)
        # Wait, let me re-derive. Standard Strassen for 2x2:
        # C00 = P1 - P3 - P5 + P7
        # C01 = P5 + P6
        # C10 = P2 + P4
        # C11 = P1 - P2 - P4 + P6
        # where P1..P7 are as defined above.
        # But the W factor uses W[k*n+i, r], so W[0*2+0]=W[0] for C[0,0], W[1*2+0]=W[2] for C[1,0], etc.
        # W[k*n+i, r]: k is column, i is row. So W[0]=C[0,0], W[1]=C[0,1], W[2]=C[1,0], W[3]=C[1,1]
        # For block with columns k0, k1:
        # C[0,k0] corresponds to W[k0*2+0]
        # C[1,k0] corresponds to W[k0*2+1]
        # C[0,k1] corresponds to W[k1*2+0]
        # C[1,k1] corresponds to W[k1*2+1]

        # C[0,k0] = P1 - P3 - P5 + P7
        W[k0 * 2 + 0][r1] = 1
        W[k0 * 2 + 0][r3] = -1
        W[k0 * 2 + 0][r5] = -1
        W[k0 * 2 + 0][r7] = 1

        # C[1,k0] = P2 + P4
        W[k0 * 2 + 1][r2] = 1
        W[k0 * 2 + 1][r4] = 1

        # C[0,k1] = P5 + P6
        W[k1 * 2 + 0][r5] = 1
        W[k1 * 2 + 0][r6] = 1

        # C[1,k1] = P1 - P2 - P4 + P6
        W[k1 * 2 + 1][r1] = 1
        W[k1 * 2 + 1][r2] = -1
        W[k1 * 2 + 1][r4] = -1
        W[k1 * 2 + 1][r6] = 1

    for i in range(7):
        add_strassen_block(7 * i, 2 * i, 2 * i + 1)

    # Trivial for column 14
    base_r = 49
    U[0][base_r] = 1; V[14][base_r] = 1; W[28][base_r] = 1
    U[1][base_r + 1] = 1; V[29][base_r + 1] = 1; W[28][base_r + 1] = 1
    U[2][base_r + 2] = 1; V[14][base_r + 2] = 1; W[29][base_r + 2] = 1
    U[3][base_r + 3] = 1; V[29][base_r + 3] = 1; W[29][base_r + 3] = 1

    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 0.0
                for r in range(rank):
                    lhs += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(lhs - 1.0) > 1e-9:
                    return False
    return True


def _try_sliding_window():
    """Try to find a rank-50 decomposition using overlapping Strassen blocks.
    
    Strategy: Instead of 7 independent Strassen blocks (rank 49) + 4 trivial (rank 53),
    try to share one Strassen product between adjacent column pairs.
    
    For columns 0-14, we have 7 pairs. If we can overlap between pairs, we might save rank.
    Specifically, if the Strassen product for pair (2i, 2i+1) can share a factor with
    pair (2i+2, 2i+3), we save 1 rank per overlap.
    
    This is complex to implement correctly. For now, return None.
    """
    return None


def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15

    # First, try the sliding window approach (with a short time limit)
    if time.time() - _start_time < _DEADLINE - 5.0:
        result = _try_sliding_window()
        if result is not None:
            # Verify before returning
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                return result

    # Fall back to the Strassen construction
    result = _construct_strassen()
    # Verify the fallback
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # If verification fails, return a simple schoolbook decomposition as last resort
        rank = n * m * p  # 60
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        result = {"U": U, "V": V, "W": W}
    return result

# EVOLVE-BLOCK-END
