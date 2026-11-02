"""Matrix multiplication 2x2x15: rank-minimal exact tensor decomposition."""
import time

_START_TIME = time.time()
_TIME_LIMIT = 110.0

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # Rank 53 decomposition: 7 Strassen blocks (14 cols) + 4 schoolbook (1 col)
    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53
    
    # A indices in U: 0:A00, 1:A01, 2:A10, 3:A11
    # B indices in V: j*p + k
    # C indices in W: k*n + i
    
    def build_strassen(k0, k1):
        """Yields the 7 Strassen factor vectors (u, v, w) for a 2x2 block on columns k0,k1."""
        # P1: (A00+A11)(B0k0+B1k1)
        yield ([1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 0, 0])
        # P2: (A10+A11)B0k0
        yield ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1])
        # P3: A00(B0k1-B1k1)
        yield ([1, 0, 0, 0], [0, 1, -1, 0], [0, 0, 1, 1])
        # P4: A11(B1k0-B0k0)
        yield ([0, 0, 0, 1], [-1, 0, 1, 0], [0, 0, 1, 1])
        # P5: (A00+A01)B1k1
        yield ([1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1])
        # P6: (A10-A00)(B0k0+B0k1)
        yield ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 0])
        # P7: (A01-A11)(B1k0+B1k1)
        yield ([0, 1, 0, -1], [0, 1, 1, 0], [1, 0, 0, 0])
    
    def emit_block(start_r, k0, k1):
        for i, (u, v, w) in enumerate(build_strassen(k0, k1)):
            r = start_r + i
            for a_idx, val in enumerate(u):
                U[a_idx][r] = val
            # v is indexed by [j*p+k]; we store flat index
            for flat_idx, val in enumerate(v):
                V[flat_idx][r] = val
            # w is indexed by [k*n+i] -> flat index
            for flat_idx, val in enumerate(w):
                W[flat_idx][r] = val
    
    # 7 Strassen blocks for column pairs (0,1), (2,3), ..., (12,13)
    for i in range(7):
        emit_block(7*i, 2*i, 2*i+1)
    
    # Trivial schoolbook block for column 14 (rank 4)
    # C0,14 = A00*B0,14 + A01*B1,14
    # C1,14 = A10*B0,14 + A11*B1,14
    base_r = 49
    # Factor 1: A00 * B0,14 -> C0,14
    U[0][base_r] = 1
    V[0*p+14][base_r] = 1
    W[14*n+0][base_r] = 1
    
    # Factor 2: A01 * B1,14 -> C0,14
    U[1][base_r+1] = 1
    V[1*p+14][base_r+1] = 1
    W[14*n+0][base_r+1] = 1
    
    # Factor 3: A10 * B0,14 -> C1,14
    U[2][base_r+2] = 1
    V[0*p+14][base_r+2] = 1
    W[14*n+1][base_r+2] = 1
    
    # Factor 4: A11 * B1,14 -> C1,14
    U[3][base_r+3] = 1
    V[1*p+14][base_r+3] = 1
    W[14*n+1][base_r+3] = 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
